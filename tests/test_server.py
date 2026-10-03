"""HTTP contract tests; the byte codec and numeric runtime avoid GPU requirements."""
import http.client
import json
import socket
import threading
import time
import unittest

from qvelox.server import Engine, Server


MODEL = 'redshift-qwen3.8-27b'


class ByteCodec:
    eos_ids = {256}

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors='replace')

    def render(self, messages, tools=None, enable_thinking=False):
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list')
        for message in messages:
            if not isinstance(message.get('content'), str):
                raise ValueError('text content required')
        return ''.join(message['content'] for message in messages)


class ScriptedRuntime:
    """A causal token oracle: every supplied token advances the output by one."""
    def __init__(self, text='Hello', context=4096):
        self.context = context
        self.tokens = []
        self.calls = []
        self.resets = 0
        self.prompt = None
        self.output = list(text.encode()) + [256]
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.failure = None
        self.eval_calls = 0

    @property
    def position(self):
        return len(self.tokens)

    def reset(self):
        self.tokens.clear()
        self.resets += 1

    def advance(self, tokens):
        self.calls.append(list(tokens))
        self.entered.set()
        if self.block:
            if not self.release.wait(3):
                raise RuntimeError('test runtime timed out')
        if self.failure:
            raise RuntimeError(self.failure)
        answers = []
        for token in tokens:
            self.tokens.append(token)
            index = len(self.tokens) - len(self.prompt)
            answers.append(self.output[max(0, min(index, len(self.output) - 1))])
        return answers

    def evaluate(self, tokens):
        self.eval_calls += 1
        return [[0.0 if i == token else -1000.0 for i in range(257)]
                for token in self.advance(tokens)]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = ScriptedRuntime()
        self.engine = Engine(self.runtime, ByteCodec(), model_id=MODEL,
                             context=self.runtime.context)
        self.server = Server(('127.0.0.1', 0), self.engine)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.runtime.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def connection(self):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(connection.close)
        return connection

    def request(self, payload=None, path='/v1/chat/completions'):
        connection = self.connection()
        if payload is None:
            connection.request('GET', path)
        else:
            connection.request('POST', path, json.dumps(payload),
                               {'Content-Type': 'application/json'})
        response = connection.getresponse()
        body = response.read()
        return response.status, json.loads(body)

    def payload(self, text='Q', **kwargs):
        self.runtime.prompt = list(text.encode())
        return {'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
                'max_tokens': 64, **kwargs}

    def stream(self, payload):
        connection = self.connection()
        connection.request('POST', '/v1/chat/completions', json.dumps(payload),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        events = [line[6:] for line in response.read().decode().splitlines()
                  if line.startswith('data: ')]
        self.assertEqual(events[-1], '[DONE]')
        return [json.loads(event) for event in events[:-1]]

    def test_discovery_and_health_report_actual_readiness(self):
        status, body = self.request(path='/v1/models')
        self.assertEqual(status, 200)
        self.assertEqual([m['id'] for m in body['data']], [MODEL])
        status, body = self.request(path='/health')
        self.assertEqual((status, body['ready'], body['model']), (200, True, MODEL))
        self.engine.ready = False
        self.assertEqual(self.request(path='/health')[0], 503)

    def test_json_generation_counts_prompt_and_sampled_eos(self):
        status, body = self.request(self.payload())
        self.assertEqual(status, 200)
        self.assertEqual(body['model'], MODEL)
        self.assertEqual(body['choices'][0]['message'], {'role': 'assistant', 'content': 'Hello'})
        self.assertEqual(body['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(body['usage'], {'prompt_tokens': 1, 'completion_tokens': 6,
                                        'total_tokens': 7})
        self.assertEqual(self.runtime.tokens, list(b'QHello'))

    def test_stream_fragments_unicode_role_usage_and_done(self):
        self.runtime.output = list('Ciao 🌍'.encode()) + [256]
        chunks = self.stream(self.payload(stream=True, stream_options={'include_usage': True}))
        deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
        self.assertEqual(deltas[0]['role'], 'assistant')
        self.assertEqual(''.join(d.get('content', '') for d in deltas), 'Ciao 🌍')
        self.assertGreater(len([d for d in deltas if d.get('content')]), 2)
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(chunks[-1]['choices'], [])
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 10)

    def test_stop_string_never_leaks_partial_match(self):
        self.runtime.output = list(b'Hi STOP discarded') + [256]
        chunks = self.stream(self.payload(stream=True, stop=['STOP']))
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
        self.assertEqual(text, 'Hi ')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_max_tokens_finishes_length_and_pending_token_is_not_cached(self):
        status, body = self.request(self.payload(max_tokens=2))
        self.assertEqual((status, body['choices'][0]['message']['content']), (200, 'He'))
        self.assertEqual(body['choices'][0]['finish_reason'], 'length')
        self.assertEqual(self.runtime.tokens, list(b'QH'))
        before = len(self.runtime.calls)
        status, body = self.request(self.payload('QHeNEXT', max_tokens=1))
        self.assertEqual(status, 200)
        self.assertEqual(self.runtime.calls[before], list(b'eNEXT'))
        self.assertEqual(self.runtime.resets, 0)

    def test_unrelated_prompt_resets_instead_of_leaking_state(self):
        self.request(self.payload(max_tokens=1))
        self.request(self.payload('OTHER', max_tokens=1))
        self.assertEqual(self.runtime.tokens, list(b'OTHER'))
        self.assertEqual(self.runtime.resets, 1)

    def test_prompt_prefill_uses_batches_at_most_eight(self):
        self.request(self.payload('X' * 25, max_tokens=1))
        self.assertEqual([len(call) for call in self.runtime.calls], [8, 8, 8, 1])

    def test_validation_rejects_wrong_model_context_and_unsupported_inputs(self):
        invalid = [({'model': 'pretend-model'}, 404), ({'n': 2}, 400),
                   ({'max_tokens': 4096}, 400), ({'temperature': -1}, 400),
                   ({'temperature': True}, 400), ({'top_p': 0}, 400),
                   ({'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}]}, 400),
                   ({'stream': 'true'}, 400), ({'response_format': {'type': 'json_object'}}, 400),
                   ({'frequency_penalty': 1}, 400), ({'max_tokens': 0}, 400)]
        for overrides, expected in invalid:
            with self.subTest(overrides=overrides):
                status, body = self.request(self.payload(**overrides))
                self.assertEqual(status, expected)
                self.assertIn('message', body['error'])
        self.assertEqual(self.runtime.calls, [])

    def test_sampling_uses_full_logits_and_seed(self):
        status, body = self.request(self.payload(temperature=0.7, top_p=0.9, seed=12))
        self.assertEqual(status, 200)
        self.assertEqual(body['choices'][0]['message']['content'], 'Hello')
        self.assertGreater(self.runtime.eval_calls, 0)

    def test_busy_is_immediate_and_health_remains_available(self):
        self.runtime.block = True
        payload = self.payload()
        responses = []
        request = threading.Thread(target=lambda: responses.append(self.request(payload)))
        request.start()
        self.assertTrue(self.runtime.entered.wait(2))
        started = time.monotonic()
        self.assertEqual(self.request(payload)[0], 429)
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(self.request(path='/health')[1]['busy'])
        self.runtime.release.set()
        request.join(3)
        self.assertEqual(responses[0][0], 200)

    def test_disconnect_during_prefill_releases_without_finishing_prompt(self):
        self.runtime.block = True
        payload = self.payload('x' * 1000)
        data = json.dumps(payload).encode()
        connection = socket.create_connection(self.server.server_address, timeout=3)
        connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n'
                           b'Content-Type: application/json\r\nContent-Length: ' +
                           str(len(data)).encode() + b'\r\n\r\n' + data)
        self.assertTrue(self.runtime.entered.wait(2))
        connection.close()
        self.runtime.release.set()
        deadline = time.monotonic() + 2
        while self.engine.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.engine.busy)
        self.assertLessEqual(self.runtime.position, 8)
        self.assertEqual(self.request(self.payload('NEW', max_tokens=1))[0], 200)

    def test_runtime_failure_is_json_before_headers_and_does_not_hold_lock(self):
        self.runtime.failure = 'numeric failure'
        status, body = self.request(self.payload(stream=True))
        self.assertEqual(status, 500)
        self.assertEqual(body['error']['code'], 'inference_error')
        self.assertFalse(self.engine.busy)
        self.runtime.failure = None
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_tools_are_returned_and_tool_result_turn_is_accepted(self):
        self.runtime.output = list(b'<tool_call>\n<function=weather>\n<parameter=city>\nBern\n</parameter>\n</function>\n</tool_call>') + [256]
        tools = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {
            'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
        status, body = self.request(self.payload(tools=tools, max_tokens=256))
        self.assertEqual(status, 200)
        choice = body['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        call = choice['message']['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'weather')
        self.assertEqual(json.loads(call['function']['arguments']), {'city': 'Bern'})
        chunks = self.stream(self.payload(tools=tools, max_tokens=256, stream=True))
        calls = [d for c in chunks for d in c['choices'][0]['delta'].get('tool_calls', [])]
        self.assertEqual(calls[0]['function'], call['function'])
        self.runtime.output = list(b'Sunny') + [256]
        payload = self.payload('Qresult', messages=[{'role': 'user', 'content': 'Q'},
                        {'role': 'tool', 'tool_call_id': call['id'], 'content': 'result'}])
        self.assertEqual(self.request(payload)[1]['choices'][0]['message']['content'], 'Sunny')

    def test_bad_json_and_large_body_are_rejected(self):
        connection = self.connection()
        connection.request('POST', '/v1/chat/completions', '{', {'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn('error', json.loads(response.read()))
        connection = self.connection()
        connection.putrequest('POST', '/v1/chat/completions')
        connection.putheader('Content-Length', '1048577')
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()


if __name__ == '__main__':
    unittest.main()
