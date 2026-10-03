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

    def token_bytes(self, token):
        return bytes([token])

    def render(self, messages, tools=None, enable_thinking=False, reasoning_effort='xhigh'):
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

    def test_reasoning_effort_enables_thinking_in_json_and_stream(self):
        from test_text import TextCodec, tiny_metadata

        self.engine.codec = TextCodec(tiny_metadata())
        for effort, instruction in (('low', 'Keep your thinking brief and focused'),
                                    ('medium', None),
                                    ('xhigh', 'Please think carefully through the task')):
            with self.subTest(effort=effort):
                payload = self.payload(reasoning_effort=effort)
                prepared = self.engine.prepare(payload)
                self.assertTrue(prepared.thinking)
                prompt = self.engine.codec.decode(prepared.prompt)
                self.assertTrue(prompt.endswith('<think>\n'))
                if instruction is None:
                    self.assertNotIn('Reasoning effort is set to', prompt)
                else:
                    self.assertIn(instruction, prompt)
                self.runtime.prompt = prepared.prompt
                self.runtime.output = (self.engine.codec.encode('check</think>Hello')
                                       + [min(self.engine.codec.eos_ids)])
                status, body = self.request(payload)
                self.assertEqual(status, 200)
                self.assertEqual(body['choices'][0]['message'], {
                    'role': 'assistant', 'content': 'Hello', 'reasoning_content': 'check'})
                chunks = self.stream({**payload, 'stream': True})
                deltas = [chunk['choices'][0]['delta'] for chunk in chunks]
                self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), 'check')
                self.assertEqual(''.join(d.get('content', '') for d in deltas), 'Hello')

    def test_invalid_or_disabled_reasoning_effort_is_rejected_before_inference(self):
        invalid = [{'reasoning_effort': effort}
                   for effort in ('high', 'none', '', None, True, 3, [], {})]
        invalid.append({'reasoning_effort': 'low', 'enable_thinking': False})
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                status, body = self.request(self.payload(**overrides))
                self.assertEqual(status, 400)
                self.assertIn('reasoning_effort', body['error']['message'])
        self.assertEqual(self.runtime.calls, [])

    def test_requests_without_effort_preserve_thinking_defaults(self):
        from test_text import TextCodec, tiny_metadata

        self.engine.codec = TextCodec(tiny_metadata())
        for overrides, thinking in (({}, False), ({'enable_thinking': False}, False),
                                    ({'enable_thinking': True}, True)):
            with self.subTest(overrides=overrides):
                request = self.engine.prepare(self.payload(**overrides))
                self.assertEqual(request.thinking, thinking)
                prompt = self.engine.codec.decode(request.prompt)
                if thinking:
                    self.assertIn('Reasoning effort is set to xhigh.', prompt)
                    self.assertTrue(prompt.endswith('<think>\n'))
                else:
                    self.assertNotIn('Reasoning effort is set to', prompt)
                    self.assertTrue(prompt.endswith('<think>\n\n</think>\n\n'))

    def test_sampling_can_choose_non_argmax_and_top_p_limits_candidates(self):
        original = self.runtime.evaluate

        def equal_logits(tokens):
            rows = original(tokens)
            for row in rows:
                row[:] = [-1000.0] * len(row)
                row[65] = row[66] = 0.0
            return rows

        self.runtime.evaluate = equal_logits
        status, body = self.request(self.payload(temperature=1, seed=0, max_tokens=1))
        self.assertEqual((status, body['choices'][0]['message']['content']), (200, 'B'))
        status, body = self.request(self.payload(temperature=1, top_p=0.5, seed=0, max_tokens=1))
        self.assertEqual((status, body['choices'][0]['message']['content']), (200, 'A'))

    def test_stream_sends_role_before_blocked_prefill(self):
        self.runtime.block = True
        connection = self.connection()
        connection.request('POST', '/v1/chat/completions', json.dumps(self.payload(stream=True)),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertTrue(self.runtime.entered.wait(2))
        self.assertEqual(response.status, 200)
        role = json.loads(response.readline().decode().removeprefix('data: '))
        self.assertEqual(role['choices'][0]['delta'], {'role': 'assistant'})
        self.runtime.release.set()
        self.assertIn(b'[DONE]', response.read())

    def test_incomplete_utf8_at_length_is_replaced_once(self):
        self.runtime.output = list('🌍'.encode()) + [256]
        chunks = self.stream(self.payload(stream=True, max_tokens=2))
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
        self.assertEqual(text, '\ufffd')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'length')

    def test_default_completion_budget_fits_remaining_context(self):
        self.engine.context = self.runtime.context = 4
        payload = self.payload()
        payload.pop('max_tokens')
        status, body = self.request(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body['choices'][0]['message']['content'], 'Hel')
        self.assertEqual(body['choices'][0]['finish_reason'], 'length')

    def test_invalid_stop_and_stream_options_are_not_silently_ignored(self):
        for overrides in ({'stop': 0}, {'stop': ''}, {'stream_options': []}):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.request(self.payload(**overrides))[0], 400)

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

    def test_runtime_failure_is_json_or_terminal_sse_and_does_not_hold_lock(self):
        self.runtime.failure = 'numeric failure'
        status, body = self.request(self.payload())
        self.assertEqual(status, 500)
        self.assertEqual(body['error']['code'], 'inference_error')
        chunks = self.stream(self.payload(stream=True))
        self.assertEqual(chunks[-1]['error']['code'], 'inference_error')
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
        self.assertNotEqual(calls[0]['id'], call['id'])
        self.runtime.output = list(b'Sunny') + [256]
        payload = self.payload('Qresult', messages=[{'role': 'user', 'content': 'Q'},
                        {'role': 'tool', 'tool_call_id': call['id'], 'content': 'result'}])
        self.assertEqual(self.request(payload)[1]['choices'][0]['message']['content'], 'Sunny')

    def test_tool_token_limit_returns_length_without_incomplete_call_json_and_sse(self):
        tools = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {
            'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
        self.runtime.output = list(b'Before<tool_call><function=weather><parameter=city>Bern') + [256]
        payload = self.payload(tools=tools, max_tokens=48)
        status, body = self.request(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body['choices'][0]['finish_reason'], 'length')
        self.assertEqual(body['choices'][0]['message'], {'role': 'assistant', 'content': 'Before'})
        chunks = self.stream({**payload, 'stream': True})
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'length')
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks), 'Before')
        self.assertFalse(any(c['choices'][0]['delta'].get('tool_calls') for c in chunks))

    def test_length_keeps_complete_call_and_discards_partial_following_call(self):
        tools = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {
            'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
        complete = '<tool_call><function=weather><parameter=city>Bern</parameter></function></tool_call>'
        partial = '<tool_call><function=weather><parameter=city>'
        self.runtime.output = list((complete + partial + 'Basel').encode()) + [256]
        status, body = self.request(self.payload(tools=tools, max_tokens=len(complete + partial)))
        self.assertEqual(status, 200)
        self.assertEqual(body['choices'][0]['finish_reason'], 'length')
        calls = body['choices'][0]['message']['tool_calls']
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'city': 'Bern'})

    def test_completed_malformed_tool_is_error_even_at_token_limit(self):
        tools = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}]
        malformed = '<tool_call><function=unknown></function></tool_call>'
        self.runtime.output = list(malformed.encode()) + [256]
        status, body = self.request(self.payload(tools=tools, max_tokens=len(malformed)))
        self.assertEqual(status, 500)
        self.assertEqual(body['error']['code'], 'inference_error')

    def test_one_completion_keeps_stream_and_final_tool_ids_identical(self):
        tools = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}]
        self.runtime.output = list(b'<tool_call><function=weather></function></tool_call>') + [256]
        prepared = self.engine.prepare(self.payload(tools=tools, max_tokens=100))
        deltas = []
        message, finish, _ = self.engine.generate(prepared, lambda: None, deltas.append)
        self.assertEqual(finish, 'tool_calls')
        streamed = [call for delta in deltas for call in delta.get('tool_calls', [])]
        self.assertEqual([c['id'] for c in streamed], [c['id'] for c in message['tool_calls']])

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
