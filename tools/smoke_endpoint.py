#!/usr/bin/env python3
"""Exercise an actual Redshift HTTP endpoint; never runs generated tools."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-long', action='store_true', help='skip the slow cold long-prompt probe')
    args = parser.parse_args()
    reports = []

    def request(name, messages, **options):
        payload = dict(model='redshift-qwen3.8-27b', messages=messages,
                       max_tokens=256, temperature=0)
        payload.update(options)
        start = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/chat/completions',
                                     json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
        first = None
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                if not payload.get('stream'):
                    result = json.load(response)
                else:
                    content, calls, usage, finish, done = '', [], None, None, False
                    for raw in response:
                        if not raw.startswith(b'data: '):
                            continue
                        if raw.strip() == b'data: [DONE]':
                            done = True
                            break
                        event = json.loads(raw[6:])
                        assert 'error' not in event, event
                        if event.get('usage'):
                            usage = event['usage']
                        for choice in event['choices']:
                            delta = choice['delta']
                            if (delta.get('content') or delta.get('tool_calls')) and first is None:
                                first = time.monotonic() - start
                            content += delta.get('content') or ''
                            calls.extend(delta.get('tool_calls', []))
                            finish = choice['finish_reason'] or finish
                    assert done and finish and usage, (done, finish, usage)
                    result = {'choices': [{'message': {'role': 'assistant', 'content': content,
                                                       **({'tool_calls': calls} if calls else {})},
                                           'finish_reason': finish}], 'usage': usage}
            record = {'name': name, 'elapsed_s': time.monotonic() - start,
                      'first_content_s': first, 'result': result}
        except urllib.error.HTTPError as error:
            record = {'name': name, 'status': error.code, 'result': json.load(error)}
        reports.append(record)
        args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record['result']

    user = lambda text: {'role': 'user', 'content': text}
    short = [user('Quanto fa 6 per 7? Rispondi solo con il numero.')]
    ordinary = request('json', short, max_tokens=24)
    streamed = request('sse', short, max_tokens=24, stream=True,
                       stream_options={'include_usage': True})
    assert ordinary['choices'][0]['message']['content'].strip() == '42'
    assert streamed['choices'][0]['message']['content'] == ordinary['choices'][0]['message']['content']
    assert ordinary['usage'] == streamed['usage']

    tools = [{'type': 'function', 'function': {
        'name': 'lookup_temperature', 'description': 'Read current temperature in a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city'], 'additionalProperties': False}}}]
    conversation = [{'role': 'system', 'content': 'Use the supplied tool to retrieve temperatures.'},
                    user('Usa lookup_temperature per leggere la temperatura a Zurigo.')]
    tool_result = request('tool_sse', conversation, tools=tools, stream=True,
                          stream_options={'include_usage': True})
    choice = tool_result['choices'][0]
    assert choice['finish_reason'] == 'tool_calls', choice
    assistant = choice['message']
    call = assistant['tool_calls'][0]
    assert call['function']['name'] == 'lookup_temperature'
    assert isinstance(json.loads(call['function']['arguments'])['city'], str)
    for item in assistant['tool_calls']:
        item.pop('index', None)
    conversation += [assistant, {'role': 'tool', 'tool_call_id': call['id'],
                                 'content': '{"city":"Zurigo","temperature_c":12}'}]
    continuation = request('tool_result_continuation', conversation, tools=tools, max_tokens=100)
    assert '12' in continuation['choices'][0]['message']['content'], continuation

    if not args.skip_long:
        long_result = request('long_context', [user('Contesto di prova da ignorare:\n' + 'nota ' * 2300
                                                   + '\nQuanto fa 6 per 7? Rispondi solo con il numero.')],
                              max_tokens=24, stream=True, stream_options={'include_usage': True})
        assert long_result['usage']['prompt_tokens'] > 2048, long_result['usage']
        assert long_result['choices'][0]['message']['content'].strip() == '42', long_result

    request('generation_speed', [user('Scrivi una funzione Python per Fibonacci iterativo, '
                                     'poi spiega in italiano come funziona in tre frasi.')],
            max_tokens=180, stream=True, stream_options={'include_usage': True})
    sampled = request('sampling', short, temperature=0.2, top_p=0.95, seed=42, max_tokens=24)
    assert sampled['choices'][0]['message']['content'].strip() == '42'
    overflow = request('context_overflow', short, max_tokens=32768)
    assert reports[-1].get('status') == 400
    assert overflow['error']['code'] == 'context_length_exceeded'
    print('All live checks passed.', flush=True)


if __name__ == '__main__':
    main()
