#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
"""Benchmark one non-streaming Redshift chat request using only Python's standard library.

Usage: bash tools/bench_endpoint.sh [--max-tokens 1024]
The reported token/s includes network, prompt processing and generation time.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://10.10.10.55:8081',
                        help='endpoint base URL (default: %(default)s)')
    parser.add_argument('--model', default='redshift-qwen3.8-27b')
    parser.add_argument('--prompt', default='Spiega in italiano come funziona un motore elettrico, '
                        'in circa 500 parole.')
    parser.add_argument('--max-tokens', type=positive_int, default=512)
    parser.add_argument('--timeout', type=positive_int, default=180,
                        help='request timeout in seconds (default: %(default)s)')
    args = parser.parse_args()

    payload = dict(model=args.model, messages=[{'role': 'user', 'content': args.prompt}],
                   temperature=0, max_tokens=args.max_tokens, stream=False, enable_thinking=False)
    req = urllib.request.Request(
        args.url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        choice = result['choices'][0]
        content = choice['message']['content']
        tokens = result['usage']['completion_tokens']
        if type(tokens) is not int or tokens < 0 or not isinstance(content, str):
            raise ValueError('invalid content or completion_tokens')
        finish = choice.get('finish_reason', 'unknown')
    except urllib.error.HTTPError as error:
        print(f'Errore HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}',
              file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f'Test fallito: {error}', file=sys.stderr)
        return 1

    print(content)
    print(f'\n{tokens} token in {elapsed:.2f} s → {tokens / elapsed:.2f} token/s complessivi')
    print(f'finish_reason: {finish}')
    if finish == 'length':
        print('Risposta interrotta per il limite di token.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
PY
