"""Inspect the supported GGUF or measure full-vocabulary group verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from .gguf import read_gguf
from .model import schema, validate_qwen27b
from .runtime import Runtime


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True, timeout=10).strip()


def verify(runtime, tokens):
    runtime.restore()
    sequential = [runtime.evaluate([token])[0] for token in tokens]
    errors = {}
    for batch in (1, 2, 4, 8):
        runtime.restore()
        rows = runtime.evaluate(tokens[:batch])
        worst = 0.
        for row, reference in zip(rows, sequential):
            if not all(math.isfinite(x) for x in row) or not all(math.isfinite(x) for x in reference):
                raise RuntimeError('non-finite model output; benchmark rejected')
            worst = max(worst, max(abs(a-b) / (1+abs(b)) for a, b in zip(row, reference)))
            if max(range(len(row)), key=row.__getitem__) != max(range(len(reference)), key=reference.__getitem__):
                raise RuntimeError('group and sequential argmax differ; benchmark rejected')
        if worst > 3e-4:
            raise RuntimeError('group and sequential logits differ; benchmark rejected')
        errors[str(batch)] = worst
    return errors


def benchmark(args):
    root = Path(__file__).resolve().parents[1]
    model_path = args.model.resolve()
    sources = sorted([root / 'Makefile', *root.glob('src/*'), *root.glob('qvelox/*.py')])
    sources = [path for path in sources if path.is_file()]
    prefix = list(range(10, 10 + args.prefix))
    tokens = [100 + i * 13 for i in range(8)]
    print('Hashing the existing model and sources...', file=sys.stderr, flush=True)
    report = {
        'measurement': 'numeric group verification, all vocabulary rows, synthetic token IDs',
        'limitations': [
            'This is not autoregressive generation throughput or accepted speculative tokens/s.',
            'No tokenizer, text-quality validation, proposal model or acceptance loop.',
            'Full-model correctness is checked against this runtime sequentially, not an external engine.',
            'The timing includes host output allocation, all GPU work, logits transfer and Python row copies.',
            'The primary throughput also includes restoring the same prefix checkpoint before each call.',
        ],
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': {'path': str(model_path), 'bytes': model_path.stat().st_size, 'sha256': digest(model_path)},
        'source_sha256': {str(path.relative_to(root)): digest(path) for path in sources},
        'library_sha256': digest(root / 'build/libqvelox.so'),
        'platform': platform.platform(), 'python': platform.python_version(),
        'gpu_before': command_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu', '--format=csv']),
        'nvcc': command_output(['/usr/local/cuda/bin/nvcc', '--version']),
        'prefix_ids': prefix, 'candidate_ids': tokens, 'repeats': args.repeats,
        'context_capacity': args.prefix + 8, 'vocabulary_rows_per_token': 248320,
        'weight_upload_chunk_bytes': 16 * 1024 * 1024,
        'samples': [],
    }
    print('Loading quantized weights...', file=sys.stderr, flush=True)
    start = time.perf_counter()
    with Runtime(model_path, context=args.prefix + 8) as runtime:
        report['load_seconds'] = time.perf_counter() - start
        for start in range(0, len(prefix), 8):
            runtime.evaluate(prefix[start:start+8])
        runtime.checkpoint()
        print('Checking grouped logits against sequential evaluation...', file=sys.stderr, flush=True)
        report['max_scaled_logit_error'] = verify(runtime, tokens)
        report['warmup'] = 'sequential correctness pass and one full call per group size'
        report['gpu_loaded'] = command_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu', '--format=csv'])
        for repeat in range(args.repeats):
            order = (1, 2, 4, 8) if repeat % 2 == 0 else (8, 4, 2, 1)
            for batch in order:
                started = time.perf_counter()
                runtime.restore()
                restored = time.perf_counter()
                rows = runtime.evaluate(tokens[:batch])
                ended = time.perf_counter()
                checksum = hashlib.sha256()
                for row in rows:
                    checksum.update(row.tobytes())
                report['samples'].append({
                    'repeat': repeat, 'batch': batch,
                    'restore_seconds': restored - started,
                    'evaluate_seconds': ended - restored,
                    'total_seconds': ended - started,
                    'logits_sha256': checksum.hexdigest(),
                })
            print(f'Measurement round {repeat + 1}/{args.repeats}', file=sys.stderr, flush=True)
    report['summary'] = {}
    for batch in (1, 2, 4, 8):
        samples = [row for row in report['samples'] if row['batch'] == batch]
        median = statistics.median(row['total_seconds'] for row in samples)
        forward = statistics.median(row['evaluate_seconds'] for row in samples)
        if len({row['logits_sha256'] for row in samples}) != 1:
            raise RuntimeError('outputs changed between repeated measurements')
        report['summary'][str(batch)] = {
            'median_total_seconds': median, 'median_evaluate_seconds': forward,
            'verified_input_tokens_per_second_including_restore': batch / median,
            'verified_input_tokens_per_second_evaluate_only': batch / forward,
            'min_total_seconds': min(row['total_seconds'] for row in samples),
            'max_total_seconds': max(row['total_seconds'] for row in samples),
        }
    report['gpu_after'] = command_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu', '--format=csv'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='validate header and model manifest without CUDA')
    inspect.add_argument('model', type=Path)
    bench = commands.add_parser('bench', help='measure groups 1/2/4/8 on the existing model')
    bench.add_argument('model', type=Path)
    bench.add_argument('--prefix', type=int, default=16, help='synthetic prefix length, 0..2040')
    bench.add_argument('--repeats', type=int, default=5)
    bench.add_argument('--output', type=Path, help='new JSON report path; existing files are not overwritten')
    args = parser.parse_args()
    if args.command == 'bench':
        if not 0 <= args.prefix <= 2040 or args.repeats < 1:
            parser.error('prefix must be 0..2040 and repeats must be positive')
        if args.output is not None and args.output.exists():
            parser.error('output file already exists')
    try:
        if args.command == 'inspect':
            model = read_gguf(args.model)
            eps = validate_qwen27b(model)
            result = {'model': str(model.path), 'epsilon': eps, 'tensor_count': len(model.tensors),
                      'weight_bytes': sum(t.nbytes for t in model.tensors.values()),
                      'manifest': [{'slot': slot, 'name': name, 'shape': shape, 'kind': kind,
                                    'offset': model.tensors[name].offset, 'bytes': model.tensors[name].nbytes}
                                   for slot, name, shape, kind in schema()]}
        else:
            result = benchmark(args)
        data = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.command == 'bench' and args.output:
            with args.output.open('x') as file:
                file.write(data)
        else:
            print(data, end='')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
