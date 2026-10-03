"""Controlled acceptance benchmark: full-forward replay vs recurrent transitions.

Synthetic supplied IDs, not a proposer, sampler or generation benchmark.
All verification logits are downloaded in both strategies. Acceptance is passed
only to commit, after verification. GPU ABI calls are synchronous.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASES = ('checkpoint_seconds', 'verify_seconds', 'restore_seconds',
          'replay_seconds', 'commit_seconds', 'total_seconds')
RECURRENT_BYTES = 48 * (48*128*128 + 3*10240) * 4
TRACE_BYTES = 48 * 8 * (2*10240 + 2*48) * 4
KV_BYTES_PER_TOKEN = 16 * 2 * 4 * 256 * 2


def run_cycle(runtime, tokens, accepted, strategy):
    if (strategy not in ('replay', 'transitions') or not 1 <= len(tokens) <= 8
            or type(accepted) is not int or not 0 <= accepted <= len(tokens)):
        raise ValueError('invalid controlled-acceptance cell')
    start_position = runtime.position
    start = time.perf_counter()
    if strategy == 'replay':
        runtime.checkpoint()
        checkpointed = time.perf_counter()
        runtime.evaluate(tokens)
        verified = time.perf_counter()
        if accepted < len(tokens):
            runtime.restore()
            restored = time.perf_counter()
            if accepted:
                # The existing lightweight ABI avoids downloading unused logits.
                runtime.advance(tokens[:accepted])
            replayed = time.perf_counter()
        else:
            restored = replayed = verified
        committed = replayed
    else:
        checkpointed = start  # Recurrent start-state save is inside verify.
        runtime.verify(tokens)
        verified = time.perf_counter()
        restored = replayed = verified
        runtime.commit(accepted)
        committed = time.perf_counter()
    position = runtime.position
    if position != start_position + accepted:
        raise ValueError('commit position differs from accepted prefix')
    return dict(strategy=strategy, batch=len(tokens), accepted=accepted,
                position_after=position, checkpoint_seconds=checkpointed-start,
                verify_seconds=verified-checkpointed, restore_seconds=restored-verified,
                replay_seconds=replayed-restored, commit_seconds=committed-replayed,
                total_seconds=committed-start)


def compare_rows(actual, reference, tolerance=3e-4):
    if len(actual) != len(reference) or not reference:
        raise ValueError('invalid logit row count')
    worst = absolute = 0.
    for row, expected in zip(actual, reference):
        if len(row) != len(expected) or not expected:
            raise ValueError('invalid logit row width')
        for a, b in zip(row, expected):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('non-finite logits')
            absolute = max(absolute, abs(a-b))
            worst = max(worst, abs(a-b)/(1+abs(b)))
        if max(range(len(row)), key=row.__getitem__) != max(range(len(expected)), key=expected.__getitem__):
            raise ValueError('continuation argmax differs from serial reference')
    if worst > tolerance:
        raise ValueError(f'continuation drift {worst} exceeds {tolerance}')
    return dict(max_scaled_error=worst, max_absolute_error=absolute, argmax_matches=True)


def summarize(samples):
    cells = []
    keys = sorted({(s['prefix'], s['batch'], s['accepted'], s['strategy']) for s in samples})
    for prefix, batch, accepted, strategy in keys:
        selected = [s for s in samples if
                    (s['prefix'], s['batch'], s['accepted'], s['strategy']) == (prefix, batch, accepted, strategy)]
        timings = {}
        for phase in PHASES:
            values = [s[phase] for s in selected]
            timings[phase] = dict(median=statistics.median(values), min=min(values), max=max(values),
                                 stdev=statistics.stdev(values) if len(values)>1 else 0.)
        total = timings['total_seconds']['median']
        cells.append(dict(prefix=prefix, batch=batch, accepted=accepted, strategy=strategy,
                          repetitions=len(selected), timings=timings,
                          accepted_inputs_per_second=accepted/total if total else None))
    return cells


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_info():
    try:
        return subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw',
            '--format=csv'], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {'unavailable': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--prefixes', nargs='+', type=int, default=[0, 128, 2048])
    parser.add_argument('--groups', nargs='+', type=int, choices=[2, 4, 8], default=[8])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--library', type=Path, default=ROOT/'build/libqvelox.so')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.prefixes) < 0 or max(args.prefixes)+max(args.groups)+2 > 32768:
        parser.error('need >=2 repetitions, nonnegative prefixes and capacity <=32768')
    args.model = args.model.resolve()
    args.library = args.library.resolve()
    # Hashing and loading are excluded from steady-state cycle timings.
    sources = [ROOT/p for p in ('src/runtime.cu', 'src/runtime.h', 'src/kernels.cu',
                               'src/kernels.h', 'qvelox/runtime.py', 'tools/bench_commit.py', 'Makefile')]
    report = dict(kind='controlled_acceptance_not_generation', samples=[], validation=[],
        prefixes=args.prefixes, groups=args.groups, repeats=args.repeats,
        model=dict(path=str(args.model), bytes=args.model.stat().st_size, sha256=sha256(args.model)),
        library=dict(path=str(args.library), sha256=sha256(args.library)),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        git_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        gpu_before=gpu_info(),
        timing_scope='synchronous host wall time: start-state save + all-logit verify + accepted-prefix commit; no draft, sampler or correction token',
        transition_save_timing='initial recurrent save and trace recording are included in verify_seconds',
        memory=dict(recurrent_start_bytes=RECURRENT_BYTES, transition_trace_bytes=TRACE_BYTES,
                    extra_allocated_bytes=RECURRENT_BYTES+TRACE_BYTES, kv_bytes_per_token=KV_BYTES_PER_TOKEN,
                    note='derived buffer payload sizes, not measured HBM traffic; nvidia-smi includes other GPU processes'))
    from qvelox.runtime import Runtime
    started = time.perf_counter()
    with Runtime(args.model, context=max(args.prefixes)+max(args.groups)+2, library=args.library) as runtime:
        report['load_seconds'] = time.perf_counter()-started
        report['gpu_loaded'] = gpu_info()
        for prefix in args.prefixes:
            runtime.reset()
            prefix_ids = [10+i%1000 for i in range(prefix)]
            for offset in range(0, prefix, 8):
                runtime.advance(prefix_ids[offset:offset+8])
            runtime.checkpoint()
            cells = [(batch, accepted) for batch in args.groups for accepted in range(batch+1)]
            for batch, accepted in cells:
                tokens = [100+13*i for i in range(batch)]
                runtime.restore()
                for token in tokens[:accepted]:
                    runtime.evaluate([token])
                reference = runtime.evaluate([900, 901])
                for strategy in ('replay', 'transitions'):
                    runtime.restore()
                    # Untimed warmup also allocates the lazy transition buffers.
                    run_cycle(runtime, tokens, accepted, strategy)
                    metrics = compare_rows(runtime.evaluate([900, 901]), reference)
                    report['validation'].append(dict(prefix=prefix, batch=batch, accepted=accepted,
                                                     strategy=strategy, **metrics))
            report['gpu_with_transitions'] = gpu_info()
            for repeat in range(args.repeats):
                ordered = cells if repeat%2==0 else list(reversed(cells))
                strategies = ('replay', 'transitions') if repeat%2==0 else ('transitions', 'replay')
                for batch, accepted in ordered:
                    for strategy in strategies:
                        runtime.restore()  # Fixture setup excluded; each cycle saves its own start.
                        sample = run_cycle(runtime, [100+13*i for i in range(batch)], accepted, strategy)
                        report['samples'].append(dict(prefix=prefix, repeat=repeat, **sample))
        report['gpu_end'] = gpu_info()
    report['summary'] = summarize(report['samples'])
    report['comparisons'] = []
    indexed = {(cell['prefix'], cell['batch'], cell['accepted'], cell['strategy']): cell
               for cell in report['summary']}
    for prefix, batch, accepted, strategy in indexed:
        if strategy != 'replay':
            continue
        replay = indexed[prefix, batch, accepted, 'replay']['timings']
        transitions = indexed[prefix, batch, accepted, 'transitions']['timings']
        replay_commit = statistics.median(s['restore_seconds']+s['replay_seconds']
            for s in report['samples'] if
            (s['prefix'], s['batch'], s['accepted'], s['strategy']) == (prefix, batch, accepted, 'replay'))
        report['comparisons'].append(dict(prefix=prefix, batch=batch, accepted=accepted,
            controlled_cycle_speedup=replay['total_seconds']['median']/transitions['total_seconds']['median'],
            transition_verify_overhead_seconds=transitions['verify_seconds']['median']-replay['verify_seconds']['median'],
            replay_restore_and_forward_seconds=replay_commit,
            transition_commit_seconds=transitions['commit_seconds']['median']))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), cells=len(report['summary']),
                         validations=len(report['validation'])), indent=2))


if __name__ == '__main__':
    main()
