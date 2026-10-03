"""Compare complete-model logits with the original FP32 projection graph.

Build the diagnostic graph with `make reference`. Models are loaded sequentially
so the comparison needs only one model's VRAM. This checks numerical agreement
on supplied IDs, not text quality or speculative acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvelox.runtime import Runtime
from compare_engines import metrics


def sample(model, library, prefixes):
    rows = {}
    with Runtime(model, context=max(prefixes)+8, library=library) as runtime:
        for prefix in sorted(prefixes):
            while runtime.position < prefix:
                start = runtime.position
                runtime.evaluate(list(range(10+start, 10+min(start+8, prefix))))
            runtime.checkpoint()
            rows[prefix] = runtime.evaluate([100+13*i for i in range(8)])
            runtime.restore()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('build/libqvelox-reference.so'))
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('--prefixes', type=int, nargs='+', default=[16, 128, 1024])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(p < 1 or p > 2040 for p in args.prefixes):
        parser.error('prefixes must be in 1..2040')
    report = {
        'reference': str(args.reference),
        'reference_contract': 'Original GGUF weights, FP32 activations, original k_mm; same remaining model graph',
        'model': {'path': str(args.model), 'bytes': args.model.stat().st_size},
        'prefixes': args.prefixes,
        'prefix_ids': '10..10+prefix-1',
        'candidate_ids': [100+13*i for i in range(8)],
        'library_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [args.reference, *args.candidate]},
        'comparison': {},
    }
    reference = sample(args.model, args.reference, args.prefixes)
    for library in args.candidate:
        actual = sample(args.model, library, args.prefixes)
        report['comparison'][str(library)] = {
            str(prefix): [metrics(r, a) for r, a in zip(reference[prefix], actual[prefix])]
            for prefix in args.prefixes
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for library, comparisons in report['comparison'].items():
        for prefix, rows in comparisons.items():
            print(library, prefix, 'max_rmse', max(r['rmse'] for r in rows),
                  'min_cosine', min(r['cosine'] for r in rows),
                  'argmax_matches', sum(r['argmax_matches'] for r in rows), '/8')


if __name__ == '__main__':
    main()
