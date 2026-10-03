"""Compare saved llama.cpp logits with Redshift, then time identical token groups."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvelox.runtime import Runtime


def metrics(reference, actual):
    if len(reference) != len(actual) or not all(math.isfinite(x) for x in (*reference, *actual)):
        raise ValueError('invalid logit row')
    differences = [a-b for a, b in zip(actual, reference)]
    dot = sum(a*b for a, b in zip(actual, reference))
    cosine = dot / math.sqrt(sum(x*x for x in actual) * sum(x*x for x in reference))
    ra, rb = max(reference), max(actual)
    rsum = sum(math.exp(x-ra) for x in reference)
    asum = sum(math.exp(x-rb) for x in actual)
    rlogz, alogz = ra + math.log(rsum), rb + math.log(asum)
    kl = sum(math.exp(r-rlogz) * (r-rlogz-a+alogz) for r, a in zip(reference, actual))
    top_reference = sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:5]
    top_actual = sorted(range(len(actual)), key=actual.__getitem__, reverse=True)[:5]
    return {'max_absolute_error': max(abs(x) for x in differences),
            'rmse': math.sqrt(sum(x*x for x in differences) / len(differences)),
            'cosine': cosine, 'kl_reference_to_redshift': kl,
            'reference_top5': top_reference, 'redshift_top5': top_actual,
            'argmax_matches': top_reference[0] == top_actual[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('reference_stem', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    reference = json.loads(args.reference_stem.with_suffix('.json').read_text())
    prefix = reference['prefix']
    raw = array('f')
    with args.reference_stem.with_suffix('.f32').open('rb') as file:
        raw.fromfile(file, 8 * 248320)
        if file.read(1):
            raise ValueError('unexpected reference payload size')
    tokens = [100 + i * 13 for i in range(8)]
    report = {'llama': reference, 'prefix_ids': list(range(10,10+prefix)),
              'candidate_ids': tokens, 'comparison': [], 'redshift_samples': [],
              'timing_scope': 'GPU checkpoint restore + all-row forward + copy every vocabulary logit to host',
              'timing_difference': 'Redshift uses Python/ctypes output allocation and row copies; llama reference uses C++ vector allocation/copies',
              'llama_commit': subprocess.check_output(['git','-C','/opt/src/llama.cpp','rev-parse','HEAD'],text=True).strip(),
              'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,clocks.sm,clocks.mem','--format=csv'],text=True).strip(),
              'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted([*Path('src').glob('*'), *Path('qvelox').glob('*.py'), *Path('tools').glob('*')]) if p.is_file()},
              'library_sha256': hashlib.sha256(Path('build/libqvelox.so').read_bytes()).hexdigest(),
              'model': {'path':str(args.model),'bytes':args.model.stat().st_size}}
    load_started=time.perf_counter()
    with Runtime(args.model, context=prefix+8) as runtime:
        report['load_seconds']=time.perf_counter()-load_started
        report['gpu_loaded']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv'],text=True).strip()
        for start in range(0,prefix,8):
            runtime.evaluate(list(range(10+start,10+min(start+8,prefix))))
        runtime.checkpoint()
        rows=runtime.evaluate(tokens)
        for t,row in enumerate(rows):
            report['comparison'].append(metrics(raw[t*248320:(t+1)*248320],row))
        # Warm each group once, matching the reference harness.
        for batch in (1,2,4,8):
            runtime.restore()
            runtime.evaluate(tokens[:batch])
        for repeat in range(args.repeats):
            order=(1,2,4,8) if repeat%2==0 else (8,4,2,1)
            for batch in order:
                started=time.perf_counter()
                runtime.restore()
                restored=time.perf_counter()
                rows=runtime.evaluate(tokens[:batch])
                ended=time.perf_counter()
                report['redshift_samples'].append({'repeat':repeat,'batch':batch,
                    'restore_seconds':restored-started,'evaluate_seconds':ended-restored,'total_seconds':ended-started})
    report['summary']={}
    for batch in (1,2,4,8):
        item={}
        for engine,samples in (('llama',reference['samples']),('redshift',report['redshift_samples'])):
            selected=[row for row in samples if row['batch']==batch]
            total=statistics.median(row['total_seconds'] for row in selected)
            forward=statistics.median(row['evaluate_seconds'] for row in selected)
            item[engine]={'total_seconds':total,'evaluate_seconds':forward,
                          'verified_tokens_per_second':batch/total}
        item['redshift_speedup']=item['llama']['total_seconds']/item['redshift']['total_seconds']
        report['summary'][str(batch)]=item
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':report['summary'],'comparison':report['comparison']},indent=2))


if __name__=='__main__':
    main()
