"""Strict matched-workload comparison; no model imports or performance assumptions."""
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
CASES = {'native-page': 'examples/data/native-page.png',
         'grid': 'tests/fixtures/table-pages/grid.png', 'merged': 'tests/fixtures/table-pages/merged.png'}
STAGES = ('preprocess', 'vision', 'prefill', 'decode', 'detokenize', 'generation')


def summary(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError('Expected finite nonnegative timing samples')
    return dict(median=median(samples), minimum=min(samples), maximum=max(samples))


def validate_run(run, cases=CASES):
    if (run['dtype'] != 'float32' or run['max_new_tokens'] != 512 or run['warmups'] != 1
            or run['repeats'] != 3 or set(run['cases']) != set(cases)
            or run['task'] != 'Convert this page to docling.'):
        raise ValueError('Benchmark policy mismatch')
    for case in run['cases'].values():
        if not case['tokens'] or len(case['samples']) != 3:
            raise ValueError('Missing tokens or repeats')
        for sample in case['samples']:
            for stage in STAGES:
                summary([sample[stage]])
            if sample['generation'] + 1e-6 < sum(sample[s] for s in STAGES[:-1]):
                raise ValueError('Generation total cannot be shorter than its stages')


def compare_runs(runs, cases=CASES):
    if len(runs) < 2:
        raise ValueError('Comparison requires multiple runs')
    for run in runs:
        validate_run(run, cases)
        for name in cases:
            for field in ('prompt_ids', 'tiles', 'tokens', 'raw', 'stop_reason'):
                if run['cases'][name][field] != runs[0]['cases'][name][field]:
                    raise ValueError(f'Unmatched workload/output: {name} {field}')
    return {run['runtime'] + '/' + run['device']: {
        name: {stage: summary([s[stage] for s in case['samples']]) for stage in STAGES}
        for name, case in run['cases'].items()} for run in runs}
