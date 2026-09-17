"""M6.5 fixes the already selected M5.7 model and all four final-test pages."""
import json
from benchmark_contract import ROOT

CASES = {name: f'tests/fixtures/selection-pages/{name}.png'
         for name in ('library', 'pets', 'fruit', 'tea')}


def validate_selected(cases):
    if set(cases) != set(CASES):
        raise ValueError('All four selected-model pages are required')
    for name, case in cases.items():
        expected = json.loads((ROOT / f'tests/fixtures/selection-results/final/adapted/{name}.json').read_text())
        if case['tiles'] != 13 or any(case[key] != expected[key] for key in ('tokens', 'raw', 'stop_reason')):
            raise ValueError('Selected output changed: ' + name)
