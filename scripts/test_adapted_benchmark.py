"""Model-free policy and evidence checks for the frozen selected adapter."""
import copy
import hashlib
import json
import unittest
from benchmark_contract import ROOT, compare_runs
from test_benchmark import run
from adapted_benchmark_contract import CASES, validate_selected


class AdaptedBenchmarkTests(unittest.TestCase):
    def test_case_set_is_explicit(self):
        data = run()
        data['cases'] = {name: copy.deepcopy(data['cases']['grid']) for name in CASES}
        compare_runs([data, data], cases=CASES)
        with self.assertRaises(ValueError):
            compare_runs([data, data])
        data['cases'].pop('pets')
        with self.assertRaises(ValueError):
            compare_runs([data, data], cases=CASES)

    def test_selected_output_guard(self):
        cases = {}
        for name in CASES:
            expected = json.loads((ROOT / f'tests/fixtures/selection-results/final/adapted/{name}.json').read_text())
            cases[name] = {**expected, 'tiles': 13}
        validate_selected(cases)
        for field, value in [('tokens', [49279]), ('raw', 'repaired'), ('tiles', 1), ('stop_reason', 'length')]:
            bad = copy.deepcopy(cases)
            bad['pets'][field] = value
            with self.assertRaises(ValueError):
                validate_selected(bad)

    def test_recorded_measurements(self):
        folder = ROOT / 'tests/fixtures/adapted-benchmark'
        report = json.loads((folder / 'report.json').read_text())
        self.assertEqual(report['runtime_order'], ['native-adapter', 'native-merged',
                                                 'python-transformers-merged', 'python-mlx-merged'])
        rows = [dict(json.loads((folder / (name + '.json')).read_text()), runtime=name)
                for name in report['runtime_order']]
        self.assertEqual(compare_runs(rows, cases=CASES), report['summary'])
        self.assertEqual(sum(len(c['samples']) for r in rows for c in r['cases'].values()), 48)
        for row in rows:
            validate_selected(row['cases'])
            for case in row['cases'].values():
                if row['runtime'].startswith('native'):
                    self.assertTrue(case['public_api_equal'])
                    self.assertEqual({s['native_handles'] for s in case['samples']},
                                     {591 if row['runtime'] == 'native-adapter' else 471})
                    self.assertTrue(all(s['mlx_peak_bytes'] >= s['mlx_active_bytes'] > 0 for s in case['samples']))
        selection = json.loads((ROOT / 'tests/fixtures/selection-results/selection.json').read_text())
        self.assertEqual(report['adapter_sha256'], selection['adapter_sha256'])
        self.assertEqual(report['selected_step'], 16)
        for name, path in CASES.items():
            self.assertEqual(report['image_sha256'][name], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_recorded_export_and_numerics(self):
        folder = ROOT / 'tests/fixtures/adapted-benchmark'
        report = json.loads((folder / 'export-check.json').read_text())
        self.assertEqual((report['weight_count'], report['exact_unchanged_weights'], report['merged_weights'], report['adapter_factors']),
                         (471, 411, 60, 120))
        self.assertEqual(len(report['per_weight_max_error']), 60)
        self.assertEqual(len(report['byte_identical_assets']), 10)
        self.assertTrue(report['config_only_fp32_declarations'])
        audit = json.loads((folder / 'numerical-audit.json').read_text())
        self.assertEqual((audit['atol'], audit['rtol']), (1e-3, 3e-4))
        self.assertEqual(set(audit['results']['corrected']), set(CASES))
        for case in audit['results']['corrected'].values():
            for key, shape in [('features', [832, 576]), ('last_prefill_logits', [1, 49280])]:
                self.assertTrue(case[key]['passes'])
                self.assertTrue(case[key]['finite'])
                self.assertEqual(case[key]['outside_tolerance'], 0)
                self.assertEqual(case[key]['shape'], shape)
        python = json.loads((folder / 'python-mlx-merged.json').read_text())
        for name in CASES:
            self.assertEqual(python['cases'][name]['pixel_sha256'], audit['results']['corrected'][name]['pixel_sha256'])


if __name__ == '__main__':
    unittest.main()
