"""Replay M6.4 measurements and independent numerical reports without a model."""
import hashlib
import json
import unittest
from benchmark_contract import ROOT, CASES, compare_runs
from mlx_vlm_contract import compare_output

FOLDER = ROOT / 'tests/fixtures/memory-benchmark'


def read(name):
    return json.loads((FOLDER / name).read_text())


class MemoryBenchmarkTests(unittest.TestCase):
    def test_exact_outputs_and_repeats(self):
        report = read('report.json')
        rows = [dict(read(name + '.json'), runtime=name) for name in report['runtime_order']]
        self.assertEqual(compare_runs(rows), report['summary'])
        expected = json.loads((ROOT / 'tests/fixtures/benchmark/python-cpu.json').read_text())
        for row in rows:
            for name, case in row['cases'].items():
                self.assertEqual(compare_output(case, expected['cases'][name]), [])
                if row['runtime'].startswith('native-'):
                    self.assertTrue(case['public_api_equal'])
                    self.assertEqual({s['native_handles'] for s in case['samples']}, {471})
                    self.assertEqual(case['markdown_available'], name != 'merged')
        self.assertEqual(sum(len(c['samples']) for r in rows for c in r['cases'].values()), 27)
        for name, path in CASES.items():
            self.assertEqual(report['image_sha256'][name], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_measured_memory_reduction(self):
        before, after = read('native-before.json'), read('native-after.json')
        for name in CASES:
            old, new = before['cases'][name]['samples'], after['cases'][name]['samples']
            self.assertLess(max(s['mlx_peak_bytes'] for s in new), min(s['mlx_peak_bytes'] for s in old))
            self.assertEqual({s['mlx_active_bytes'] for s in new}, {s['mlx_active_bytes'] for s in old})
            for sample in old + new:
                self.assertGreaterEqual(sample['mlx_peak_bytes'], sample['mlx_active_bytes'])

    def test_independent_vision(self):
        for kind in ('tiny', 'real', 'real-padded'):
            for device in ('cpu', 'gpu'):
                report = read(f'vision-{kind}-{device}.json')
                old = json.loads((ROOT / f'tests/fixtures/upload-benchmark/vision-{kind}-{device}.json').read_text())
                self.assertEqual(report['checks'], old['checks'])
                self.assertEqual(report['sha256'], old['sha256'])
                self.assertEqual(report['max_absolute_error'], old['max_absolute_error'])
                self.assertEqual((report['atol'], report['rtol']), (old['atol'], old['rtol']))


if __name__ == '__main__':
    unittest.main()
