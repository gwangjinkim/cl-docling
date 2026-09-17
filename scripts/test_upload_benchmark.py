"""Replay M6.3 evidence without numerical dependencies or hardware."""
import hashlib
import json
import unittest
from benchmark_contract import ROOT, CASES, compare_runs
from mlx_vlm_contract import compare_output

FOLDER = ROOT / 'tests/fixtures/upload-benchmark'


def read(name):
    return json.loads((FOLDER / name).read_text())


class UploadBenchmarkTests(unittest.TestCase):
    def test_matched_outputs_and_statistics(self):
        report = read('report.json')
        rows = [dict(read(name + '.json'), runtime=name) for name in report['runtime_order']]
        self.assertEqual(compare_runs(rows), report['summary'])
        expected = json.loads((ROOT / 'tests/fixtures/benchmark/python-cpu.json').read_text())
        for row in rows:
            for name, case in row['cases'].items():
                self.assertEqual(compare_output(case, expected['cases'][name]), [])
                self.assertEqual(case['tokens'][-1], 49279)
                if row['runtime'].startswith('native-'):
                    self.assertTrue(case['public_api_equal'])
                    self.assertEqual({s['native_handles'] for s in case['samples']}, {471})
                    self.assertEqual(case['markdown_available'], name != 'merged')
        self.assertEqual(sum(len(c['samples']) for r in rows for c in r['cases'].values()), 27)
        for name, file in CASES.items():
            self.assertEqual(report['image_sha256'][name], hashlib.sha256((ROOT / file).read_bytes()).hexdigest())

    def test_profile_evidence(self):
        probe = read('attention-mask.json')
        self.assertTrue(probe['equivalent'])
        self.assertEqual(probe['maximum_absolute_difference'], 0)
        self.assertEqual(probe['shape'], [1, 12, 1024, 64])
        for name in ('vision-scopes.json', 'vision-transfers-before.json', 'vision-transfers-after.json'):
            probe = read(name)
            for result in probe['results'].values():
                self.assertEqual(result['post_request_handles'], 471)
                self.assertEqual(result['maximum_absolute_difference'], 0)
                self.assertEqual(len(result['samples']), 3)
                for sample in result['samples']:
                    self.assertGreater(sample['seconds'], 0)
                    self.assertGreaterEqual(sample['seconds'], sample.get('host_to_native_seconds', 0))

    def test_independent_vision_gates(self):
        for fixture in ('tiny', 'real', 'real-padded'):
            for device in ('cpu', 'gpu'):
                report = read(f'vision-{fixture}-{device}.json')
                self.assertEqual(report['native_device'], device)
                self.assertEqual(report['execution_dtype'], 'float32')
                self.assertEqual(report['checks'], 2498 if fixture == 'tiny' else 3970067)
                self.assertEqual(set(report['max_absolute_error']),
                                 {'patch-projection', 'embeddings', 'first-layer', 'vision', 'shuffle', 'connector'})


if __name__ == '__main__':
    unittest.main()
