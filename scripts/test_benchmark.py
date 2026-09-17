import hashlib
import json
import unittest
from benchmark_contract import ROOT, summary, validate_run, compare_runs, CASES


def run():
    row = dict(runtime='test', device='cpu', dtype='float32', max_new_tokens=512,
               task='Convert this page to docling.', warmups=1, repeats=3, cases={})
    for name in CASES:
        row['cases'][name] = dict(tokens=[1, 2], raw='text', stop_reason='eos', prompt_ids=[3], tiles=13,
                                 samples=[dict(preprocess=1., vision=2., prefill=3., decode=4., detokenize=.1,
                                               generation=10.2) for _ in range(3)])
    return row


class BenchmarkTests(unittest.TestCase):
    def test_statistics(self):
        self.assertEqual(summary([3, 1, 2]), dict(median=2, minimum=1, maximum=3))

    def test_validation(self):
        validate_run(run())
        for key, value in [('vision', float('nan')), ('decode', -1), ('generation', 1)]:
            data = run()
            data['cases']['grid']['samples'][0][key] = value
            with self.assertRaises(ValueError):
                validate_run(data)
        data = run()
        data['cases']['grid']['samples'].pop()
        with self.assertRaises(ValueError):
            validate_run(data)

    def test_exact_workload_required(self):
        compare_runs([run(), run()])
        for field, value in [('tokens', [1, 9]), ('raw', 'other'), ('prompt_ids', [4]), ('tiles', 1)]:
            data = run()
            data['cases']['grid'][field] = value
            with self.assertRaises(ValueError):
                compare_runs([run(), data])
        data = run()
        data['dtype'] = 'float16'
        with self.assertRaises(ValueError):
            compare_runs([run(), data])

    def test_warmup_excluded(self):
        data = run()
        for case in data['cases'].values():
            case['warmup'] = {'generation': 9999.0}
        result = compare_runs([data, run()])
        self.assertEqual(result['test/cpu']['grid']['generation']['median'], 10.2)

    def test_recorded_measurements(self):
        folder = ROOT / 'tests/fixtures/benchmark'
        report = json.loads((folder / 'report.json').read_text())
        rows = [json.loads((folder / f'{name}.json').read_text())
                for name in ('lisp-cpu', 'python-cpu', 'lisp-metal')]
        self.assertEqual(compare_runs(rows), report['summary'])
        for name, path in CASES.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), report['image_sha256'][name])
        for row in rows:
            for name, count in [('native-page', 152), ('grid', 56), ('merged', 53)]:
                case = row['cases'][name]
                self.assertEqual(len(case['tokens']), count)
                self.assertEqual(case['stop_reason'], 'eos')
                self.assertEqual(case['tiles'], 13)
                self.assertEqual(len(case['prompt_ids']), 878 if name == 'native-page' else 877)
                if name != 'native-page':
                    previous = json.loads((ROOT / 'tests/fixtures/table-recognition/metal' / f'{name}.json').read_text())
                    self.assertEqual(case['raw'], previous['raw'])
                    self.assertEqual(case['tokens'], previous['tokens'])
                if row['runtime'] == 'lisp-mlx':
                    self.assertTrue(case['public_api_equal'])
                    self.assertEqual(case['markdown_available'], name != 'merged')
                    self.assertEqual(case['diagnostics'], 3 if name == 'merged' else 0)
                    self.assertEqual({s['native_handles'] for s in case['samples']}, {471})
                    self.assertTrue(all(s['mlx_peak_bytes'] >= s['mlx_active_bytes'] > 0 for s in case['samples']))
        self.assertTrue(all(m['whole_process_peak_rss_bytes'] > 0 for m in report['whole_process_memory'].values()))


if __name__ == '__main__':
    unittest.main()
