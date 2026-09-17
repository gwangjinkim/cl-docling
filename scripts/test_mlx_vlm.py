"""Dependency-free policy tests; numerical gates run in the optional environment."""
import copy
import hashlib
import json
import unittest
from benchmark_contract import ROOT, CASES, compare_runs
from mlx_vlm_contract import ATOL, RTOL, SOURCE_REVISION, validate_config, compare_output


def config():
    return dict(model_type='idefics3', image_token_id=49190, scale_factor=4,
                text_config=dict(hidden_size=576, num_hidden_layers=30, num_attention_heads=9,
                                 num_key_value_heads=3, rope_theta=100000, vocab_size=49280),
                vision_config=dict(hidden_size=768, image_size=512, patch_size=16,
                                   num_attention_heads=12))


class MlxVlmTests(unittest.TestCase):
    def test_checkpoint_defaults(self):
        self.assertEqual(validate_config(config()), ('gelu_pytorch_tanh', 1e-6))

    def test_unsupported_semantics(self):
        for section, key, value in [(None, 'scale_factor', 2), (None, 'model_type', 'smolvlm'),
                                    ('vision_config', 'hidden_act', 'relu'),
                                    ('vision_config', 'layer_norm_eps', 1e-5),
                                    ('text_config', 'tie_word_embeddings', True)]:
            data = copy.deepcopy(config())
            (data if section is None else data[section])[key] = value
            with self.assertRaises(ValueError):
                validate_config(data)

    def test_exact_output_gate(self):
        output = dict(prompt_ids=[1], tiles=13, tokens=[4, 49279], raw='x', stop_reason='eos')
        self.assertEqual(compare_output(output, output), [])
        for key, value in [('prompt_ids', [2]), ('tiles', 1), ('tokens', [4]),
                            ('raw', 'y'), ('stop_reason', 'length')]:
            changed = {**output, key: value}
            self.assertEqual(compare_output(changed, output), [key])

    def test_retained_numerics(self):
        folder = ROOT / 'tests/fixtures/mlx-vlm-benchmark'
        report = json.loads((folder / 'numerics.json').read_text())
        self.assertEqual((report['atol'], report['rtol']), (ATOL, RTOL))
        self.assertEqual(report['upstream_source_revision'], SOURCE_REVISION)
        self.assertEqual(set(report['results']), {'stock', 'corrected'})
        for cases in report['results'].values():
            self.assertEqual(set(cases), set(CASES))
            for case in cases.values():
                for key, shape in [('features', [832, 576]), ('last_prefill_logits', [1, 49280])]:
                    self.assertEqual(case[key]['shape'], shape)
                    self.assertTrue(case[key]['passes'])
                    self.assertTrue(case[key]['finite'])
                    self.assertEqual(case[key]['outside_tolerance'], 0)
                    self.assertGreaterEqual(case[key]['maximum_absolute_error'], 0)

    def test_retained_measurements(self):
        folder = ROOT / 'tests/fixtures/mlx-vlm-benchmark'
        report = json.loads((folder / 'report.json').read_text())
        rows = [json.loads((folder / f'{name}.json').read_text()) for name in report['runtime_order']]
        self.assertEqual(compare_runs(rows), report['summary'])
        self.assertEqual(report['environment_lock_sha256'],
                         hashlib.sha256((ROOT / 'references/mlx-vlm/uv.lock').read_bytes()).hexdigest())
        old = json.loads((ROOT / 'tests/fixtures/benchmark/python-cpu.json').read_text())
        numerics = json.loads((folder / 'numerics.json').read_text())
        for name, file in CASES.items():
            self.assertEqual(report['image_sha256'][name], hashlib.sha256((ROOT / file).read_bytes()).hexdigest())
        for row in rows:
            self.assertEqual(row['device'], 'gpu')
            for name, case in row['cases'].items():
                self.assertEqual(compare_output(case, old['cases'][name]), [])
                self.assertEqual(case['tokens'][-1], 49279)
                for sample in case['samples']:
                    self.assertGreaterEqual(sample['mlx_peak_bytes'], sample['mlx_active_bytes'])
                    self.assertGreater(sample['mlx_active_bytes'], 0)
                if row['runtime'] == 'lisp-mlx':
                    self.assertTrue(case['public_api_equal'])
                    self.assertEqual(case['markdown_available'], name != 'merged')
                    self.assertEqual({s['native_handles'] for s in case['samples']}, {471})
                else:
                    mode = row['runtime'].removeprefix('python-mlx-vlm-')
                    self.assertEqual(case['reference_mismatches'], [])
                    self.assertEqual(case['pixel_sha256'], numerics['results'][mode][name]['pixel_sha256'])
        self.assertEqual(rows[1]['corrections'], [])
        self.assertEqual(rows[2]['corrections'], ['vision post LayerNorm epsilon 1e-6'])
        self.assertTrue(all(m['whole_process_peak_rss_bytes'] > 0 for m in report['whole_process_memory'].values()))


if __name__ == '__main__':
    unittest.main()
