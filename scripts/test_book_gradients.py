"""Offline contracts for a bounded, no-update book gradient probe."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from book_gradients import validate_native, run_child


def authored():
    stage = dict(seconds=1.0, mlx_peak_bytes=1024, mlx_active_bytes=512)
    case = dict(tiles=17, sequence_tokens=1296, gradient_tensors=120, loss=0.5,
                base_and_adapter_unchanged=True, handles_restored=True,
                features=copy.deepcopy(stage), gradients=copy.deepcopy(stage),
                parity_passed=True)
    return dict(device='gpu', remaining_handles=0, updates=0, cases={'page-002': case})


class BookGradientTests(unittest.TestCase):
    def policy(self):
        return dict(cases={'page-002': dict(tiles=17, sequence_tokens=1296)},
                    max_stage_mlx_bytes=4096, gradient_tensors=120)

    def test_valid_no_update_probe(self):
        self.assertTrue(validate_native(authored(), self.policy(), 'gpu'))

    def test_case_loss_or_tile_corruption(self):
        for key, value in [('tiles', 13), ('sequence_tokens', 1295), ('gradient_tensors', 119),
                           ('loss', float('nan')), ('loss', float('inf')), ('loss', -1),
                           ('base_and_adapter_unchanged', False), ('handles_restored', False), ('parity_passed', False)]:
            run = authored()
            run['cases']['page-002'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native(run, self.policy(), 'gpu')

    def test_resource_bound_or_invalid_counters(self):
        for key, value in [('mlx_peak_bytes', 4097), ('mlx_peak_bytes', 1),
                           ('seconds', float('nan')), ('seconds', -1)]:
            run = authored()
            run['cases']['page-002']['gradients'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native(run, self.policy(), 'gpu')

    def test_no_dropped_cases_device_fallback_update_or_leak(self):
        for key, value in [('cases', {}), ('device', 'cpu'), ('updates', 1), ('remaining_handles', 1)]:
            run = authored()
            run[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native(run, self.policy(), 'gpu')

    def test_foreground_runner_waits_and_keeps_failure_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = run_child([sys.executable, '-c', 'print("done")'], os.environ,
                               path / 'ok.log', 5)
            self.assertEqual((path / 'ok.log').read_text().strip(), 'done')
            self.assertGreaterEqual(result['wall_seconds'], 0)
            with self.assertRaises(subprocess.CalledProcessError):
                run_child([sys.executable, '-c', 'raise RuntimeError("intentional")'], os.environ,
                          path / 'failed.log', 5)
            self.assertIn('intentional', (path / 'failed.log').read_text())

    def test_timeout_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'timeout.log'
            with self.assertRaises(subprocess.TimeoutExpired):
                run_child([sys.executable, '-c', 'import os,time; print(os.getpid(), flush=True); time.sleep(30)'],
                          os.environ, log, 1)
            pid = int(log.read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_retained_six_page_device_checks(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / 'tests/fixtures/book-gradients/report.json').read_text())
        policy_path = root / 'references/book-gradients.lock.json'
        self.assertEqual(report['policy'], json.loads(policy_path.read_text()))
        for key, path in [('policy_sha256', policy_path),
                          ('context_report_sha256', root / 'tests/fixtures/book-context/report.json'),
                          ('model_lock_sha256', root / 'references/smoldocling.lock.json')]:
            self.assertEqual(report[key], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(report['training_ready'])
        self.assertTrue(report['gradient_probe_passed'])
        self.assertEqual(report['updates'], 0)
        self.assertEqual(set(report['native']), {'gpu', 'cpu'})
        self.assertEqual(set(report['process_memory']), {'python', 'gpu', 'cpu'})
        for device, run in report['native'].items():
            self.assertTrue(validate_native(run, report['policy'], device))
            for name, case in run['cases'].items():
                self.assertLessEqual(case['loss_error'], report['policy']['loss_atol'])
                self.assertAlmostEqual(case['loss_error'], abs(case['loss'] - report['python']['cases'][name]['loss']))
                # Max absolute errors may exceed atol: native checked every value with atol + rtol * abs(reference).
                self.assertGreaterEqual(case['max_feature_error'], 0)
                self.assertGreaterEqual(case['max_gradient_error'], 0)
                for stage in ('features', 'gradients'):
                    self.assertGreaterEqual(case[stage]['mlx_cache_bytes'], 0)
        for process in report['process_memory'].values():
            self.assertLessEqual(process['whole_process_peak_rss_bytes'], report['policy']['max_child_rss_bytes'])


if __name__ == '__main__':
    unittest.main()
