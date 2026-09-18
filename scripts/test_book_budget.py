"""Offline operational-budget rejection cases; no model or ML dependency."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from book_budget import validate_native, validate_resume, tensor_payloads, compare_checkpoints
from replay_dpbench import digest

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'references/book-budget.lock.json').read_text())


def authored(phase='train'):
    start = 0 if phase == 'train' else POLICY['resume_after']
    steps = []
    for index, name in enumerate(POLICY['schedule'][start:], start + 1):
        steps.append(dict(step=index, name=name, **POLICY['cases'][name], loss=0.5,
                          adapter_changed=True, base_unchanged=True, handles=600,
                          seconds=1.0, mlx_active_bytes=512, mlx_peak_bytes=1024,
                          mlx_cache_bytes=2048, post_cleanup_active_bytes=256))
    return dict(device='gpu', phase=phase, initial_step=start, final_step=6,
                remaining_handles=0, steps=steps)


class BookBudgetTests(unittest.TestCase):
    def test_retained_cpu_metal_training_and_resume(self):
        report = json.loads((ROOT / 'tests/fixtures/book-budget/report.json').read_text())
        self.assertEqual(report['policy'], POLICY)
        self.assertEqual(report['policy_sha256'], digest(ROOT / 'references/book-budget.lock.json'))
        self.assertEqual(report['context_report_sha256'], digest(ROOT / 'tests/fixtures/book-context/report.json'))
        self.assertEqual(report['model_lock_sha256'], digest(ROOT / 'references/smoldocling.lock.json'))
        self.assertFalse(report['training_ready'])
        self.assertTrue(report['resource_probe_passed'])
        self.assertEqual(set(report['native']), {'gpu', 'cpu'})
        self.assertEqual(POLICY['schedule'], list(POLICY['cases'])*2)
        self.assertEqual(set(POLICY['cases']), {'page-000', 'page-002', 'page-004'})
        for name in ('scripts/book_budget.py', 'scripts/book-budget.lisp', 'scripts/book_gradients.py'):
            self.assertEqual(digest(ROOT / name), report['implementation_sha256'][name])
        for device, phases in report['native'].items():
            self.assertEqual(set(phases), {'train', 'resume'})
            for phase, run in phases.items():
                self.assertTrue(validate_native(run, POLICY, device, phase))
                self.assertLessEqual(report['process_memory'][device][phase]['whole_process_peak_rss_bytes'], POLICY['max_child_rss_bytes'])
                self.assertIn(f'{device}/{phase}.json', report['retained_sha256'])
            self.assertTrue(validate_resume(phases['train'], phases['resume'], POLICY))
            for phase in ('restored', 'final'):
                result = report['resume'][device][phase]
                self.assertTrue(result['exact'])
                self.assertEqual(result['tensors'], {'adapter_model.safetensors': 120, 'optimizer.safetensors': 240})

    def test_exact_checkpoint_payloads_and_mutations(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder) / n for n in ('a', 'b')]
            def save(path, value=1.0, dtype='F32'):
                header = json.dumps({'factor': dict(dtype=dtype, shape=[1], data_offsets=[0, 4])}).encode()
                path.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', value))
            for root in roots:
                (root / 'decoder').mkdir(parents=True)
                for name in ('adapter_model.safetensors', 'optimizer.safetensors'):
                    save(root / 'decoder' / name)
                (root / 'decoder/training_state.json').write_text(json.dumps(dict(step=6)))
            policy = dict(adapter_tensors=1, optimizer_tensors=1)
            self.assertTrue(compare_checkpoints(*roots, policy, 6)['exact'])
            target = roots[1] / 'decoder/optimizer.safetensors'
            save(target, 1.0001)
            with self.assertRaises(ValueError): compare_checkpoints(*roots, policy, 6)
            for value, dtype in [(float('nan'), 'F32'), (float('inf'), 'F32'), (1, 'F16')]:
                save(target, value, dtype)
                with self.assertRaises(ValueError): tensor_payloads(target)
            save(target)
            with self.assertRaises(ValueError): compare_checkpoints(*roots, policy, 3)
            with self.assertRaises(ValueError): compare_checkpoints(*roots, dict(adapter_tensors=2, optimizer_tensors=1), 6)

    def test_complete_runs_and_exact_resume_losses(self):
        self.assertTrue(validate_native(authored(), POLICY, 'gpu', 'train'))
        self.assertTrue(validate_native(authored('resume'), POLICY, 'gpu', 'resume'))
        self.assertTrue(validate_resume(authored(), authored('resume'), POLICY))

    def test_reject_step_skip_reorder_or_final_data(self):
        for mutate in [lambda r: r['steps'].pop(),
                       lambda r: r['steps'].reverse(),
                       lambda r: r['steps'][0].update(name='final-008'),
                       lambda r: r.update(initial_step=1)]:
            run = authored()
            mutate(run)
            with self.assertRaises(ValueError): validate_native(run, POLICY, 'gpu', 'train')

    def test_reject_loss_geometry_update_or_base_failure(self):
        for k, v in [('loss', float('nan')), ('loss', float('inf')), ('loss', -1),
                     ('tiles', 1), ('sequence_tokens', 100), ('adapter_changed', False),
                     ('base_unchanged', False), ('handles', 601)]:
            run = authored()
            run['steps'][1][k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_native(run, POLICY, 'gpu', 'train')

    def test_reject_memory_time_device_or_cleanup_failure(self):
        for k, v in [('mlx_peak_bytes', 2**36), ('mlx_cache_bytes', 2**36),
                     ('mlx_active_bytes', 2048), ('mlx_cache_bytes', -1),
                     ('seconds', float('nan')), ('seconds', -1), ('post_cleanup_active_bytes', 2048)]:
            run = authored()
            run['steps'][0][k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_native(run, POLICY, 'gpu', 'train')
        for k, v in [('device', 'cpu'), ('remaining_handles', 1), ('final_step', 5)]:
            run = authored()
            run[k] = v
            with self.assertRaises(ValueError): validate_native(run, POLICY, 'gpu', 'train')

    def test_changed_resume_loss_rejected_without_tolerance(self):
        resumed = authored('resume')
        resumed['steps'][0]['loss'] += 1e-12
        with self.assertRaises(ValueError): validate_resume(authored(), resumed, POLICY)


if __name__ == '__main__':
    unittest.main()
