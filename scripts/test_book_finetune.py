"""Offline selection/protocol tests; no model, GPU or final data."""
import copy
import json
from pathlib import Path
import unittest

from book_finetune import (BASELINE_REPORT, PICTURE_EXPORT_REPORT, PICTURE_REPORT,
                           POLICY, candidate_summary, choose_candidate, validate_manifest,
                           validate_training)
from replay_dpbench import digest


def page(accepted=False, characters=10, words=2, strict=True):
    return dict(accepted=accepted, character_edits=characters, word_edits=words,
                strict_export=strict)


def candidate(a=0, characters=30, words=6):
    per_char, remainder = divmod(characters, 3)
    per_word, word_remainder = divmod(words, 3)
    return {f'validation-00{i+5}': page(i < a, per_char + (i < remainder),
                                       per_word + (i < word_remainder)) for i in range(3)}


class BookFinetuneTests(unittest.TestCase):
    def test_candidate_summary_counts_every_fixed_page(self):
        summary = candidate_summary(candidate(2, 11, 5))
        self.assertEqual(summary, dict(pages=3, strict_exports=3, accepted_pages=2,
                                       character_edits=11, word_edits=5))
        missing = candidate(); missing.pop('validation-006')
        with self.assertRaises(ValueError): candidate_summary(missing)

    def test_selection_order_is_frozen_and_base_wins_exact_tie(self):
        reports = {'base': candidate(0, 30, 6), 'step-3': candidate(1, 100, 20),
                   'step-6': candidate(0, 1, 1)}
        selected, _ = choose_candidate(reports)
        self.assertEqual(selected, 'step-3')
        tied = {name: candidate(1, 4, 2) for name in reports}
        self.assertEqual(choose_candidate(tied)[0], 'base')

    def test_character_then_word_then_earliest_checkpoint_break_ties(self):
        reports = {'base': candidate(0, 1, 1), 'step-3': candidate(1, 5, 9),
                   'step-6': candidate(1, 6, 1)}
        self.assertEqual(choose_candidate(reports)[0], 'step-3')
        reports['step-6'] = candidate(1, 5, 8)
        self.assertEqual(choose_candidate(reports)[0], 'step-6')
        reports['step-3'] = candidate(1, 5, 8)
        self.assertEqual(choose_candidate(reports)[0], 'step-3')

    def test_candidate_inventory_cannot_change(self):
        reports = {'base': candidate(), 'step-3': candidate(), 'step-6': candidate()}
        for changed in ({**reports, 'step-9': candidate()}, {'base': candidate()},
                        {3: candidate(), 'base': candidate(), 'step-6': candidate()}):
            with self.assertRaises(ValueError): choose_candidate(changed)

    def test_protocol_is_bounded_and_has_no_final_case(self):
        policy = json.loads(POLICY.read_text())
        self.assertEqual(policy['candidate_steps'], [0, 3, 6])
        self.assertEqual(len(policy['schedule']), 6)
        self.assertEqual(policy['device'], 'gpu')
        self.assertEqual(policy['validation_cases'], ['validation-005', 'validation-006', 'validation-007'])
        self.assertFalse(policy['training_ready'])
        text = POLICY.read_text().lower()
        self.assertNotIn('final-008', text)
        self.assertNotIn('final-009', text)
        self.assertNotIn('final-010', text)

    def test_manifest_requires_exact_complete_development_inventory(self):
        policy = json.loads(POLICY.read_text())
        files = {f'{name}.{suffix}': '0' * 64 for name in policy['training_cases']
                 for suffix in ('png', 'doctags')}
        files.update({f'{name}.png': '0' * 64 for name in policy['validation_cases']})
        files.update({'validation-005.doctags': '0' * 64,
                      'validation-006.expected.json': '0' * 64,
                      'validation-007.expected.json': '0' * 64})
        files.update({f'base/{name}.{suffix}': '0' * 64 for name in policy['validation_cases']
                      for suffix in ('json', 'doctags')})
        manifest = dict(protocol=policy, protocol_sha256=digest(POLICY),
                        training_cases=list(policy['training_cases']),
                        validation_cases=policy['validation_cases'], files=files,
                        final_content_opened=False,
                        source_evidence_sha256=dict(
                            baseline_inputs='6eef9e6fcb29b469e900856b388824be9979b12d60f3aee062915dbc882e7f62',
                            baseline=digest(BASELINE_REPORT), picture=digest(PICTURE_REPORT),
                            picture_export=digest(PICTURE_EXPORT_REPORT)))
        self.assertTrue(validate_manifest(manifest))
        for mutate in [lambda value: value['files'].pop('validation-005.png'),
                       lambda value: value.update(final_content_opened=True),
                       lambda value: value['source_evidence_sha256'].update(picture='0' * 64)]:
            changed = copy.deepcopy(manifest); mutate(changed)
            with self.assertRaises(ValueError): validate_manifest(changed)

    def test_training_trajectory_is_exact_and_finite(self):
        policy = json.loads(POLICY.read_text())
        rows = []
        for step, name in enumerate(policy['schedule'], 1):
            expected = policy['training_cases'][name]
            rows.append(dict(step=step, name=name, tiles=expected['tiles'],
                             sequence_tokens=expected['sequence_tokens'], loss=1.0,
                             base_unchanged=True, adapter_changed=True, handles=10,
                             seconds=1.0, mlx_peak_bytes=100, mlx_active_bytes=50,
                             mlx_cache_bytes=25))
        training = dict(device='gpu', dtype='float32', final_step=6,
                        remaining_handles=0, steps=rows)
        self.assertTrue(validate_training(training, policy))
        for mutate in [lambda value: value['steps'].pop(),
                       lambda value: value['steps'][0].update(loss=float('nan')),
                       lambda value: value['steps'][1].update(base_unchanged=False),
                       lambda value: value['steps'][2].update(mlx_peak_bytes=policy['max_stage_mlx_bytes'] + 1),
                       lambda value: value.update(remaining_handles=1)]:
            changed = copy.deepcopy(training); mutate(changed)
            with self.assertRaises(ValueError): validate_training(changed, policy)

    def test_versioned_manifest_is_the_frozen_complete_input(self):
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               'tests/fixtures/book-finetune/manifest.json').read_text())
        self.assertTrue(validate_manifest(manifest))

    def test_versioned_run_replay_seals_honest_base_fallback(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / 'tests/fixtures/book-finetune/report.json').read_text())
        self.assertEqual(report['protocol_sha256'], digest(POLICY))
        self.assertEqual(report['manifest_sha256'],
                         digest(root / 'tests/fixtures/book-finetune/manifest.json'))
        for name, wanted in report['implementation_sha256'].items():
            self.assertEqual(digest(root / name), wanted)
        self.assertEqual((report['selected_id'], report['selected_kind']), ('base', 'base'))
        self.assertFalse(report['final_content_opened'])
        self.assertTrue(report['replay_only'])
        self.assertFalse(report['model_inference'])
        self.assertEqual(set(report['summaries']), {'base', 'step-3', 'step-6'})
        self.assertEqual(len({tuple(summary.values()) for summary in report['summaries'].values()}), 1)
        self.assertNotEqual(report['retained_sha256']['training/adapter-3/adapter_model.safetensors'],
                            report['retained_sha256']['training/adapter-6/adapter_model.safetensors'])


if __name__ == '__main__':
    unittest.main()
