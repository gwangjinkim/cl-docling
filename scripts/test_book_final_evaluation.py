"""Offline M5.11 final-evaluation contracts; never run a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from book_final_evaluation import (
    FROZEN_MANIFEST, POLICY, summarize, validate_final_manifest, validate_policy,
)
from book_final_python import validate_native_report


class BookFinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(POLICY.read_text())
        self.manifest = json.loads(FROZEN_MANIFEST.read_text())

    def test_protocol_is_one_way_base_identity(self):
        self.assertTrue(validate_policy(self.policy))
        self.assertEqual(self.policy['selected_kind'], 'base')
        self.assertEqual(self.policy['native_runs'], 1)
        self.assertEqual(self.policy['parity_cases'], self.policy['final_cases'])

    def test_complete_hash_bound_final_manifest(self):
        self.assertTrue(validate_final_manifest(self.manifest, self.policy))
        self.assertEqual([case['decision'] for case in self.manifest['cases']],
                         ['picture-reference', 'picture-reference', 'text-reference'])
        self.assertTrue(self.manifest['reference_complete'])
        self.assertFalse(self.manifest['model_inference'])

    def test_protocol_and_manifest_mutations_fail_closed(self):
        changed = copy.deepcopy(self.policy)
        changed['selected_kind'] = 'adapter'
        with self.assertRaises(ValueError):
            validate_policy(changed)
        changed = copy.deepcopy(self.manifest)
        changed['cases'].pop()
        with self.assertRaises(ValueError):
            validate_final_manifest(changed, self.policy)

    def test_summary_keeps_parser_and_recognition_separate(self):
        rows = {
            'a': dict(strict_export=True, accepted=True, character_edits=0,
                      reference_characters=10, word_edits=0, reference_words=2),
            'b': dict(strict_export=False, accepted=False, character_edits=20,
                      reference_characters=20, word_edits=4, reference_words=4),
        }
        value = summarize(['a', 'b'], rows)
        self.assertEqual(value['parser_valid_pages'], 1)
        self.assertEqual(value['exact_recognition_pages'], 1)
        self.assertEqual(value['markdown_cer'], 2 / 3)
        self.assertEqual(value['markdown_wer'], 2 / 3)

    def test_python_parity_requires_exact_identity_for_every_final_case(self):
        report = dict(selected_kind='base', base_selected_identity=True,
                      native_inference_runs=1, final={name: {'same_native_output': True}
                                                       for name in self.policy['final_cases']})
        self.assertTrue(validate_native_report(report, self.policy))
        report['final']['final-009']['same_native_output'] = False
        with self.assertRaises(ValueError):
            validate_native_report(report, self.policy)

    def test_recorded_result_preserves_success_and_failure(self):
        evidence = json.loads((FROZEN_MANIFEST.parent / 'report.json').read_text())
        self.assertEqual(evidence['selected_kind'], 'base')
        self.assertEqual(evidence['summary']['final']['parser_valid_pages'], 2)
        self.assertEqual(evidence['summary']['final']['exact_recognition_pages'], 1)
        self.assertEqual(evidence['cases']['final-008']['stop_reason'], 'length')
        self.assertEqual(evidence['cases']['final-008']['tokens'], 2048)
        self.assertTrue(evidence['python_parity']['exact_all_native_ids_and_raw'])
        self.assertEqual(evidence['python_parity']['matched_cases'], 3)


if __name__ == '__main__':
    unittest.main()
