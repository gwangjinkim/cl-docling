"""Offline guard tests; importing/running these must never open final page content."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from book_final_guard import ROOT, POLICY, validate_selection_seal
from replay_dpbench import digest


class BookFinalGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.selection = self.root / 'selection'
        self.selected = self.root / 'selected'
        self.selection.mkdir(); self.selected.mkdir()
        self.protocol = self.root / 'selection.lock.json'
        self.protocol.write_text('{"frozen":true}\n')
        (self.selection / 'report.json').write_text('{"winner":"step-6"}\n')
        (self.selected / 'adapter.bin').write_bytes(b'adapter')
        policy = json.loads(POLICY.read_text())
        self.seal = dict(schema_version=1, status='selection-sealed', selection_criterion='validation-only',
                         final_content_opened=False, workflow_sha256=digest(POLICY),
                         selection_protocol_sha256=digest(self.protocol),
                         selection_report='report.json',
                         selection_report_sha256=digest(self.selection / 'report.json'),
                         selected_id='step-6', selected_kind='adapter',
                         selected_files={'adapter.bin': digest(self.selected / 'adapter.bin')},
                         final_cases=[case['name'] for case in policy['cases']],
                         development_evidence=policy['required_development_evidence'],
                         source_revision='a' * 40)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, seal=None):
        return validate_selection_seal(self.seal if seal is None else seal,
                                       self.protocol, self.selection, self.selected)

    def test_valid_seal_binds_every_pre_final_input(self):
        result = self.validate()
        self.assertEqual(result['selected_id'], 'step-6')
        self.assertFalse(result['final_content_opened'])
        self.assertEqual(result['final_cases'], ['final-008', 'final-009', 'final-010'])

    def test_selection_and_final_state_mutations_fail_closed(self):
        changes = [('status', 'draft'), ('selection_criterion', 'training-loss'),
                   ('final_content_opened', True), ('workflow_sha256', '0' * 64),
                   ('selection_protocol_sha256', '0' * 64), ('selection_report_sha256', '0' * 64),
                   ('selected_id', ''), ('selected_kind', 'unknown'), ('source_revision', 'short'),
                   ('final_cases', ['final-008'])]
        for key, value in changes:
            seal = copy.deepcopy(self.seal); seal[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(seal)

    def test_changed_evidence_or_artifact_fails_closed(self):
        for key in self.seal['development_evidence']:
            seal = copy.deepcopy(self.seal); seal['development_evidence'][key] = '0' * 64
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(seal)
        (self.selected / 'adapter.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.validate()

    def test_base_fallback_has_no_adapter_payload(self):
        seal = copy.deepcopy(self.seal)
        seal.update(selected_id='base', selected_kind='base', selected_files={})
        self.assertEqual(self.validate(seal)['selected_kind'], 'base')
        seal['selected_files'] = {'adapter.bin': hashlib.sha256(b'adapter').hexdigest()}
        with self.assertRaises(ValueError): self.validate(seal)

    def test_paths_cannot_escape_or_follow_symlinks(self):
        for field, value in [('selection_report', '../report.json')]:
            seal = copy.deepcopy(self.seal); seal[field] = value
            with self.assertRaises(ValueError): self.validate(seal)
        link = self.selected / 'link.bin'
        link.symlink_to(self.selected / 'adapter.bin')
        seal = copy.deepcopy(self.seal)
        seal['selected_files'] = {'link.bin': digest(link)}
        with self.assertRaises(ValueError): self.validate(seal)

    def test_policy_keeps_final_metadata_only_and_unopened(self):
        policy = json.loads(POLICY.read_text())
        self.assertEqual(policy['status'], 'guard-ready-final-unopened')
        self.assertFalse(policy['training_ready'])
        self.assertEqual([case['row'] for case in policy['cases']], [8, 9, 10])
        self.assertEqual(set(policy), {'schema_version', 'scope', 'source', 'pdf_sha256', 'cases',
                                      'render_dpi', 'poppler_version', 'required_development_evidence',
                                      'unlock', 'reference_policy', 'evaluation_policy', 'failure_policy',
                                      'status', 'training_ready'})
        self.assertNotIn('markdown', POLICY.read_text().lower())

    def test_versioned_guard_evidence_has_no_false_final_claim(self):
        report = json.loads((ROOT / 'tests/fixtures/book-final-guard/report.json').read_text())
        self.assertEqual(report['workflow_sha256'], digest(POLICY))
        self.assertEqual(report['split_report_sha256'],
                         digest(ROOT / 'tests/fixtures/book-splits/report.json'))
        for name, wanted in report['implementation_sha256'].items():
            self.assertEqual(digest(ROOT / name), wanted)
        self.assertTrue(report['selection_seal_required'])
        self.assertFalse(report['reference_complete'])
        self.assertFalse(report['final_content_opened'])
        self.assertFalse(report['model_inference'])
        self.assertEqual([case['name'] for case in report['cases']],
                         ['final-008', 'final-009', 'final-010'])


if __name__ == '__main__':
    unittest.main()
