"""Replay the public-clone installation and tutorial evidence, without a model."""
import hashlib
import json
import unittest
from install_contract import ROOT, validate_pair

DIRECTORY = ROOT / 'tests/fixtures/public-install'


class PublicInstallTests(unittest.TestCase):
    def test_public_clone_cpu_and_metal(self):
        reports = [json.loads((DIRECTORY / name).read_text()) for name in ('cpu.json', 'metal.json')]
        validate_pair(*reports)
        for report in reports:
            self.assertEqual(report['source_systems']['cl-docling']['version'], '0.25.0')

    def test_release_provenance(self):
        report = json.loads((DIRECTORY / 'release.json').read_text())
        self.assertEqual(report['engine_commit'], '159ea7c67026c2388622ff2212db242de5a224a2')
        self.assertEqual(report['document_commit'], '305b166225c7aa7d7f6b9cb4082ed0ab74b21c45')
        self.assertTrue(report['fresh_public_clones'])
        self.assertFalse(report['copied_compiled_artifacts'])
        self.assertFalse(report['second_machine'])
        self.assertFalse(report['model_weight_upload'])
        self.assertEqual(report['model_files_verified'], 13)
        for path, digest in report['client_input_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)

    def test_tutorial_merged_model_python_tokens(self):
        native = json.loads((DIRECTORY / 'tutorial-native.json').read_text())
        python = json.loads((DIRECTORY / 'tutorial-python.json').read_text())
        self.assertEqual(set(native['cases']), {'native-page', 'grid', 'merged'})
        self.assertEqual(set(native['cases']), set(python['cases']))
        for name, case in native['cases'].items():
            for key in ('tokens', 'raw', 'stop_reason', 'prompt_ids', 'tiles'):
                self.assertEqual(case[key], python['cases'][name][key], (name, key))
            self.assertTrue(case['public_api_equal'])
            self.assertEqual(len(case['samples']), 3)
            self.assertEqual(len(python['cases'][name]['samples']), 3)


if __name__ == '__main__':
    unittest.main()
