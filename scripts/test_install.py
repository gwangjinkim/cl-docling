import copy
import hashlib
import json
import unittest
from install_contract import ROOT, validate_install, validate_pair


class InstallEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'tests/fixtures/clean-install'
        self.cpu = json.loads((directory / 'cpu.json').read_text())
        self.metal = json.loads((directory / 'metal.json').read_text())

    def test_exact_outputs_and_clean_sources(self):
        validate_pair(self.cpu, self.metal)

    def test_token_and_processor_changes_are_rejected(self):
        for field in ('tokens', 'tiles', 'prompt_tokens', 'raw', 'markdown', 'stop_reason'):
            changed = copy.deepcopy(self.cpu)
            changed['case'][field] = None
            with self.assertRaises(ValueError):
                validate_install(changed)

    def test_handle_leak_is_rejected(self):
        changed = copy.deepcopy(self.cpu)
        changed['remaining_handles'] = 1
        with self.assertRaises(ValueError):
            validate_install(changed)

    def test_external_libraries_and_dependencies_are_rejected(self):
        changed = copy.deepcopy(self.cpu)
        changed['direct_foreign_libraries'][0] = '/old/build/libdocling_images.dylib'
        with self.assertRaises(ValueError):
            validate_install(changed)
        changed = copy.deepcopy(self.cpu)
        changed['source_systems']['cffi']['source'] = '/old/build/cffi/'
        with self.assertRaises(ValueError):
            validate_install(changed)

    def test_cpu_does_not_substitute_for_metal(self):
        with self.assertRaises(ValueError):
            validate_pair(self.cpu, self.cpu)

    def test_missing_or_wrong_engine_provenance_is_rejected(self):
        changed = copy.deepcopy(self.cpu)
        del changed['source_systems']['yason']
        with self.assertRaises(ValueError):
            validate_install(changed)
        changed = copy.deepcopy(self.cpu)
        changed['source_systems']['cl-transformer-blocks']['version'] = '0.41.1'
        with self.assertRaises(ValueError):
            validate_install(changed)

    def test_build_record_matches_versioned_inputs(self):
        report = json.loads((ROOT / 'tests/fixtures/clean-install/build.json').read_text())
        for key, path in {'document_uv_lock': 'uv.lock', 'model_lock': 'references/smoldocling.lock.json',
                          'install_check': 'scripts/check-native-install.lisp',
                          'image_build': 'scripts/build-images.sh'}.items():
            self.assertEqual(report['hashes'][key], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        self.assertEqual(len(report['source_archive_sha256']), 8)
        self.assertEqual(report['model_files_verified'], 13)
        self.assertEqual(report['image_checks'], 80)
        self.assertEqual(report['exact_image_fp32_comparisons'], 72192)
        self.assertFalse(report['publication_performed'])


if __name__ == '__main__':
    unittest.main()
