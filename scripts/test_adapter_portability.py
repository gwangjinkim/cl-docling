"""Dependency-free tests of explicit, content-verified adapter staging."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from adapter_portability import BASE_FILES, file_identity, read_json, seal, relocate


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'original'
        self.base.mkdir()
        for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'processor_config.json'):
            (self.base / name).write_bytes(name.encode())
        self.adapter = self.root / 'adapter'
        self.adapter.mkdir()
        self.config = dict(base_model_name_or_path=str(self.base) + '/', revision=None,
                           peft_type='LORA', r=2, lora_alpha=4, target_modules='unchanged')
        (self.adapter / 'adapter_config.json').write_text(json.dumps(self.config))
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'adapter factors')
        self.bundle = self.root / 'sealed'
        self.moved = self.root / 'new base with spaces'
        self.output = self.root / 'staged'

    def prepare(self):
        seal(self.base, self.adapter, self.bundle)
        shutil.copytree(self.base, self.moved)

    def test_relocate_without_original_preserves_everything_except_base_path(self):
        self.prepare()
        original = {p.name: p.read_bytes() for p in self.bundle.iterdir()}
        shutil.rmtree(self.base)
        shutil.rmtree(self.adapter)
        receipt = relocate(self.bundle, self.moved, self.output)
        config = json.loads((self.output / 'adapter_config.json').read_text())
        self.assertEqual(config, dict(self.config, base_model_name_or_path=str(self.moved) + '/'))
        self.assertEqual((self.output / 'adapter_model.safetensors').read_bytes(), b'adapter factors')
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.bundle.iterdir()})
        self.assertEqual(receipt['changed_config_fields'], ['base_model_name_or_path'])
        self.assertEqual(set(receipt['base_files']), set(BASE_FILES))

    def test_every_present_or_absent_base_file_is_bound(self):
        self.prepare()
        for name in BASE_FILES:
            with self.subTest(name=name):
                path = self.moved / name
                before = path.read_bytes() if path.exists() else None
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    relocate(self.bundle, self.moved, self.output)
                self.assertFalse(self.output.exists())
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(before)

    def test_missing_optional_asset_is_rejected(self):
        self.prepare()
        (self.moved / 'tokenizer.json').unlink()
        with self.assertRaises(ValueError):
            relocate(self.bundle, self.moved, self.output)

    def test_adapter_tampering_is_rejected(self):
        self.prepare()
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = self.bundle / name
            before = path.read_bytes()
            path.write_bytes(before + b' ')
            with self.assertRaises(ValueError):
                relocate(self.bundle, self.moved, self.output)
            self.assertFalse(self.output.exists())
            path.write_bytes(before)

    def test_source_identity_and_revision_are_not_guessed(self):
        for changes in ({'base_model_name_or_path': '/other/base/'},
                        {'base_model_name_or_path': 'relative/path'}, {'revision': 'main'}):
            with self.subTest(changes=changes):
                (self.adapter / 'adapter_config.json').write_text(json.dumps(dict(self.config, **changes)))
                with self.assertRaises(ValueError):
                    seal(self.base, self.adapter, self.bundle)
                self.assertFalse(self.bundle.exists())

    def test_existing_or_symlink_destination_is_never_overwritten(self):
        self.prepare()
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            relocate(self.bundle, self.moved, self.output)
        self.assertEqual(marker.read_bytes(), b'keep')
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'nonexistent')
        with self.assertRaises(FileExistsError):
            relocate(self.bundle, self.moved, link)

    def test_symlink_input_files_and_unknown_checkpoint_entries_rejected(self):
        self.prepare()
        path = self.moved / 'tokenizer.json'
        path.unlink()
        path.symlink_to(self.base / 'tokenizer.json')
        with self.assertRaises(ValueError):
            relocate(self.bundle, self.moved, self.output)
        path.unlink()
        shutil.copyfile(self.base / path.name, path)
        for name in ('extra.safetensors', 'config.json.backup', '.cache'):
            path = self.moved / name
            path.write_bytes(b'extra')
            with self.assertRaises(ValueError):
                relocate(self.bundle, self.moved, self.output)
            path.unlink()

    def test_manifest_cannot_omit_inventory_or_inject_paths(self):
        self.prepare()
        path = self.bundle / 'portable_adapter.json'
        good = path.read_text()
        for mutation in ('omit', 'path', 'version', 'digest'):
            lock = json.loads(good)
            if mutation == 'omit':
                del lock['base_files']['tokenizer.json']
            elif mutation == 'path':
                lock['base_files']['../outside'] = lock['base_files'].pop('tokenizer.json')
            elif mutation == 'version':
                lock['schema_version'] = True
            else:
                lock['base_files']['tokenizer.json']['sha256'] = 'bad'
            path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError):
                relocate(self.bundle, self.moved, self.output)
            self.assertFalse(self.output.exists())

    def test_duplicate_json_keys_rejected(self):
        (self.adapter / 'adapter_config.json').write_text('{"revision":null,"revision":null}')
        with self.assertRaises(ValueError):
            seal(self.base, self.adapter, self.bundle)

    def test_output_cannot_modify_an_input_directory(self):
        for source in (self.base, self.adapter):
            with self.assertRaises(ValueError):
                seal(self.base, self.adapter, source / 'output')
        self.prepare()
        for source in (self.moved, self.bundle):
            with self.assertRaises(ValueError):
                relocate(self.bundle, self.moved, source / 'output')

    def test_missing_required_base_file_is_rejected_before_output(self):
        (self.base / 'model.safetensors').unlink()
        with self.assertRaises(ValueError):
            seal(self.base, self.adapter, self.bundle)
        self.assertFalse(self.bundle.exists())

    def test_huggingface_download_metadata_is_not_model_content(self):
        cache = self.base / '.cache'
        cache.mkdir()
        (cache / 'download-metadata').write_bytes(b'local download bookkeeping')
        self.prepare()
        shutil.rmtree(self.moved / '.cache')
        relocate(self.bundle, self.moved, self.output)

    def test_same_path_staging_does_not_claim_a_config_value_change(self):
        self.prepare()
        receipt = relocate(self.bundle, self.base, self.output)
        self.assertEqual(receipt['changed_config_fields'], [])
        self.assertEqual(read_json(self.output / 'adapter_config.json'), self.config)


class RecordedQualificationTests(unittest.TestCase):
    def test_inventory_covers_the_native_loader_assets(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'src/model.lisp').read_text()
        declaration = source.split('(defparameter +document-asset-names+')[1].split('(defun snapshot-document-assets')[0]
        assets = set(re.findall(r'"([^"]+)"', declaration))
        self.assertEqual(set(BASE_FILES), assets | {'config.json', 'model.safetensors'})

    def test_actual_selected_artifact_and_base_remained_exact(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'tests/fixtures/adapter-portability'
        manifest = read_json(evidence / 'portable_adapter.json')
        receipt = read_json(evidence / 'relocation.json')
        lock = read_json(root / 'references/smoldocling.lock.json')
        selected = read_json(root / 'tests/fixtures/selection-results/selection.json')
        self.assertEqual({k: v for k, v in manifest['base_files'].items() if v is not None}, lock['files'])
        self.assertEqual(receipt['base_files'], manifest['base_files'])
        self.assertEqual(receipt['source_manifest'], file_identity(evidence / 'portable_adapter.json'))
        self.assertEqual(receipt['source_adapter_files'], manifest['adapter_files'])
        self.assertEqual({k: v['sha256'] for k, v in manifest['adapter_files'].items()}, selected['adapter_sha256'])
        self.assertEqual(receipt['source_adapter_files']['adapter_model.safetensors'],
                         receipt['staged_adapter_files']['adapter_model.safetensors'])
        before = read_json(evidence / 'source-adapter-config.json')
        after = read_json(evidence / 'staged-adapter-config.json')
        self.assertNotEqual(before['base_model_name_or_path'], after['base_model_name_or_path'])
        self.assertEqual(after, dict(before, base_model_name_or_path=receipt['target_base']))
        for label, field in (('source', 'source_adapter_files'), ('staged', 'staged_adapter_files')):
            self.assertEqual(file_identity(evidence / f'{label}-adapter-config.json'),
                             receipt[field]['adapter_config.json'])

    def test_all_recorded_consumers_match_selected_outputs(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'tests/fixtures/adapter-portability'
        for label in ('native-cpu', 'native-metal', 'python'):
            report = read_json(evidence / f'{label}.json')
            self.assertEqual(set(report['cases']), {'library', 'pets', 'fruit', 'tea'})
            for name, case in report['cases'].items():
                expected = read_json(root / f'tests/fixtures/selection-results/final/adapted/{name}.json')
                for key in ('tokens', 'raw', 'stop_reason'):
                    self.assertEqual(case[key], expected[key])
            if label.startswith('native'):
                self.assertEqual(report['device'], 'cpu' if label == 'native-cpu' else 'gpu')
                self.assertIn('base identity does not match', report['original_rejection'])
                self.assertEqual(report['remaining_handles'], 0)
                self.assertEqual(report['adapter_factors'], 120)
            else:
                self.assertEqual(report['exact_adapter_factors'], 120)
                self.assertTrue(report['all_selected_outputs_equal'])


if __name__ == '__main__':
    unittest.main()
