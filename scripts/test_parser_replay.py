"""Offline integrity and real SBCL parser-replay checks; no dataset or model needed."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from replay_dpbench import ROOT, digest, parse_records, verify_files


class ParserReplayTests(unittest.TestCase):
    def test_reference_hashes(self):
        folder = ROOT / 'tests/fixtures/headers-footnotes'
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual(manifest['docling-core'], '2.97.0')
        verify_files(folder, manifest['files'])

    def test_records(self):
        records = parse_records('compile noise\nREPLAY\t0\t1\t2\t\nREPLAY\t1\t0\t1\tunclosed-tag\n', 2)
        self.assertTrue(records[0]['markdown_written'])
        self.assertEqual(records[1]['diagnostics'], [{'code': 'unclosed-tag'}])

    def test_recorded_replay_keeps_original_quality(self):
        baseline_path = ROOT / 'tests/fixtures/dpbench/report.json'
        baseline = json.loads(baseline_path.read_text())
        replay = json.loads((ROOT / 'tests/fixtures/dpbench-parser-replay/report.json').read_text())
        self.assertEqual(replay['baseline_report_sha256'], digest(baseline_path))
        self.assertEqual(replay['original_artifacts_sha256'], baseline['retained_sha256'])
        self.assertEqual(replay['manifest_sha256'], baseline['manifest_sha256'])
        self.assertEqual(baseline['summary']['strict_exports'], 0)
        self.assertEqual(replay['summary']['strict_exports'], 1)
        self.assertEqual(replay['summary']['accepted'], 0)
        for name, result in replay['cases'].items():
            self.assertNotIn('raw', result)
            self.assertNotIn('tokens', result)
            for key in ('stop_reason', 'generated_tokens', 'expected_tables', 'predicted_tables',
                        'raw_text_character_edits', 'raw_text_reference_characters',
                        'raw_text_word_edits', 'raw_text_reference_words'):
                self.assertEqual(result[key], baseline['cases'][name][key])
            self.assertEqual(result['strict_export'], name == 'page-001')
            self.assertNotIn('unsupported-tag', result['diagnostic_codes'])
            self.assertFalse(result['accepted'])
        for metric, edits, total in (
            ('markdown_cer', 'character_edits', 'reference_characters'),
            ('markdown_wer', 'word_edits', 'reference_words'),
            ('raw_text_cer', 'raw_text_character_edits', 'raw_text_reference_characters'),
            ('raw_text_wer', 'raw_text_word_edits', 'raw_text_reference_words'),
        ):
            cases = replay['cases'].values()
            self.assertEqual(replay['summary'][metric], sum(r[edits] for r in cases) / sum(r[total] for r in cases))
            if metric.startswith('raw_'):
                self.assertEqual(replay['summary'][metric], baseline['summary'][metric])

    def test_invalid_records(self):
        for value in ('', 'REPLAY\t1\t1\t2\t\n', 'REPLAY\t0\t1\t2\tbad\n',
                      'REPLAY\t0\t0\t2\t\n', 'REPLAY\t0\t2\t2\t\n',
                      'REPLAY\t0\t1\t-1\t\n', 'REPLAY\t0\t1\t2\t\n' * 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_records(value, 1)

    def test_file_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'raw'
            path.write_text('unchanged')
            files = {'raw': digest(path)}
            verify_files(root, files)
            path.write_text('tampered')
            with self.assertRaises(ValueError):
                verify_files(root, files)
            for name in ('../outside', str(path.resolve())):
                with self.assertRaises(ValueError):
                    verify_files(root, {name: 'ignored'})

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = root / 'evidence'
            evidence.mkdir()
            outside = root / 'outside'
            outside.write_text('not evidence')
            (evidence / 'raw').symlink_to(outside)
            with self.assertRaises(ValueError):
                verify_files(evidence, {'raw': digest(outside)})

    def test_native_replay_retains_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'output'
            output.mkdir()
            raw = '<doctag><page_header>Header 日本語</page_header><footnote>1 Note.</footnote></doctag>'
            good, bad = root / 'good', root / 'bad'
            good.write_text(raw)
            bad.write_text('<doctag><footnote><literal angle snippet></footnote></doctag>')
            before = digest(good), digest(bad)
            command = ['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
                       'scripts/replay-dpbench.lisp', str(output), str(good), str(bad)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30, check=True)
            records = parse_records(result.stdout, 2)
            self.assertTrue(records[0]['markdown_written'])
            self.assertFalse(records[1]['markdown_written'])
            self.assertIn({'code': 'malformed-token'}, records[1]['diagnostics'])
            self.assertEqual((output / 'page-000.md').read_text(), 'Header 日本語\n\n1 Note.\n')
            self.assertFalse((output / 'page-001.md').exists())
            self.assertEqual((digest(good), digest(bad)), before)
            # Never overwrite an earlier export.
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == '__main__':
    unittest.main()
