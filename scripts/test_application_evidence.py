"""Replay the recorded application smoke runs, not a general bundle reader."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/application'


class ApplicationEvidenceTests(unittest.TestCase):
    def test_cpu_metal_and_resume_keep_exact_generation(self):
        for page, count, reason in ((1, 152, 'EOS'), (2, 512, 'LENGTH')):
            name = f'page-{page}'
            cpu = FIXTURES / 'cpu' / 'export'
            gpu = FIXTURES / 'gpu' / 'export'
            for suffix in ('.doctags', '.sexp'):
                self.assertEqual((cpu / (name + suffix)).read_bytes(), (gpu / (name + suffix)).read_bytes())
            for device in ('cpu', 'gpu'):
                folder = FIXTURES / device
                for suffix in ('.doctags', '.sexp'):
                    self.assertEqual((folder / 'export' / (name + suffix)).read_bytes(),
                                     (folder / 'resumed-export' / (name + suffix)).read_bytes())
                metadata = (folder / 'export' / (name + '.sexp')).read_text()
                match = re.search(r':TOKEN-IDS #\(([0-9 ]*)\) :STOP-REASON :(EOS|LENGTH)', metadata)
                self.assertIsNotNone(match)
                self.assertEqual(len(match[1].split()), count)
                self.assertEqual(match[2], reason)

    def test_blocked_and_clean_exports_remain_distinct(self):
        for device in ('cpu', 'gpu'):
            folder = FIXTURES / device
            self.assertIn(':STATUS :BLOCKED', (folder / 'export/manifest.sexp').read_text())
            self.assertIn(':GENERATION-TRUNCATED', (folder / 'export/page-2.sexp').read_text())
            self.assertEqual(list((folder / 'export').glob('*.md')), [])
            self.assertNotIn('Converting page', (folder / 'resume.log').read_text())
            self.assertIn('Remaining tensor handles: 0', (folder / 'run.log').read_text())
        clean = FIXTURES / 'clean/export'
        self.assertIn(':STATUS :COMPLETE', (clean / 'manifest.sexp').read_text())
        self.assertEqual((clean / 'page-1.md').read_bytes(), (ROOT / 'tests/fixtures/doctags/native-page.md').read_bytes())

    def test_executed_sources_and_limits(self):
        report = json.loads((FIXTURES / 'report.json').read_text())
        self.assertFalse(report['quality_improvement_claim'])
        self.assertEqual(report['exit_codes'], {'cpu': 2, 'gpu': 2, 'resume_cpu': 2, 'resume_gpu': 2, 'clean': 0})
        for path, digest in report['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)


if __name__ == '__main__':
    unittest.main()
