"""Model-free provenance and unchanged-quality regression checks."""
import hashlib
import json
from pathlib import Path
import unittest

from public_pdf_score import score_page


class FooterEvidenceTests(unittest.TestCase):
    def test_independent_reference_and_source_hashes(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'tests/fixtures/footers'
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual(manifest['docling-core'], '2.97.0')
        for name, sha in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), sha)
        for name, sha in manifest['replay_source_sha256'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), sha)

    def test_export_does_not_correct_recognition(self):
        root = Path(__file__).resolve().parents[1] / 'tests/fixtures'
        reports = json.loads((root / 'footers/scores.json').read_text())
        expected = json.loads((root / 'public-pdf/expected.json').read_text())
        self.assertEqual(set(expected), set(reports))
        for name, rows in expected.items():
            result = json.loads((root / f'public-pdf/results/native/{name}.json').read_text())
            self.assertFalse(result['markdown_written'])  # Historical failure is immutable.
            self.assertEqual(len(result['diagnostics']), 2)
            result['diagnostics'], result['markdown_written'] = [], True
            score = json.loads(json.dumps(score_page(rows, result)))
            self.assertEqual(score, reports[name])
            self.assertFalse(score['table_accepted'])


if __name__ == '__main__':
    unittest.main()
