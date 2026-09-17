import hashlib
import json
from pathlib import Path
import runpy
import unittest


class BundleReferenceTests(unittest.TestCase):
    def test_exact_generation_metadata(self):
        root = Path(__file__).resolve().parents[1]
        fixture_data = runpy.run_path(str(root / 'scripts/bundle-reference.py'))['fixture_data']
        text, hashes = fixture_data(root)
        folder = root / 'tests/fixtures/bundles'
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual((folder / 'generation.sexp').read_text(), text)
        self.assertEqual(manifest['source_sha256'], hashes)
        self.assertEqual(manifest['generation_sha256'], hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    unittest.main()
