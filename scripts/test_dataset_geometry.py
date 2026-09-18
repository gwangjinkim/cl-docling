"""Offline tests for explicitly declared pixel-coordinate targets."""
import json
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from dataset_geometry import convert_pixel_locations
from dataset_audit import ROOT, native_probe


class GeometryTests(unittest.TestCase):
    def test_retained_geometry_audit_and_privacy_gate(self):
        report = json.loads((ROOT/'tests/fixtures/dataset-geometry/report.json').read_text())
        previous_path = ROOT/'tests/fixtures/dataset-audit/report.json'
        previous = json.loads(previous_path.read_text())
        self.assertEqual(report['prior_audit_sha256'], hashlib.sha256(previous_path.read_bytes()).hexdigest())
        self.assertEqual(report['summary'], dict(pages=20, pixel_boxes=174,
                                               native_renderable=9, remaining_location_diagnostics=0))
        self.assertFalse(report['training_approved'])
        self.assertEqual(sum(c['markdown_renderable'] for c in report['cases']), 9)
        self.assertEqual(sum(c['box_count'] for c in report['cases']), 174)
        for old, new in zip(previous['cases'], report['cases'], strict=True):
            self.assertEqual(old['id'], new['id'])
            self.assertEqual(old['target_sha256'], new['original_target_sha256'])
            self.assertNotEqual(new['original_target_sha256'], new['normalized_target_sha256'])
            self.assertEqual(set(new['diagnostic_codes']), set(old['diagnostic_codes']) - {'location'})
            self.assertNotIn('raw', new)

    def convert(self, raw, width=1000, height=2000, **kwargs):
        return convert_pixel_locations(raw, width, height,
                                       convention=kwargs.get('convention', 'top-left-pixels'))

    def test_axis_and_endpoint_mapping(self):
        raw = '<text><loc_100><loc_200><loc_1000><loc_2000>Hello</text>'
        self.assertEqual(self.convert(raw), '<text><loc_50><loc_50><loc_499><loc_499>Hello</text>')

    def test_no_inferred_convention(self):
        for convention in (None, 'auto', 'bottom-left-pixels', 'normalized'):
            with self.assertRaises(ValueError):
                self.convert('<text>Hello</text>', convention=convention)

    def test_bounds_and_types(self):
        for width in (0, -1, True, 1.5, 20001):
            with self.assertRaises(ValueError):
                self.convert('<text>Hello</text>', width=width)
        for box in ('1001,0,1001,2000', '3,0,2,4', '0,5,5,4', '0,0,1,2001'):
            raw = '<text>' + ''.join(f'<loc_{v}>' for v in box.split(',')) + 'x</text>'
            with self.assertRaises(ValueError):
                self.convert(raw)

    def test_malformed_or_orphan_locations(self):
        for body in ('<loc_1>', '<loc_-1><loc_0><loc_3><loc_4>',
                     '<loc_1><loc_2>text<loc_3><loc_4>',
                     '<loc_1><loc_2><loc_3><loc_4><loc_5>', '<loc_x>', '<loc_1'):
            with self.assertRaises(ValueError):
                self.convert('<text>' + body + '</text>')

    def test_preserve_every_noncoordinate_byte(self):
        raw = '<doctag>\n<unknown><loc_10><loc_20><loc_30><loc_40>日本語 &amp; \\x</unknown></doctag>'
        converted = self.convert(raw)
        strip = lambda s: re.sub(r'<loc_[0-9]+>', '', s)
        self.assertEqual(strip(raw), strip(converted))
        self.assertEqual(self.convert('<text>Hello</text>'), '<text>Hello</text>')

    def test_budget_and_missing_target(self):
        for raw in (None, '', 'x' * 2_000_001):
            with self.assertRaises(ValueError):
                self.convert(raw)

    def test_native_parser_still_rejects_semantic_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i, tag in enumerate(('text', 'unknown')):
                raw = f'<doctag><{tag}><loc_0><loc_0><loc_1000><loc_2000>x</{tag}></doctag>'
                path = Path(tmp) / f'{i}.doctags'
                path.write_text(self.convert(raw))
                files.append(path)
            results = native_probe(files)
            self.assertTrue(results[0]['markdown_renderable'])
            self.assertFalse(results[1]['markdown_renderable'])
            self.assertIn('unsupported-tag', results[1]['diagnostic_codes'])

    def test_independent_docling_reference(self):
        reference = json.loads((ROOT/'tests/fixtures/dataset-geometry/oracle.json').read_text())
        self.assertEqual(reference['docling_core'], '2.97.0')
        for case in reference['cases']:
            raw = '<text>' + ''.join(f'<loc_{v}>' for v in case['box']) + 'X</text>'
            self.assertEqual(self.convert(raw, *case['size']), '<text>' + case['expected'] + 'X</text>')


if __name__ == '__main__':
    unittest.main()
