import copy
import json
import hashlib
import math
import unittest
from adaptation_data import ROOT, cases, validate, answer, validate_manifest
from table_score import score


class AdaptationTests(unittest.TestCase):
    def test_split(self):
        data = cases()
        validate(data)
        self.assertEqual([sum(c['split'] == split for c in data)
                          for split in ('train', 'heldout', 'regression')], [4, 3, 2])

    def test_leakage_and_duplicates(self):
        for field in ('name', 'source'):
            data = cases()
            data[-1][field] = data[0][field]
            with self.assertRaises(ValueError):
                validate(data)
        data = cases()
        data[-1]['expected'] = copy.deepcopy(data[0]['expected'])
        with self.assertRaises(ValueError):
            validate(data)

    def test_bad_geometry(self):
        for cell in ([0, 0, 1, 1, 'Overlap'], [99, 0, 1, 1, 'Outside']):
            data = cases()
            data[0]['expected']['cells'].append(cell)
            with self.assertRaises(ValueError):
                validate(data)

    def test_targets(self):
        target = answer(cases()[0]['expected'])
        self.assertIn('<ched>Garden stock<lcel><ched>Count<nl>', target)
        self.assertIn('<ucel><fcel>Rakes<fcel>2<nl>', target)
        self.assertTrue(target.endswith('</otsl></doctag>'))

    def test_frozen_manifest(self):
        manifest = json.loads((ROOT / 'tests/fixtures/adaptation-pages/manifest.json').read_text())
        validate_manifest(manifest)
        for change in ('split', 'answer', 'protocol', 'files'):
            other = copy.deepcopy(manifest)
            if change == 'split':
                other['cases'][4]['split'] = 'train'
            elif change == 'answer':
                other['cases'][4]['answer'] = 'Evaluation label must not train'
            elif change == 'protocol':
                other['protocol']['epochs'] += 1
            else:
                other['files'].pop('garden.png')
            with self.assertRaises(ValueError):
                validate_manifest(other)

    def test_recorded_experiment(self):
        fixtures = ROOT / 'tests/fixtures/adaptation-pages'
        results = ROOT / 'tests/fixtures/adaptation-results'
        manifest = json.loads((fixtures / 'manifest.json').read_text())
        report = json.loads((results / 'report.json').read_text())
        self.assertEqual(hashlib.sha256((fixtures / 'manifest.json').read_bytes()).hexdigest(), report['manifest_sha256'])
        for name, digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((fixtures / name).read_bytes()).hexdigest(), digest)
        for case in manifest['cases']:
            name = case['name']
            expected = json.loads((fixtures / f'{name}.json').read_text())
            for phase in ('base', 'adapted'):
                result = json.loads((results / phase / f'{name}.json').read_text())
                self.assertEqual(result['raw'], (results / phase / f'{name}.doctags').read_text())
                self.assertEqual(result['markdown_written'], (results / phase / f'{name}.md').exists())
                self.assertEqual(json.loads(json.dumps(score(expected, result))), report['cases'][name][phase])
            if case['split'] == 'heldout':
                native = json.loads((results / 'adapted' / f'{name}.json').read_text())
                reference = json.loads((results / 'python' / f'{name}.json').read_text())
                for key in ('tokens', 'raw', 'stop_reason'):
                    self.assertEqual(native[key], reference[key])
        losses = json.loads((results / 'losses.json').read_text())
        training = [c['name'] for c in manifest['cases'] if c['split'] == 'train']
        self.assertEqual([entry['page'] for entry in losses], training * 4)
        self.assertTrue(all(math.isfinite(entry['loss']) and entry['tiles'] == 13 for entry in losses))
        self.assertEqual(report['summary']['heldout']['base'], report['summary']['heldout']['adapted'])
        self.assertEqual(report['summary']['heldout']['adapted']['accepted'], 1)


if __name__ == '__main__':
    unittest.main()
