import copy
import hashlib
import json
from pathlib import Path
import unittest
from public_pdf_score import score_page, word_inventory


class PublicPdfScorerTests(unittest.TestCase):
    def setUp(self):
        self.expected = [['Label', 'Value'], ['Diameter', '3,476 km']]
        self.result = dict(tables=[dict(rows=2, columns=2, renderable=True,
            cells=[[0, 0, 1, 1, 'Label'], [0, 1, 1, 1, 'Value'],
                   [1, 0, 1, 1, 'Diameter'], [1, 1, 1, 1, '3,476   km']])],
            stop_reason='eos', diagnostics=[], non_table_elements=2, markdown_written=True,
            raw='<doctag><otsl>Label Value Diameter 3,476 km</otsl></doctag>')

    def test_whitespace_and_unscored_margins_do_not_fail_table(self):
        self.assertTrue(score_page(self.expected, self.result)['table_accepted'])

    def test_wrong_cell_placement_and_truncation_fail(self):
        bad = copy.deepcopy(self.result)
        bad['tables'][0]['cells'][3][1] = 0
        self.assertFalse(score_page(self.expected, bad)['table_accepted'])
        bad = dict(self.result, stop_reason='length')
        self.assertFalse(score_page(self.expected, bad)['table_accepted'])

    def test_invalid_structure_gets_no_cell_geometry_credit(self):
        bad = copy.deepcopy(self.result)
        bad['tables'][0]['renderable'] = False
        self.assertEqual(score_page(self.expected, bad)['exact_cell_matches'], 0)

    def test_superscripts_are_not_silently_flattened(self):
        bad = copy.deepcopy(self.result)
        bad['tables'][0]['cells'][3][4] = '10²⁴'
        self.assertFalse(score_page([['Label', 'Value'], ['Diameter', '1024']], bad)['table_accepted'])

    def test_inventory_is_unordered_and_duplicate_bounded(self):
        self.assertEqual(word_inventory('a b a', 'b a')['matched'], 2)
        self.assertEqual(word_inventory('a b', 'b a')['recall'], 1)

    def test_frozen_inputs_and_complete_recorded_results(self):
        root = Path(__file__).resolve().parents[1]
        fixtures = root / 'tests/fixtures/public-pdf'
        protocol = json.loads((fixtures / 'protocol.json').read_text())
        for name, wanted in protocol['files'].items():
            self.assertEqual(hashlib.sha256((fixtures / name).read_bytes()).hexdigest(), wanted)
        expected = json.loads((fixtures / 'expected.json').read_text())
        recorded = json.loads((fixtures / 'results/scores.json').read_text())
        self.assertEqual(set(expected), set(recorded))
        for name, rows in expected.items():
            native = json.loads((fixtures / f'results/native/{name}.json').read_text())
            python = json.loads((fixtures / f'results/python/{name}.json').read_text())
            for key, value in score_page(rows, native).items():
                self.assertEqual(recorded[name][key], json.loads(json.dumps(value)))
            self.assertEqual(recorded[name]['text_layer_word_inventory'], word_inventory(
                ' '.join(text for row in rows for text in row),
                (fixtures / (name + '-text-layer.txt')).read_text()))
            for key in ('tokens', 'raw', 'stop_reason'):
                self.assertEqual(native[key], python[key])
            self.assertTrue(recorded[name]['python_equal'])
            self.assertEqual(recorded[name]['output_tokens'], len(native['tokens']))
            self.assertEqual(native['raw'], (fixtures / f'results/native/{name}.doctags').read_text())
        self.assertEqual([recorded[n]['exact_cell_matches'] for n in expected], [34, 22])
        self.assertTrue(all(not r['table_accepted'] for r in recorded.values()))
        completion = json.loads((fixtures / 'results/completion.json').read_text())
        self.assertEqual(completion['remaining_handles'], 0)
        self.assertEqual(completion['device'], 'gpu')


if __name__ == '__main__':
    unittest.main()
