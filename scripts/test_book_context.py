"""Offline contracts for complete, untruncated training-input accounting."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from book_context import summarize_case, complete_cases


def authored():
    return dict(ids=[1, 9, 9, 2, 4, 7], labels=[-100, -100, -100, -100, 4, 7],
                attention=[1] * 6, answer_start=4, tiles=1, pixel_shape=[1, 1, 3, 2, 2],
                image_token_id=9, eos_token_id=7, image_seq_len=2, context_limit=8)


class BookContextTests(unittest.TestCase):
    def test_exact_accounting_and_independent_inputs(self):
        native = authored()
        oracle = copy.deepcopy(native)
        result = summarize_case(native, oracle)
        self.assertEqual((result['sequence_tokens'], result['prompt_tokens'], result['supervised_tokens']), (6, 4, 2))
        self.assertEqual(result['context_remaining'], 2)
        self.assertEqual(result['pixel_payload_bytes'], 48)
        self.assertEqual(result['ids_labels_payload_bytes'], 48)
        self.assertEqual(result['attention_payload_bits'], 6)
        self.assertEqual(native, oracle)

    def test_same_counts_different_ids_not_parity(self):
        oracle = authored()
        oracle['ids'][4] = oracle['labels'][4] = 5
        with self.assertRaises(ValueError):
            summarize_case(authored(), oracle)

    def test_masks_eos_and_image_targets_rejected(self):
        for key, index, value in [('labels', 0, 1), ('labels', 4, -100), ('attention', 2, 0),
                                  ('ids', 5, 6), ('ids', 4, 9), ('ids', 4, 7)]:
            native = authored()
            native[key][index] = value
            if key == 'ids':
                native['labels'][index] = value
            with self.subTest(key=key, index=index, value=value), self.assertRaises(ValueError):
                summarize_case(native, copy.deepcopy(native))

    def test_bounds_tiles_and_shapes(self):
        for key, value in [('context_limit', 5), ('context_limit', 8193), ('answer_start', 0),
                           ('answer_start', 6), ('tiles', 2), ('pixel_shape', [1, 1, 4, 2, 2]),
                           ('ids', [True, 9, 9, 2, 4, 7]), ('labels', []), ('image_seq_len', 0)]:
            native = authored()
            native[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_case(native, copy.deepcopy(native))

    def test_no_missing_extra_or_duplicate_case(self):
        self.assertEqual(complete_cases(['a', 'b'], [{'name': 'a'}, {'name': 'b'}]),
                         {'a': {'name': 'a'}, 'b': {'name': 'b'}})
        for records in ([{'name': 'a'}], [{'name': 'a'}, {'name': 'a'}],
                        [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}]):
            with self.assertRaises(ValueError):
                complete_cases(['a', 'b'], records)

    def test_recorded_full_input_evidence(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / 'tests/fixtures/book-context/report.json').read_text())
        self.assertFalse(report['training_ready'])
        self.assertTrue(report['context_passed'])
        self.assertEqual(report['remaining_handles'], 0)
        for field, name in [('book_report_sha256', 'tests/fixtures/book-pilot/report.json'),
                            ('model_lock_sha256', 'references/smoldocling.lock.json')]:
            self.assertEqual(report[field], hashlib.sha256((root / name).read_bytes()).hexdigest())
        book = json.loads((root / 'tests/fixtures/book-pilot/report.json').read_text())
        expected = {'page-000': (1310, 878, 432, 13), 'page-002': (1296, 1142, 154, 17),
                    'page-004': (1289, 1142, 147, 17)}
        self.assertEqual(set(report['cases']), set(expected))
        for entry in book['inventory']:
            if entry['decision'] == 'approved':
                for key in ('image', 'target'):
                    self.assertEqual(report['input_sha256'][entry[key]], entry[key + '_sha256'])
        for name, case in report['cases'].items():
            self.assertEqual(tuple(case[k] for k in ('sequence_tokens', 'prompt_tokens', 'supervised_tokens', 'tiles')),
                             expected[name])
            self.assertTrue(case['native_python_exact'])
            self.assertEqual(case['context_remaining'] + case['sequence_tokens'], 8192)
            self.assertEqual(case['pixel_payload_bytes'], case['tiles'] * 3 * 512 * 512 * 4)
            self.assertEqual(case['image_tokens'], 64 * case['tiles'])
            for key in ('ids', 'labels', 'attention'):
                self.assertRegex(case[key + '_int32_le_sha256'], r'^[0-9a-f]{64}$')


if __name__ == '__main__':
    unittest.main()
