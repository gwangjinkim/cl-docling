"""Authored supplemental picture score cases; no model or image dependency."""
import unittest
from picture_score import box_iou, score_picture


def reference():
    return dict(markdown='8\n\n![Picture](assets/picture-1.png)\n\nText.\n',
                signature=['page-header:0', 'picture:0', 'text:0'], picture_boxes=[[67, 77, 363, 178]])


def result():
    return dict(markdown='8\n\n![Picture](assets/picture-1.png)\n\nText.\n',
                signature=['page-header:0', 'picture:0', 'text:0'], picture_boxes=[[73, 77, 358, 174]],
                classification='other', stop_reason='eos', diagnostics=[], error=False, asset_pixels_verified=True)


class PictureScoreTests(unittest.TestCase):
    def test_iou_and_complete_acceptance(self):
        self.assertAlmostEqual(box_iou([67, 77, 363, 178], [73, 77, 358, 174]), 0.9247056462402997)
        score = score_picture(reference(), result(), 0.8)
        self.assertTrue(score['accepted'])
        self.assertEqual(score['classification'], 'other')

    def test_text_structure_asset_and_geometry_all_gate(self):
        mutations = [('markdown', 'wrong'), ('signature', ['picture:0']), ('picture_boxes', [[0, 0, 1, 1]]),
                     ('asset_pixels_verified', False), ('stop_reason', 'length'), ('diagnostics', ['bad']), ('error', True)]
        for key, value in mutations:
            actual = {**result(), key: value}
            with self.subTest(key=key): self.assertFalse(score_picture(reference(), actual, 0.8)['accepted'])
        self.assertFalse(score_picture(reference(), None, 0.8)['accepted'])

    def test_picture_count_and_invalid_boxes(self):
        actual = result(); actual['picture_boxes'] = []
        score = score_picture(reference(), actual, 0.8)
        self.assertFalse(score['picture_count_exact'])
        self.assertIsNone(score['picture_iou'])
        for box in [[], [0, 0, 1], [0, 0, 0, 0], [0, 0, 1.5, 2]]:
            with self.assertRaises(ValueError): box_iou([0, 0, 1, 1], box)

    def test_failures_score_as_empty_and_threshold_is_bounded(self):
        score = score_picture(reference(), None, 0.8)
        expected = ' '.join(reference()['markdown'].split())
        self.assertEqual(score['character_edits'], len(expected))
        self.assertEqual(score['word_edits'], len(expected.split()))
        self.assertEqual(score['markdown_cer'], 1.0)
        self.assertEqual(score['markdown_wer'], 1.0)
        for threshold in [-0.1, 1.1, True, '0.8']:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                score_picture(reference(), result(), threshold)


if __name__ == '__main__':
    unittest.main()
