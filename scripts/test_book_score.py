"""Authored failure-aware book scoring cases; no model required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from book_baseline import ROOT, POLICY, project, validate_manifest
from book_score import score_case, summarize


def reference():
    return dict(markdown='6\n\nHello world.', signature=['page-header:0', 'text:0'])


def result():
    return dict(markdown='6\n\nHello world.', signature=['page-header:0', 'text:0'],
                stop_reason='eos', diagnostics=[], error=False, tokens=4)


class BookScoreTests(unittest.TestCase):
    def test_retained_real_baseline_has_every_page_and_recomputes(self):
        from replay_dpbench import digest
        fixture = ROOT / 'tests/fixtures/book-baseline'
        manifest = json.loads((fixture / 'manifest.json').read_text())
        report = json.loads((fixture / 'report.json').read_text())
        validate_manifest(manifest)
        self.assertEqual(report['manifest_sha256'], digest(fixture / 'manifest.json'))
        self.assertEqual(report['summary'], summarize(manifest['cases'], report['cases']))
        self.assertEqual(report['completion'], dict(remaining_handles=0, device='gpu', updates=0))
        self.assertEqual(report['exit_code'], 0)
        self.assertFalse(report['training_ready'])
        self.assertEqual(sum(c['generated_tokens'] for c in report['cases'].values()), 2535)
        for name, case in report['cases'].items():
            self.assertEqual(case['stop_reason'], 'eos')
            for extension in ('doctags', 'json'):
                self.assertIn(f'native/base/{name}.{extension}', report['retained_sha256'])
        unknown = report['cases']['validation-005']
        self.assertIsNone(unknown['accepted'])
        self.assertIsNone(unknown['markdown_cer'])
        self.assertFalse(unknown['strict_export'])
        self.assertEqual(report['summary']['train']['reviewed_character_edits'], 12)
        self.assertEqual(report['summary']['validation']['reviewed_character_edits'], 11)
        for split in report['summary'].values():
            self.assertEqual(split['verified_conversions'], 0)
            self.assertFalse(split['full_quality_gate'])
        for name, expected in report['scorer_sha256'].items():
            self.assertEqual(digest(ROOT / name), expected)

    def test_native_projection_preserves_stop_reason_and_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'page.doctags'
            raw.write_text('<doctag><page_header>6</page_header><text>Hello world.</text></doctag>')
            good, truncated = project([raw, raw], root / 'projected', ['eos', 'length'])
            self.assertEqual(good['signature'], ['page-header:0', 'text:0'])
            self.assertEqual(good['diagnostics'], [])
            self.assertIsNotNone(good['markdown'])
            self.assertIsNone(truncated['markdown'])
            self.assertIn('generation-truncated', truncated['diagnostics'])

    def test_frozen_manifest_rejects_final_and_changed_coverage(self):
        policy = json.loads(POLICY.read_text())
        from replay_dpbench import digest
        cases = [dict(name=n, split=s, reference_available=n not in policy['unscorable'],
                      expected=n + '.expected.json' if n not in policy['unscorable'] else None)
                 for n, s in zip(policy['names'], policy['splits'], strict=True)]
        manifest = dict(protocol=policy, protocol_sha256=digest(POLICY), cases=cases)
        validate_manifest(manifest)
        for field, value in [('split', 'final'), ('reference_available', False), ('expected', None)]:
            changed = copy.deepcopy(manifest)
            changed['cases'][0][field] = value
            with self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_exact_whitespace_only_normalization(self):
        actual = result()
        actual['markdown'] = ' 6\n\nHello   world.\n'
        score = score_case(reference(), actual)
        self.assertTrue(score['accepted'])
        self.assertEqual(score['character_edits'], 0)
        actual['markdown'] = '6 hello world.'
        self.assertFalse(score_case(reference(), actual)['accepted'])

    def test_matching_text_cannot_hide_structure_loss(self):
        actual = result()
        actual['signature'] = ['text:0', 'text:0']
        score = score_case(reference(), actual)
        self.assertEqual(score['character_edits'], 0)
        self.assertFalse(score['structure_exact'])
        self.assertFalse(score['accepted'])

    def test_all_failure_paths_score_as_empty_on_reviewed_pages(self):
        for change in [dict(stop_reason='length'), dict(diagnostics=['unsupported-tag']),
                       dict(error='execution'), dict(markdown=None)]:
            score = score_case(reference(), {**result(), **change})
            self.assertFalse(score['strict_export'])
            self.assertFalse(score['accepted'])
            self.assertEqual(score['markdown_cer'], 1)
            self.assertEqual(score['markdown_wer'], 1)
        self.assertTrue(score_case(reference(), None)['execution_error'])

    def test_unsupported_reference_is_not_a_zero_error_or_model_failure(self):
        score = score_case(None, result())
        self.assertTrue(score['strict_export'])
        self.assertIsNone(score['accepted'])
        self.assertIsNone(score['markdown_cer'])
        self.assertIsNone(score['character_edits'])
        self.assertFalse(score['execution_error'])

    def test_no_empty_reference_and_errors_not_capped(self):
        with self.assertRaises(ValueError):
            score_case(dict(markdown=' ', signature=[]), result())
        score = score_case(dict(markdown='x', signature=['text:0']),
                           {**result(), 'markdown': 'xxxx', 'signature': ['text:0']})
        self.assertEqual(score['markdown_cer'], 3)

    def test_complete_denominators_and_micro_aggregation(self):
        cases = [{'name': n, 'split': 'validation'} for n in ['a', 'b', 'c']]
        reports = {'a': score_case(reference(), result()), 'b': score_case(reference(), None),
                   'c': score_case(None, result())}
        summary = summarize(cases, reports)['validation']
        self.assertEqual((summary['pages'], summary['reviewed_pages'], summary['unscorable_pages']), (3, 2, 1))
        self.assertEqual(summary['strict_exports'], 2)
        self.assertEqual(summary['verified_conversions'], 1)
        self.assertAlmostEqual(summary['verified_conversion_yield'], 1/3)
        self.assertEqual(summary['reviewed_exact_rate'], 0.5)
        self.assertEqual(summary['reviewed_markdown_cer'], 0.5)
        self.assertFalse(summary['full_quality_gate'])
        self.assertFalse(summary['reference_coverage_complete'])

    def test_cannot_drop_add_or_repeat_pages(self):
        cases = [{'name': 'a', 'split': 'validation'}]
        report = {'a': score_case(reference(), result())}
        for altered, reports in [(cases, {}), (cases, {**report, 'extra': report['a']}), (cases*2, report)]:
            with self.assertRaises(ValueError):
                summarize(altered, reports)


if __name__ == '__main__':
    unittest.main()
