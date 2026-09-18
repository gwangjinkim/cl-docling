"""Authored corpus-policy tests. Synthetic approvals never approve real sources."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import corpus_preflight
from corpus_preflight import ROOT, checked_bytes, content_fingerprint, evaluate_candidates


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source(number):
    return 'doc_' + f'{number:040x}'


def case(number, split='train', group=None):
    return dict(dataset='authored-test', revision='a' * 40,
                id=source(group or number) + f'_p{number:05d}', split=split,
                image_sha256=sha('image' + str(number)), target_sha256=sha('target' + str(number)),
                content_sha256=sha('content' + str(number)), markdown_renderable=True,
                diagnostic_codes=[])


def approved(cases):
    sources = sorted({c['id'].split('_p')[0] for c in cases})
    return dict(schema_version=1, dataset='authored-test', revision='a' * 40,
                training_source_allowlist=sources, quarantined_source_ids=[],
                source_reviews={s: dict(original_source_url='https://example.org/authored',
                                       license_url='https://example.org/authored-license',
                                       reviewer='authored test only', decision_note='Synthetic fixture, not a real approval',
                                       training_approved=True, privacy_cleared=True) for s in sources},
                approved_targets={c['id']: c['target_sha256'] for c in cases})


class CorpusPreflightTests(unittest.TestCase):
    def test_approved_separated_candidates(self):
        cases = [case(1), case(2, 'validation'), case(3, 'final')]
        report = evaluate_candidates(cases, approved(cases), set())
        self.assertTrue(report['summary']['split_preflight_passed'])
        self.assertFalse(report['training_approved'])  # Not a training/resource gate.
        self.assertEqual(report['summary']['eligible'], 3)

    def test_pending_and_quarantine_override_allowlist(self):
        cases = [case(1), case(2)]
        ledger = approved(cases)
        ledger['training_source_allowlist'] = [source(2)]
        ledger['quarantined_source_ids'] = [source(2)]
        report = evaluate_candidates(cases, ledger, set())
        self.assertIn('source-not-approved', report['cases'][0]['reasons'])
        self.assertIn('quarantined-source', report['cases'][1]['reasons'])
        self.assertEqual(report['summary']['eligible'], 0)

    def test_allowlist_is_not_evidence_and_label_hash_is_required(self):
        cases = [case(1)]
        ledger = approved(cases)
        ledger['source_reviews'] = {}
        ledger['approved_targets'][cases[0]['id']] = sha('a different label')
        reasons = evaluate_candidates(cases, ledger, set())['cases'][0]['reasons']
        self.assertIn('source-review-incomplete', reasons)
        self.assertIn('target-not-reviewed', reasons)

    def test_source_review_requires_each_field(self):
        cases = [case(1)]
        for key in approved(cases)['source_reviews'][source(1)]:
            ledger = approved(cases)
            del ledger['source_reviews'][source(1)][key]
            with self.subTest(key=key):
                self.assertIn('source-review-incomplete', evaluate_candidates(cases, ledger, set())['cases'][0]['reasons'])

    def test_audited_source_cannot_be_final_even_on_another_page(self):
        cases = [case(7, 'final', group=1)]
        self.assertIn('development-source-in-final',
                      evaluate_candidates(cases, approved(cases), {source(1)})['cases'][0]['reasons'])

    def test_source_cannot_cross_splits(self):
        cases = [case(1), case(2, 'validation', group=1)]
        report = evaluate_candidates(cases, approved(cases), set())
        self.assertTrue(all('cross-split-component' in c['reasons'] for c in report['cases']))

    def test_quarantine_propagates_through_duplicate_source_alias(self):
        cases = [case(1), case(2), case(3, group=2)]
        cases[1]['image_sha256'] = cases[0]['image_sha256']
        ledger = approved(cases)
        ledger['quarantined_source_ids'] = [source(1)]
        report = evaluate_candidates(cases, ledger, set())
        self.assertIn('quarantined-component', report['cases'][2]['reasons'])
        self.assertEqual(report['summary']['eligible'], 0)

    def test_transitive_duplicate_component_and_development_taint(self):
        cases = [case(1), case(2), case(3, 'final')]
        cases[1]['image_sha256'] = cases[0]['image_sha256']
        cases[2]['content_sha256'] = cases[1]['content_sha256']
        report = evaluate_candidates(cases, approved(cases), {source(1)})
        self.assertEqual(len(report['components']), 1)
        self.assertTrue(all('cross-split-component' in c['reasons'] for c in report['cases']))
        self.assertIn('development-source-in-final', report['cases'][2]['reasons'])
        self.assertEqual(report['components'][0]['members'], [c['id'] for c in cases])

    def test_same_split_duplicates_block_without_dropping_cases(self):
        for key in ('image_sha256', 'target_sha256', 'content_sha256', 'id'):
            cases = [case(1), case(2)]
            cases[1][key] = cases[0][key]
            report = evaluate_candidates(cases, approved(cases), set())
            with self.subTest(key=key):
                self.assertEqual(report['summary']['pages'], 2)
                self.assertEqual(report['summary']['eligible'], 0)
                self.assertTrue(all('duplicate-page' in c['reasons'] for c in report['cases']))

    def test_semantic_failure_cannot_be_approved_away(self):
        cases = [case(1)]
        cases[0].update(markdown_renderable=False, diagnostic_codes=['unsupported-tag'])
        report = evaluate_candidates(cases, approved(cases), set())
        self.assertIn('native-target-rejected', report['cases'][0]['reasons'])

    def test_dataset_revision_and_external_benchmark_are_not_interchangeable(self):
        for key, value in (('dataset', 'docling-project/docling-dpbench'), ('revision', 'b' * 40)):
            cases = [case(1)]
            ledger = approved(cases)
            cases[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate_candidates(cases, ledger, set())

    def test_schema_and_bounds_fail_closed(self):
        for changes in ({'id': '../unsafe'}, {'split': 'test'}, {'image_sha256': 'bad'},
                        {'markdown_renderable': 1}, {'diagnostic_codes': ['x']},
                        {'markdown_renderable': False}):
            cases = [case(1)]
            ledger = approved(cases)
            cases[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate_candidates(cases, ledger, set())
        for cases in ([], [case(i + 1) for i in range(101)]):
            with self.assertRaises(ValueError):
                evaluate_candidates(cases, approved(cases), set())

    def test_fingerprint_ignores_only_locations_and_whitespace(self):
        a = '<doctag><text><loc_0><loc_1><loc_2><loc_3>A  B</text></doctag>'
        b = '<doctag><text><loc_4><loc_5><loc_6><loc_7>A\nB</text></doctag>'
        self.assertEqual(content_fingerprint(a), content_fingerprint(b))
        self.assertNotEqual(content_fingerprint(a), content_fingerprint(b.replace('A', 'a')))
        self.assertNotEqual(content_fingerprint(a), content_fingerprint(b.replace('text', 'title')))
        for bad in ('', 'x' * 2_000_001, None):
            with self.assertRaises(ValueError):
                content_fingerprint(bad)

    def test_inputs_unchanged_and_order_deterministic(self):
        cases = [case(2), case(1)]
        ledger = approved(cases)
        before = copy.deepcopy((cases, ledger))
        first = evaluate_candidates(cases, ledger, set())
        self.assertEqual(first, evaluate_candidates(cases, ledger, set()))
        self.assertEqual((cases, ledger), before)
        self.assertEqual([c['id'] for c in first['cases']], [c['id'] for c in cases])

    def test_local_evidence_identity_and_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'raw'
            path.write_bytes(b'original')
            self.assertEqual(checked_bytes(root, 'raw', sha('original'), 8), b'original')
            for name, wanted, limit in (('raw', sha('changed'), 8), ('raw', sha('original'), 7)):
                with self.assertRaises(ValueError):
                    checked_bytes(root, name, wanted, limit)
            child = root / 'evidence'
            child.mkdir()
            (child / 'link').symlink_to(path)
            for name in ('../raw', 'link'):
                with self.assertRaises(ValueError):
                    checked_bytes(child, name, sha('original'), 8)

    def test_regenerated_geometry_uses_content_hashes_not_old_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = '<doctag><text>Authored test</text></doctag>'
            for name, data in (('000.png', b'authored image bytes'), ('000.original.doctags', raw.encode()),
                               ('000.normalized.doctags', raw.encode())):
                (root / name).write_bytes(data)
            # A later parser's diagnostic report must not invalidate identical input bytes.
            (root / 'report.json').write_text('{"new_parser_diagnostics": []}')
            page = dict(ordinal=0, id=case(1)['id'], image_sha256=sha('authored image bytes'),
                        original_target_sha256=sha(raw), normalized_target_sha256=sha(raw))
            geometry = dict(dataset='authored-test', revision='a' * 40, cases=[page])
            audit = dict(cases=[dict(id=case(1)['id'], target_sha256=sha(raw))])
            records, paths, consumed = corpus_preflight.load_candidates(root, geometry, audit)
            self.assertEqual(len(records), 1)
            self.assertEqual(paths, [root / '000.normalized.doctags'])
            self.assertEqual(len(consumed), 3)
            self.assertNotIn('report.json', consumed)
            (root / '000.normalized.doctags').write_text('tampered')
            with self.assertRaises(ValueError):
                corpus_preflight.load_candidates(root, geometry, audit)

    def test_recorded_real_sample_is_not_training_ready(self):
        report = json.loads((ROOT / 'tests/fixtures/corpus-preflight/report.json').read_text())
        audit_path = ROOT / 'tests/fixtures/dataset-audit/report.json'
        geometry_path = ROOT / 'tests/fixtures/dataset-geometry/report.json'
        ledger_path = ROOT / 'references/dataset-source-review.json'
        audit = json.loads(audit_path.read_text())
        ledger = json.loads(ledger_path.read_text())
        for key, path in (('audit_report', audit_path), ('geometry_report', geometry_path), ('review_ledger', ledger_path)):
            self.assertEqual(report['evidence_sha256'][key], hashlib.sha256(path.read_bytes()).hexdigest())
        replay = evaluate_candidates(report['cases'], ledger, {c['source_id'] for c in audit['cases']})
        for key in ('summary', 'cases', 'components', 'training_approved'):
            self.assertEqual(replay[key], report[key])
        self.assertEqual(report['summary']['pages'], 20)
        self.assertEqual(report['summary']['sources'], 10)
        self.assertEqual(report['summary']['eligible'], 0)
        self.assertEqual(report['summary']['native_renderable'], 13)
        self.assertEqual(report['summary']['blocked_reasons']['quarantined-source'], 2)
        self.assertEqual([c['id'] for c in report['cases']], [c['id'] for c in audit['cases']])
        self.assertFalse(report['training_approved'])
        for record in report['cases']:
            self.assertFalse(record['eligible'])
            self.assertNotIn('raw', record)
            self.assertNotIn('doctags', record)
        old_geometry = json.loads(geometry_path.read_text())
        self.assertEqual(old_geometry['summary']['native_renderable'], 9)  # Never rewrite historical results.


if __name__ == '__main__':
    unittest.main()
