import copy
import hashlib
import json
import math
from datetime import datetime
import unittest
from selection_data import ROOT, cases, validate, choose_candidate, verify_manifest, STEPS
from table_score import score


class SelectionTests(unittest.TestCase):
    def test_corpus(self):
        data = cases()
        validate(data)
        self.assertEqual([sum(c['split'] == s for c in data)
                          for s in ('train', 'validation', 'final', 'regression')], [4, 3, 4, 2])

    def test_leakage(self):
        for field in ('name', 'source', 'expected'):
            data = cases()
            data[-1][field] = copy.deepcopy(data[0][field])
            with self.assertRaises(ValueError):
                validate(data)

    def scores(self):
        return {step: {name: dict(accepted=False, exact_cell_matches=0)
                       for name in ('sports', 'music', 'bakery')} for step in STEPS}

    def test_base_fallback_and_earliest_tie(self):
        scores = self.scores()
        self.assertEqual(choose_candidate(scores), 0)
        for step in (16, 48, 96):
            scores[step]['sports']['accepted'] = True
        self.assertEqual(choose_candidate(scores), 16)

    def test_acceptance_before_cells(self):
        scores = self.scores()
        scores[16]['sports']['exact_cell_matches'] = 9
        scores[48]['music']['accepted'] = True
        self.assertEqual(choose_candidate(scores), 48)
        scores[96]['music']['accepted'] = True
        scores[96]['sports']['exact_cell_matches'] = 8
        self.assertEqual(choose_candidate(scores), 96)

    def test_reject_final_scores_and_missing_candidates(self):
        scores = self.scores()
        scores[16]['pets'] = dict(accepted=True, exact_cell_matches=10)
        with self.assertRaises(ValueError):
            choose_candidate(scores)
        scores = self.scores()
        scores.pop(48)
        with self.assertRaises(ValueError):
            choose_candidate(scores)

    def test_manifest_isolation_and_checksums(self):
        folder = ROOT / 'tests/fixtures/selection-pages'
        manifest = json.loads((folder / 'manifest.json').read_text())
        verify_manifest(manifest)
        for name, wanted in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), wanted)
        dev = json.loads((folder / 'development.json').read_text())['cases']
        final = json.loads((folder / 'final.json').read_text())['cases']
        self.assertFalse({c['source'] for c in dev} & {c['source'] for c in final})
        self.assertEqual({c['split'] for c in dev}, {'train', 'validation'})
        self.assertEqual({c['split'] for c in final}, {'final', 'regression'})
        for case in dev + final:
            self.assertEqual('answer' in case, case['split'] == 'train')
        changed = copy.deepcopy(manifest)
        changed['protocol']['epochs'] += 1
        with self.assertRaises(ValueError):
            verify_manifest(changed)

    def test_recorded_selection_and_final_results(self):
        fixtures = ROOT / 'tests/fixtures/selection-pages'
        results = ROOT / 'tests/fixtures/selection-results'
        selection = json.loads((results / 'selection.json').read_text())
        report = json.loads((results / 'report.json').read_text())
        candidates = {int(k): v for k, v in selection['validation'].items()}
        self.assertEqual(choose_candidate(candidates), selection['selected_step'])
        self.assertEqual(selection['selected_step'], report['selected_step'])
        self.assertEqual(hashlib.sha256((results / 'selection.json').read_bytes()).hexdigest(), report['selection_sha256'])
        self.assertEqual(hashlib.sha256((fixtures / 'manifest.json').read_bytes()).hexdigest(), selection['manifest_sha256'])
        for step, pages in candidates.items():
            for name, scores in pages.items():
                self.check_output(fixtures, results / 'development' / f'step-{step}', name, scores)
        for name, case in report['cases'].items():
            for phase in ('base', 'adapted'):
                self.check_output(fixtures, results / 'final' / phase, name, case[phase])
            if case['split'] == 'final':
                native = json.loads((results / 'final/adapted' / f'{name}.json').read_text())
                reference = json.loads((results / 'python' / f'{name}.json').read_text())
                for key in ('raw', 'tokens', 'stop_reason'):
                    self.assertEqual(native[key], reference[key])
        for split, summary in report['summary'].items():
            subset = [c for c in report['cases'].values() if c['split'] == split]
            for phase in ('base', 'adapted'):
                self.assertEqual(summary[phase], dict(
                    accepted=sum(c[phase]['accepted'] for c in subset), pages=len(subset),
                    exact_cells=sum(c[phase]['exact_cell_matches'] for c in subset),
                    expected_cells=sum(c[phase]['expected_cells'] for c in subset)))
        timeline = json.loads((results / 'timeline.json').read_text())
        self.assertEqual([e['event'] for e in timeline], ['development_started', 'development_finished',
                                                       'selection_persisted', 'final_started', 'final_finished'])
        times = [datetime.fromisoformat(e['time']) for e in timeline]
        self.assertEqual(times, sorted(times))
        losses = json.loads((results / 'development/losses.json').read_text())
        self.assertEqual([e['page'] for e in losses], ['garden', 'kitchen', 'workshop', 'art'] * 24)
        self.assertEqual([e['step'] for e in losses], list(range(1, 97)))
        self.assertTrue(all(math.isfinite(e['loss']) and e['tiles'] == 13 for e in losses))

    def test_shared_native_generation_regression(self):
        fixtures = ROOT / 'tests/fixtures'
        for name in ('sports', 'music', 'bakery'):
            previous = json.loads((fixtures / 'adaptation-results/base' / f'{name}.json').read_text())
            current = json.loads((fixtures / 'selection-results/development/step-0' / f'{name}.json').read_text())
            for key in ('tokens', 'raw', 'stop_reason', 'tables', 'diagnostics', 'markdown_written', 'non_table_elements'):
                self.assertEqual(previous[key], current[key])

    def check_output(self, fixtures, folder, name, expected_score):
        expected = json.loads((fixtures / f'{name}.json').read_text())
        result = json.loads((folder / f'{name}.json').read_text())
        self.assertEqual(json.loads(json.dumps(score(expected, result))), expected_score)
        self.assertEqual(result['raw'], (folder / f'{name}.doctags').read_text())
        self.assertEqual(result['markdown_written'], (folder / f'{name}.md').exists())


if __name__ == '__main__':
    unittest.main()
