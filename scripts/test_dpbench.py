import copy
import hashlib
import json
import unittest
from dpbench_score import score, validate_manifest
from dataset_audit import ROOT


class DPBenchTests(unittest.TestCase):
    def test_retained_baseline_and_source_boundaries(self):
        folder=ROOT/'tests/fixtures/dpbench'
        manifest=json.loads((folder/'manifest.json').read_text())
        validate_manifest(manifest)
        report=json.loads((folder/'report.json').read_text())
        run=json.loads((folder/'run-summary.json').read_text())
        lock=ROOT/'references/dpbench.lock.json'
        self.assertEqual(manifest['protocol_sha256'],hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(manifest['protocol'],json.loads(lock.read_text()))
        self.assertEqual(report['protocol'],manifest['protocol'])
        self.assertEqual(report['manifest_sha256'],hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
        self.assertEqual(report['run_sha256'],run['original_run_sha256'])
        self.assertEqual(run['source_revision'],'8b18f762a4031e7ba96369377a218f3041aea34e')
        self.assertEqual(run['exit_code'],0)
        self.assertEqual(report['completion']['remaining_handles'],0)
        pages=report['cases']
        self.assertEqual(set(pages),{c['name'] for c in manifest['cases']})
        self.assertEqual(report['summary']['pages'],4)
        self.assertEqual(report['summary']['strict_exports'],0)
        self.assertEqual(report['summary']['reference_tables'],0)
        for page in pages.values():
            self.assertEqual(page['stop_reason'],'eos')
            self.assertTrue(page['diagnostic_codes'])
            self.assertFalse(page['accepted'])
            self.assertFalse(page['execution_error'])
            self.assertEqual(page['markdown_cer'],1)
        for key,edits,total in [('raw_text_cer','raw_text_character_edits','raw_text_reference_characters'),
                                ('raw_text_wer','raw_text_word_edits','raw_text_reference_words')]:
            self.assertEqual(report['summary'][key],sum(p[edits] for p in pages.values())/sum(p[total] for p in pages.values()))
        review=json.loads((ROOT/'references/dataset-source-review.json').read_text())
        self.assertEqual(review['training_source_allowlist'],[])
        self.assertFalse(review['external_benchmark']['training_approved'])
        audit=json.loads((ROOT/'tests/fixtures/dataset-audit/report.json').read_text())
        quarantined=[c for c in audit['cases'] if c['source_id'] in review['quarantined_source_ids']]
        self.assertEqual([c['shard_row'] for c in quarantined],[41,126])

    def expected(self):
        return dict(markdown='Hello world.\n', tables=[])

    def result(self):
        return dict(markdown_written=True, stop_reason='eos', diagnostics=[], tables=[])

    def test_exact_and_whitespace(self):
        got=score(self.expected(), self.result(), 'Hello   world.\n\n')
        self.assertTrue(got['accepted'])
        self.assertEqual((got['markdown_cer'], got['markdown_wer']), (0, 0))

    def test_blocked_output_is_not_a_success(self):
        for changes in (dict(markdown_written=False), dict(stop_reason='length'),
                        dict(diagnostics=[dict(code='unsupported-tag')]), dict(error='runtime')):
            got=score(self.expected(), {**self.result(), **changes}, 'Hello world.')
            self.assertFalse(got['accepted'])
            self.assertEqual(got['markdown_cer'], 1)
            self.assertFalse(got['strict_export'])

    def test_metrics_are_not_capped(self):
        got=score(dict(markdown='x',tables=[]), self.result(), 'xxxx')
        self.assertEqual(got['markdown_cer'], 3)

    def test_raw_recognition_does_not_hide_export_failure(self):
        result={**self.result(),'diagnostics':[dict(code='unsupported-tag')],
                'raw':'<page_header>Hello world.</page_header>'}
        got=score(self.expected(),result,None)
        self.assertEqual(got['raw_text_character_edits'],0)
        self.assertEqual(got['markdown_cer'],1)
        self.assertFalse(got['accepted'])

    def test_table_positions_spans_and_extras(self):
        table=dict(rows=1,columns=2,cells=[[0,0,1,1,'A'],[0,1,1,1,'B']])
        expected=dict(markdown='table',tables=[table])
        native={**self.result(), 'tables':[copy.deepcopy(table)]}
        self.assertTrue(score(expected,native,'table')['accepted'])
        native['tables'][0]['cells'][0][-1]='B'
        native['tables'][0]['cells'][1][-1]='A'
        got=score(expected,native,'table')
        self.assertFalse(got['accepted'])
        self.assertEqual(got['exact_cell_matches'],0)
        native['tables']=[table,table]
        self.assertFalse(score(expected,native,'table')['tables_exact'])

    def test_missing_and_empty_reference(self):
        with self.assertRaises(ValueError):
            score(dict(markdown=' ',tables=[]),self.result(),'')
        got=score(self.expected(), None, None)
        self.assertEqual(got['markdown_wer'],1)
        self.assertFalse(got['accepted'])

    def manifest(self):
        return dict(training_approved=False, cases=[
            dict(name=f'page-{i:03d}', split='regression',row=i, document_id=f'doc{i}',
                 image_sha256=str(i)*64,ground_truth_sha256=str(i+4)*64)
            for i in range(4)])

    def test_fixed_rows_and_no_training(self):
        validate_manifest(self.manifest())
        for mutate in (lambda m:m.update(training_approved=True),
                       lambda m:m['cases'][0].update(split='train'),
                       lambda m:m['cases'][0].update(row=20),
                       lambda m:m['cases'][0].update(name='../bad'),
                       lambda m:m['cases'].pop()):
            m=self.manifest(); mutate(m)
            with self.assertRaises(ValueError): validate_manifest(m)

    def test_duplicates_rejected(self):
        for key in ('document_id','image_sha256','ground_truth_sha256'):
            m=self.manifest(); m['cases'][1][key]=m['cases'][0][key]
            with self.assertRaises(ValueError): validate_manifest(m)


if __name__=='__main__': unittest.main()
