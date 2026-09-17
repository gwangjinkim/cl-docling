import copy
import hashlib
import json
from pathlib import Path
import unittest
from coverage_data import cases, ROOT
from coverage_score import distance, score, recorded_report


class CoverageTests(unittest.TestCase):
    def test_edit_metrics(self):
        self.assertEqual(distance('kitten','sitting'),3)
        self.assertEqual(distance([],['extra']),1)
        self.assertEqual(distance(['a','b'],['b','a']),2)
        case=dict(expected='Alpha beta')
        result=dict(raw='<doctag><text>Alpha beta</text></doctag>',stop_reason='eos',diagnostics=[],markdown_written=True)
        self.assertTrue(score(case,result)['accepted'])
        result['raw']='<doctag><text>beta Alpha</text></doctag>'
        s=score(case,result); self.assertFalse(s['text_exact']); self.assertEqual(s['inventory_matches'],2)
        result['raw']='<doctag><text>Alpha beta</text></doctag>'; result['stop_reason']='length'
        self.assertFalse(score(case,result)['accepted'])
        result['stop_reason']='eos'; result['diagnostics']=[dict(code='unsupported')]
        self.assertFalse(score(case,result)['accepted'])

    def test_frozen_sources(self):
        folder=ROOT/'tests/fixtures/coverage'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,sha in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha)
        data=json.loads((folder/'expected.json').read_text())
        self.assertEqual(len(data),9); self.assertEqual(len({c['source'] for c in data}),9)
        self.assertEqual(json.loads(json.dumps(cases())),data)
        final=json.loads((folder/'final.json').read_text())
        self.assertTrue(all(set(c)=={'name','split'} for c in final['cases']))

    def test_recorded_results(self):
        folder=ROOT/'tests/fixtures/coverage'
        report=recorded_report(folder)
        self.assertEqual(json.loads(json.dumps(report)),json.loads((folder/'results/scores.json').read_text()))
        self.assertTrue(all(p['python_equal'] for pages in report.values() for p in pages.values()))
        self.assertTrue(all(p['stop_reason']=='eos' for pages in report.values() for p in pages.values()))


if __name__=='__main__': unittest.main()
