"""Frozen lexical metrics; no semantic math equivalence or full layout score."""
from collections import Counter
import re
from public_pdf_score import score_page


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]


def score(case,result):
    expected=' '.join(case['expected'].split())
    actual=' '.join(re.sub(r'<[^>]*>',' ',result['raw']).split())
    words=expected.split(); found=actual.split()
    report=dict(expected_text=expected,recognized_text=actual,
                character_edits=distance(expected,actual),reference_characters=len(expected),
                word_edits=distance(words,found),reference_words=len(words),
                inventory_matches=sum((Counter(words)&Counter(found)).values()),
                text_exact=expected==actual,stop_reason=result['stop_reason'],
                diagnostics=result['diagnostics'],markdown_written=result['markdown_written'])
    report['cer']=report['character_edits']/len(expected)
    report['wer']=report['word_edits']/len(words)
    report['accepted']=report['text_exact'] and result['stop_reason']=='eos' and not result['diagnostics'] and result['markdown_written']
    if 'rows' in case:
        report['table']=score_page(case['rows'],result)
        report['accepted'] &= report['table']['table_accepted']
    return report


def recorded_report(folder):
    import json
    cases=json.loads((folder/'expected.json').read_text())
    reports={}
    for phase in ('base','adapted'):
        reports[phase]={}
        for case in cases:
            name=case['name']
            native=json.loads((folder/'results/native'/phase/(name+'.json')).read_text())
            python=json.loads((folder/'results/python'/phase/(name+'.json')).read_text())
            result=score(case,native)
            result['python_equal']=all(native[k]==python[k] for k in ('tokens','raw','stop_reason'))
            reports[phase][name]=result
    return reports


if __name__=='__main__':
    import json
    from coverage_data import ROOT
    folder=ROOT/'tests/fixtures/coverage'
    report=recorded_report(folder)
    with (folder/'results/scores.json').open('x') as out: json.dump(report,out,indent=2); out.write('\n')
    for phase,pages in report.items():
        print(phase, 'accepted',sum(p['accepted'] for p in pages.values()),'/',len(pages))
        for name,p in pages.items(): print(name,p['text_exact'],p['word_edits'],p['reference_words'],[d['code'] for d in p['diagnostics']])
