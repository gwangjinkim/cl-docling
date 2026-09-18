"""Frozen Markdown/structure metrics; unsupported references remain unscorable."""
from coverage_score import distance


def normalized(text):
    return ' '.join(text.split())


def score_case(reference, result):
    result = result or dict(error='missing-output')
    strict = (not result.get('error') and result.get('stop_reason') == 'eos' and
              not result.get('diagnostics') and isinstance(result.get('markdown'), str))
    report = dict(reference_available=reference is not None, strict_export=bool(strict),
                  execution_error=bool(result.get('error')), stop_reason=result.get('stop_reason'),
                  diagnostic_codes=result.get('diagnostics', []), generated_tokens=result.get('tokens', 0),
                  accepted=None, structure_exact=None, markdown_exact=None,
                  character_edits=None, reference_characters=None, word_edits=None, reference_words=None,
                  markdown_cer=None, markdown_wer=None)
    if reference is None:
        return report
    expected = normalized(reference['markdown'])
    if not expected or not isinstance(reference['signature'], list) or not reference['signature']:
        raise ValueError('Empty or incomplete reviewed reference')
    actual = normalized(result['markdown']) if strict else ''
    ce, we = distance(expected, actual), distance(expected.split(), actual.split())
    structure = strict and reference['signature'] == result.get('signature')
    report.update(accepted=bool(strict and expected == actual and structure), structure_exact=bool(structure),
                  markdown_exact=expected == actual, character_edits=ce, reference_characters=len(expected),
                  word_edits=we, reference_words=len(expected.split()),
                  markdown_cer=ce/len(expected), markdown_wer=we/len(expected.split()))
    return report


def summarize(cases, reports):
    names = [c['name'] for c in cases]
    if not names or len(set(names)) != len(names) or set(names) != set(reports):
        raise ValueError('Missing, duplicate or extra scored case')
    if any(c['split'] not in ('train', 'validation') for c in cases):
        raise ValueError('Development scorer cannot admit final cases')
    output = {}
    for split in sorted({c['split'] for c in cases}):
        all_pages = [reports[c['name']] for c in cases if c['split'] == split]
        reviewed = [r for r in all_pages if r['reference_available']]
        accepted = sum(r['accepted'] is True for r in reviewed)
        complete = len(reviewed) == len(all_pages)
        row = dict(pages=len(all_pages), reviewed_pages=len(reviewed), unscorable_pages=len(all_pages)-len(reviewed),
                   strict_exports=sum(r['strict_export'] for r in all_pages),
                   execution_errors=sum(r['execution_error'] for r in all_pages),
                   verified_conversions=accepted, verified_conversion_yield=accepted/len(all_pages),
                   reviewed_exact_rate=accepted/len(reviewed) if reviewed else None,
                   reference_coverage_complete=complete, full_quality_gate=complete and accepted == len(all_pages))
        for metric, edits, total in [('cer', 'character_edits', 'reference_characters'),
                                      ('wer', 'word_edits', 'reference_words')]:
            numerator, denominator = sum(r[edits] for r in reviewed), sum(r[total] for r in reviewed)
            row['reviewed_' + edits] = numerator
            row['reviewed_' + total] = denominator
            row['reviewed_markdown_' + metric] = numerator/denominator if denominator else None
        output[split] = row
    return output
