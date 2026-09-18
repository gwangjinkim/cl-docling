"""Frozen four-row smoke policy; Markdown quality is not plain-text OCR accuracy."""
from collections import Counter
import re
from coverage_score import distance


def normalized(text):
    return ' '.join(text.split())


def validate_manifest(manifest):
    cases=manifest.get('cases',[])
    if manifest.get('training_approved') is not False or len(cases)!=4:
        raise ValueError('Exactly four regression-only cases required')
    for i,c in enumerate(cases):
        if c.get('row')!=i or c.get('split')!='regression' or c.get('name')!=f'page-{i:03d}':
            raise ValueError('Changed selection or unsafe case path')
        for key in ('image_sha256','ground_truth_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', c.get(key,'')):
                raise ValueError('Invalid digest')
        if not isinstance(c.get('document_id'),str) or not c['document_id']:
            raise ValueError('Missing document identity')
    for key in ('document_id','image_sha256','ground_truth_sha256'):
        if len({c[key] for c in cases})!=4:
            raise ValueError('Duplicate benchmark item; review the fixed sample, do not substitute')


def cell_inventory(tables):
    return Counter((i,*cell[:4],normalized(cell[4]))
                   for i,table in enumerate(tables) for cell in table['cells'])


def score(expected,result,markdown):
    reference=normalized(expected['markdown'])
    if not reference:
        raise ValueError('Empty ground-truth Markdown')
    result=result or {}
    strict=bool(not result.get('error') and result.get('markdown_written') and
                result.get('stop_reason')=='eos' and not result.get('diagnostics') and
                isinstance(markdown,str))
    actual=normalized(markdown) if strict else ''
    characters=distance(reference,actual)
    words=distance(reference.split(),actual.split())
    expected_tables=expected['tables']; found_tables=result.get('tables',[])
    expected_cells=cell_inventory(expected_tables); found_cells=cell_inventory(found_tables)
    matches=sum((expected_cells & found_cells).values())
    shapes=lambda ts:[(t['rows'],t['columns']) for t in ts]
    table_exact=shapes(expected_tables)==shapes(found_tables) and expected_cells==found_cells
    # Diagnostic only: tags removed without semantic repair, even for blocked output.
    reference_text=normalized(expected.get('text',expected['markdown']))
    raw_text=normalized(re.sub(r'<[^>]*>',' ',result.get('raw','')))
    raw_character_edits=distance(reference_text,raw_text)
    raw_word_edits=distance(reference_text.split(),raw_text.split())
    return dict(strict_export=strict,markdown_exact=reference==actual,
                accepted=strict and reference==actual and table_exact,
                character_edits=characters,reference_characters=len(reference),
                word_edits=words,reference_words=len(reference.split()),
                markdown_cer=characters/len(reference),markdown_wer=words/len(reference.split()),
                expected_tables=len(expected_tables),predicted_tables=len(found_tables),
                expected_cells=sum(expected_cells.values()),predicted_cells=sum(found_cells.values()),
                exact_cell_matches=matches,tables_exact=table_exact,
                stop_reason=result.get('stop_reason'),diagnostic_codes=[d['code'] for d in result.get('diagnostics',[])],
                execution_error=bool(result.get('error')),
                raw_text_character_edits=raw_character_edits,raw_text_reference_characters=len(reference_text),
                raw_text_word_edits=raw_word_edits,raw_text_reference_words=len(reference_text.split()))
