"""Independent docling-core reconstruction of labels, without any model inference."""
from docling_core.types.doc.utils import parse_otsl_table_content
from adaptation_data import cases, answer
import sys

if '--selection-experiment' in sys.argv:
    from selection_data import cases

for case in cases():
    expected = case['expected']
    target = answer(expected).removeprefix('<doctag>').removesuffix('</doctag>')
    parsed = parse_otsl_table_content(target)
    actual = [[c.start_row_offset_idx, c.start_col_offset_idx, c.row_span, c.col_span, c.text]
              for c in parsed.table_cells]
    assert actual == expected['cells'], (case['name'], actual)
    assert (parsed.num_rows, parsed.num_cols) == (expected['rows'], expected['columns'])
print(f'PASS: all {len(cases())} independently reconstructed targets agree with expected cells/spans/text.')
