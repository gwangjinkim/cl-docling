"""Independent docling-core reconstruction of every accepted experiment output."""
import argparse
import json
from pathlib import Path
import re
from docling_core.types.doc.utils import parse_otsl_table_content
from adaptation_data import ROOT
from table_score import score

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
fixtures = ROOT / 'tests/fixtures/selection-pages'
checked = 0
outputs = [p for phase in args.results.glob('development/step-*') for p in phase.glob('*.json')]
outputs += [p for phase in ('base', 'adapted') for p in (args.results / 'final' / phase).glob('*.json')]
assert len(outputs) == 24, 'Expected every candidate and final generation'
for path in outputs:
    result = json.loads(path.read_text())
    expected = json.loads((fixtures / path.name).read_text())
    if score(expected, result)['accepted']:
        matches = re.findall(r'<otsl>.*?</otsl>', result['raw'], re.DOTALL)
        assert len(matches) == 1
        table = parse_otsl_table_content(matches[0])
        cells = [[c.start_row_offset_idx, c.start_col_offset_idx, c.row_span, c.col_span, c.text]
                 for c in table.table_cells]
        assert cells == expected['cells'], path
        assert (table.num_rows, table.num_cols) == (expected['rows'], expected['columns']), path
        checked += 1
print(f'PASS: retained all {len(outputs)} outputs; independently reconstructed {checked} accepted tables.')
