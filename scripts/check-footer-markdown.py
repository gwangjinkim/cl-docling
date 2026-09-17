"""Independently reconstruct native replay Markdown; never rerun inference."""
import argparse
import json
from pathlib import Path
import runpy

from markdown_it import MarkdownIt
from public_pdf_score import score_page, whitespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reader_class = runpy.run_path(str(root / 'scripts/check-table-markdown.py'))['TableReader']
    expected = json.loads((root / 'tests/fixtures/public-pdf/expected.json').read_text())
    reports = {}
    for name, rows in expected.items():
        old = json.loads((root / f'tests/fixtures/public-pdf/results/native/{name}.json').read_text())
        markdown = (args.markdown / (name + '.md')).read_text()
        reader = reader_class()
        reader.feed(MarkdownIt('commonmark').enable('table').render(markdown))
        assert reader.tags.count('table') == 1
        assert len(reader.rows) == len(rows)
        cells = []
        for r, row in enumerate(reader.rows):
            assert len(row) == 4
            for c, (text, attrs, tag) in enumerate(row):
                assert not attrs and tag == ('th' if r == 0 else 'td')
                cells.append([r, c, 1, 1, text])
        assert [[*c[:4], whitespace(c[4])] for c in cells] == [
            [*c[:4], whitespace(c[4])] for c in old['tables'][0]['cells']]
        # Only current parser/export fields change; model text/tokens are historical.
        old['tables'][0]['cells'] = cells
        old['diagnostics'], old['markdown_written'] = [], True
        report = score_page(rows, old)
        assert not report['table_accepted']
        reports[name] = report
        print(name, report['exact_cell_matches'], '/', report['expected_cells'], 'cells; quality still fails')
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(reports, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
