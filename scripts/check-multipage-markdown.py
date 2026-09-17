"""Independent Markdown/HTML structure check of the actual native combined output."""
import argparse
import json
from pathlib import Path
import runpy

from markdown_it import MarkdownIt
from public_pdf_score import whitespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    markdown = args.markdown.read_text(encoding='utf-8')
    bodies = [(root / f'tests/fixtures/footers/page-{p}.md').read_text() for p in (27, 28)]
    assert markdown == '# Page 27\n\n' + bodies[0] + '\n---\n\n# Page 28\n\n' + bodies[1]
    md = MarkdownIt('commonmark').enable('table')
    tokens = md.parse(markdown)
    headings = [tokens[i+1].content for i, t in enumerate(tokens) if t.type == 'heading_open' and t.tag == 'h1']
    assert headings == ['Page 27', 'Page 28'], headings
    assert sum(t.type == 'hr' for t in tokens) == 1
    assert sum(t.type == 'table_open' for t in tokens) == 2
    # Reconstruct every table cell independently, across the page separator.
    reader_class = runpy.run_path(str(root / 'scripts/check-table-markdown.py'))['TableReader']
    reader = reader_class()
    reader.feed(md.render(markdown))
    assert reader.tags.count('table') == 2
    actual = []
    for row in reader.rows:
        assert len(row) == 4
        for text, attrs, tag in row:
            assert not attrs
            actual.append(whitespace(text))
    wanted = []
    for page in (27, 28):
        result = json.loads((root / f'tests/fixtures/public-pdf/results/native/page-{page}.json').read_text())
        wanted.extend(whitespace(c[4]) for c in result['tables'][0]['cells'])
    assert actual == wanted
    print('PASS: 2 labeled pages, 1 separator, 2 tables, all 64 original cells and exact per-page bodies')


if __name__ == '__main__':
    main()
