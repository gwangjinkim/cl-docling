"""Reconstruct actual Lisp GFM/HTML tables with independent Markdown/HTML parsers."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path

from markdown_it import MarkdownIt


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cell = None
        self.attributes = None
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = ""
            self.attributes = dict(attrs)
        elif tag == "br" and self.cell is not None:
            self.cell += "\n"

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.rows[-1].append((self.cell, self.attributes, tag))
            self.cell = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--merged", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = "merged-tables" if args.merged else "tables"
    reference = json.loads((root / "tests/fixtures" / folder / (args.case + ".json")).read_text())
    reader = TableReader()
    reader.feed(MarkdownIt("commonmark").enable("table").render(args.markdown.read_text()))
    assert set(reader.tags) <= {"table", "thead", "tbody", "tr", "td", "th", "br"}, reader.tags
    assert reader.tags.count("table") == 1, "exactly one rendered table"
    if not args.merged and args.case != "headed":
        assert [cell[0] for cell in reader.rows[0]] == [""] * reference["columns"], "data must not be promoted to header"
        reader.rows.pop(0)
    assert len(reader.rows) == reference["rows"]
    occupied = set()
    actual = []
    for r, row in enumerate(reader.rows):
        c = 0
        for text, attrs, tag in row:
            assert set(attrs) <= {"rowspan", "colspan"}, attrs
            assert tag == ("th" if args.case == "headed" and r == 0 else "td"), "header policy"
            while (r, c) in occupied:
                c += 1
            rs, cs = int(attrs.get("rowspan", 1)), int(attrs.get("colspan", 1))
            assert rs > 0 and cs > 0
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    assert rr < reference["rows"] and cc < reference["columns"], "span outside table"
                    assert (rr, cc) not in occupied, "overlapping HTML cells"
                    occupied.add((rr, cc))
            actual.append([r, c, *([rs, cs] if args.merged else []), text])
            c += cs
    assert len(occupied) == reference["rows"] * reference["columns"], "uncovered grid slot"
    assert actual == reference["cells"], (actual, reference["cells"])
    print(f"PASS: {'HTML spans' if args.merged else 'GFM'} reconstruct {args.case} exactly")


if __name__ == "__main__":
    main()
