"""Independent OTSL cell fixtures from pinned docling-core; no Lisp expectations."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from docling_core.types.doc.utils import parse_otsl_table_content


def sexp(value):
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(value, int):
        return str(value)
    return '(' + ' '.join(map(sexp, value)) + ')'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged", action="store_true")
    args = parser.parse_args()
    assert importlib.metadata.version("docling-core") == "2.97.0"
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {
        "headless": "<fcel>Lisp<fcel>Native<nl><fcel>Python<fcel>Reference<nl>",
        "headed": "<ched>Component<ched>Status<nl><fcel>Vision<fcel>Native<nl><fcel>Tables<fcel>Rectangular subset<nl>",
        "empty": "<ecel><fcel>Right<nl><fcel>Left<ecel><nl>",
        "unicode": "<fcel> 日本語 <fcel>Überblick<nl>",
        "escaped": "<fcel>x|y<fcel>one\ntwo<nl><fcel>*literal*<fcel>C:\\notes & more<nl>",
        "located": "<loc_0><loc_1><loc_499><loc_400><fcel>A<fcel>B<nl>",
    }
    if args.merged:
        cases = {
            "horizontal": "<fcel>Common Lisp<lcel><lcel><nl><fcel>Vision<fcel>Decoder<fcel>Tables<nl>",
            "vertical": "<fcel>Native<fcel>Vision<nl><ucel><fcel>Language<nl><ucel><fcel>Document<nl>",
            "two-dimensional": "<fcel>One region<lcel><nl><ucel><xcel><nl><ucel><xcel><nl>",
            "mixed": "<fcel>A<lcel><fcel>B<nl><ucel><xcel><fcel>C<nl><fcel>D<fcel>E<ucel><nl>",
            "headed": "<ched>Native document pipeline<lcel><ched>Result<nl><fcel>Common Lisp<fcel>Vision<fcel>Image features<nl><ucel><fcel>Decoder<fcel>DocTags<nl><ucel><fcel>Tables<fcel>Markdown + spans<nl>",
            "empty": "<ecel><lcel><fcel>Right<nl><ucel><xcel><fcel>Bottom<nl>",
            "escaped": "<fcel>*literal* | C:\\notes & more \"quoted\" 'text'\nsecond line<lcel><nl>",
            "unicode": "<fcel> 日本語 <lcel><nl><fcel>Überblick<fcel>文書<nl>",
        }
    entries = []
    for name, body in cases.items():
        otsl = "<otsl>" + body + "</otsl>"
        data = parse_otsl_table_content(otsl)
        if not args.merged:
            assert all(c.row_span == c.col_span == 1 for c in data.table_cells)
        cells = [[c.start_row_offset_idx, c.start_col_offset_idx,
                  *([c.row_span, c.col_span] if args.merged else []), c.text] for c in data.table_cells]
        raw = "<doctag>" + otsl + "</doctag>"
        entries.append(f"(:name {sexp(name)} :raw {sexp(raw)} :rows {data.num_rows} :columns {data.num_cols} :cells {sexp(cells)})")
        (args.output / (name + ".doctags")).write_text(raw, encoding="utf-8")
        (args.output / (name + ".json")).write_text(json.dumps({"rows": data.num_rows, "columns": data.num_cols, "cells": cells}, ensure_ascii=False, indent=2) + "\n")
    (args.output / "cases.sexp").write_text('(\n' + '\n'.join(entries) + '\n)\n', encoding="utf-8")
    manifest = {"docling-core": "2.97.0", "license": "MIT",
                "source": "docling_core.types.doc.utils.parse_otsl_table_content",
                "source_sha256": hashlib.sha256(Path(inspect.getfile(parse_otsl_table_content)).read_bytes()).hexdigest(),
                "scope": ("Exact rows, columns and merged cell offsets/spans/text; NOT Python Markdown/header-role equivalence" if args.merged else
                          "Exact rows, columns and unmerged cell offsets/text; NOT Python Markdown/header-role equivalence"),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.name != "manifest.json"}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(cases)} independent {'merged' if args.merged else 'rectangular'} OTSL fixtures")


if __name__ == "__main__":
    main()
