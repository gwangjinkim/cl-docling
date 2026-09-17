"""Check retained CPU/Metal/Python evidence and independently reconstruct accepted GFM."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

from docling_core.types.doc.utils import parse_otsl_table_content
from markdown_it import MarkdownIt
from table_score import score
import importlib.util

# Reuse the independent HTML reader (hyphenated CLI filename).
spec = importlib.util.spec_from_file_location("table_markdown", Path(__file__).with_name("check-table-markdown.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser()
    for name in ("cpu", "metal", "python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/fixtures/table-pages"
    report = {"scope": "Two frozen synthetic pages, not held-out OCR or a speed benchmark",
              "versions": {name: importlib.metadata.version(name) for name in ("docling-core", "markdown-it-py")},
              "cases": {}}
    for name in ("grid", "merged"):
        cpu, metal, reference = [json.loads((folder / f"{name}.json").read_text())
                                 for folder in (args.cpu, args.metal, args.python)]
        assert cpu["device"] == "cpu" and metal["device"] == "gpu"
        for key in ("tokens", "raw", "stop_reason"):
            assert cpu[key] == metal[key] == reference[key], (name, key)
        for key in ("tables", "diagnostics", "markdown_written", "non_table_elements"):
            assert cpu[key] == metal[key], (name, key)
        for folder, result in ((args.cpu, cpu), (args.metal, metal)):
            assert (folder / f"{name}.doctags").read_text() == result["raw"]
            assert (folder / f"{name}.md").exists() == result["markdown_written"]
        image_hash = hashlib.sha256((fixtures / f"{name}.png").read_bytes()).hexdigest()
        assert image_hash == reference["image_sha256"]
        expected = json.loads((fixtures / f"{name}.json").read_text())
        scores = score(expected, metal)
        independent_export = False
        if scores["accepted"]:
            # No recovery/repair of rejected model tables; validate accepted output only.
            otsl = re.findall(r"<otsl>.*?</otsl>", metal["raw"], re.DOTALL)
            assert len(otsl) == 1
            parsed = parse_otsl_table_content(otsl[0])
            cells = [[c.start_row_offset_idx, c.start_col_offset_idx, c.row_span, c.col_span, c.text]
                     for c in parsed.table_cells]
            assert cells == expected["cells"]
            assert [parsed.num_rows, parsed.num_cols] == [expected["rows"], expected["columns"]]
            # This two-page experiment has exactly one accepted, unmerged grid.
            assert all(cell[2:4] == [1, 1] for cell in cells)
            reader = module.TableReader()
            reader.feed(MarkdownIt("commonmark").enable("table").render((args.metal / f"{name}.md").read_text()))
            reconstructed = [[r, c, 1, 1, cell[0]] for r, row in enumerate(reader.rows) for c, cell in enumerate(row)]
            assert reconstructed == expected["cells"]
            independent_export = True
        report["cases"][name] = {"cpu_metal_python_ids_and_raw_equal": True,
                                  "tokens": len(metal["tokens"]), "tiles": reference["tiles"],
                                  "prompt_tokens": reference["prompt_tokens"], "image_sha256": image_hash,
                                  "independent_accepted_export_check": independent_export, "scores": scores}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print("PASS: CPU/Metal/Python evidence matches; accepted table independently reconstructed.")
    print("Recognition acceptance: " + ", ".join(f"{name}={case['scores']['accepted']}" for name, case in report["cases"].items()))


if __name__ == "__main__":
    main()
