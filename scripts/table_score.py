"""Exact synthetic one-table recognition scoring, not a general table/OCR metric."""
import argparse
from collections import Counter
import json
from pathlib import Path


def score(expected, result):
    tables = result["tables"]
    wanted = Counter(map(tuple, expected["cells"]))
    count = sum(len(table["cells"]) for table in tables)
    comparable = len(tables) == 1 and tables[0]["renderable"]
    actual = Counter(map(tuple, tables[0]["cells"])) if comparable else Counter()
    matches = sum((wanted & actual).values())
    wanted_text = Counter(cell[4] for cell in expected["cells"])
    actual_text = Counter(cell[4] for table in tables for cell in table["cells"])
    wanted_geometry = Counter(tuple(cell[:4]) for cell in expected["cells"])
    actual_geometry = Counter(tuple(cell[:4]) for cell in tables[0]["cells"]) if comparable else Counter()
    geometry = sum((wanted_geometry & actual_geometry).values())
    dimensions = comparable and all(tables[0][key] == expected[key] for key in ("rows", "columns"))
    exact = bool(dimensions and wanted == actual)
    accepted = (exact and result["stop_reason"] == "eos" and not result["diagnostics"]
                and result["non_table_elements"] == 0 and result["markdown_written"])
    return {"accepted": accepted, "table_count": len(tables), "expected_cells": sum(wanted.values()),
            "recognized_cells": count, "exact_cell_matches": matches, "geometry_matches": geometry,
            "structure_score_eligible": bool(comparable),
            "cell_text_inventory_matches": sum((wanted_text & actual_text).values()),
            "cell_precision": matches/count if count else 0,
            "cell_recall": matches/sum(wanted.values()), "exact_table": exact,
            "dimensions_match": bool(dimensions), "stop_reason": result["stop_reason"],
            "diagnostics": result["diagnostics"], "non_table_elements": result["non_table_elements"],
            "markdown_written": result["markdown_written"],
            "missing_cells": list((wanted - actual).elements()),
            "unexpected_cells": list((actual - wanted).elements())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/table-pages")
    parser.add_argument("--require-exact", action="store_true", help="Exit unsuccessfully if any page is not accepted")
    args = parser.parse_args()
    reports = {}
    for name in ("grid", "merged"):
        expected = json.loads((args.fixtures / f"{name}.json").read_text())
        result = json.loads((args.results / f"{name}.json").read_text())
        reports[name] = score(expected, result)
    print(json.dumps(reports, indent=2))
    if args.require_exact and not all(report["accepted"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
