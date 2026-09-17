"""Scoring contract tests: recognition success is stricter than valid Markdown."""
import copy
import json
from pathlib import Path
import unittest
from table_score import score


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"rows": 2, "columns": 2, "cells": [
            [0, 0, 2, 1, "A"], [0, 1, 1, 1, "B"], [1, 1, 1, 1, "C"]]}
        self.result = {"stop_reason": "eos", "diagnostics": [], "non_table_elements": 0,
                       "markdown_written": True, "tables": [{**copy.deepcopy(self.expected), "renderable": True}]}

    def test_exact(self):
        report = score(self.expected, self.result)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["cell_recall"], 1)
        self.assertEqual(report["geometry_matches"], 3)

    def test_wrong_text(self):
        self.result["tables"][0]["cells"][1][-1] = "b"
        report = score(self.expected, self.result)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["exact_cell_matches"], 2)
        self.assertEqual(report["geometry_matches"], 3)

    def test_wrong_span(self):
        self.result["tables"][0]["cells"][0][2] = 1
        self.assertEqual(score(self.expected, self.result)["geometry_matches"], 2)

    def test_empty_output(self):
        self.result["tables"] = []
        report = score(self.expected, self.result)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["cell_recall"], 0)
        self.assertEqual(report["cell_precision"], 0)

    def test_duplicate_penalty(self):
        self.result["tables"][0]["cells"].append([0, 1, 1, 1, "B"])
        report = score(self.expected, self.result)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["exact_cell_matches"], 3)
        self.assertEqual(report["cell_precision"], .75)

    def test_extra_table(self):
        self.result["tables"] *= 2
        report = score(self.expected, self.result)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["exact_cell_matches"], 0)

    def test_truncation_diagnostics_and_extra_content(self):
        for field, value in [("stop_reason", "length"), ("diagnostics", ["unsupported"]),
                             ("non_table_elements", 1), ("markdown_written", False)]:
            result = {**self.result, field: value}
            self.assertFalse(score(self.expected, result)["accepted"], field)

    def test_unrenderable(self):
        self.result["tables"][0]["renderable"] = False
        report = score(self.expected, self.result)
        self.assertEqual(report["exact_cell_matches"], 0)
        self.assertFalse(report["structure_score_eligible"])
        self.assertEqual(report["cell_text_inventory_matches"], 3)

    def test_dimensions(self):
        self.result["tables"][0]["columns"] = 3
        self.assertFalse(score(self.expected, self.result)["accepted"])

    def test_recorded_live_outcomes(self):
        fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures"
        for name, accepted, count in [("grid", True, 12), ("merged", False, 10)]:
            expected = json.loads((fixtures / "table-pages" / f"{name}.json").read_text())
            result = json.loads((fixtures / "table-recognition/metal" / f"{name}.json").read_text())
            report = score(expected, result)
            self.assertEqual(report["accepted"], accepted)
            self.assertEqual(report["cell_text_inventory_matches"], count)
            self.assertEqual(report["structure_score_eligible"], accepted)


if __name__ == "__main__":
    unittest.main()
