"""Authored inputs only; no Hub, model, Arrow or tokenizer dependency."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dataset_audit import ROOT, select_rows, inspect_label, verify_file, native_probe


def row(number=1, language="en", text="<doctag><text>Hello</text></doctag>"):
    return dict(id=f"doc_{'a' * 40}_p{number:05d}", language=language,
                doctags=text, markdown="Hello")


class DatasetAuditTests(unittest.TestCase):
    def test_retained_report_policy_and_totals(self):
        report = json.loads((ROOT / "tests/fixtures/dataset-audit/report.json").read_text())
        lock = ROOT / "references/dataset-audit.lock.json"
        self.assertEqual(report["protocol"], json.loads(lock.read_text()))
        self.assertEqual(report["protocol_sha256"], hashlib.sha256(lock.read_bytes()).hexdigest())
        cases = report["cases"]
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c["source_id"] for c in cases}), 10)
        self.assertEqual(len({c["id"] for c in cases}), 20)
        self.assertEqual([c["shard_row"] for c in cases], sorted(c["shard_row"] for c in cases))
        self.assertEqual(report["summary"], dict(
            pages=20, sources=10, native_renderable=0,
            pages_with_out_of_range_locations=20, answer_alone_exceeds_context=0,
            min_answer_tokens=116, max_answer_tokens=1315))
        for c in cases:
            self.assertFalse(c["markdown_renderable"])
            self.assertIn("location", c["diagnostic_codes"])
            self.assertGreater(c["out_of_range_locations"], 0)
            self.assertLess(c["answer_tokens_without_eos"], report["model_context"])
            self.assertNotIn("doctags", c)
        self.assertEqual(sum("unsupported-tag" in c["diagnostic_codes"] for c in cases), 9)
        self.assertEqual(sum("ambiguous-entity" in c["diagnostic_codes"] for c in cases), 2)

    def test_stored_order_not_content_selection(self):
        rows = [row(1, "de"), row(3), row(2), row(4)]
        self.assertEqual([r[0] for r in select_rows(rows, 2)], [1, 2])
        self.assertEqual(select_rows(rows, 2)[0][1]["id"], row(3)["id"])

    def test_too_small_sample_fails(self):
        with self.assertRaises(ValueError):
            select_rows([row()], 2)

    def test_duplicate_ids_and_invalid_identity_fail(self):
        for rows in ([row(), row()], [dict(row(), id="../unsafe")]):
            with self.assertRaises(ValueError):
                select_rows(rows, len(rows))

    def test_sample_bounds(self):
        for size in (0, 101, True):
            with self.assertRaises(ValueError):
                select_rows([row()], size)

    def test_location_bounds_and_source_group(self):
        result = inspect_label(row(text="<text><loc_0><loc_1><loc_499><loc_500>x</text>"))
        self.assertEqual(result["out_of_range_locations"], 1)
        self.assertEqual(result["maximum_location"], 500)
        self.assertEqual(result["source_id"], "doc_" + "a" * 40)
        self.assertNotIn("doctags", result)

    def test_empty_and_nontext_label_rejected(self):
        for value in ("", None, 15):
            with self.assertRaises(ValueError):
                inspect_label(row(text=value))

    def test_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_bytes(b"abc")
            wanted = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(verify_file(path, wanted, 3), wanted)
            for digest, size in (("0" * 64, 3), (wanted, 2)):
                with self.assertRaises(ValueError):
                    verify_file(path, digest, size)

    def test_actual_lisp_parser_retains_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            labels = ["<doctag><text>Hello</text></doctag>",
                      "<doctag><text><loc_0><loc_0><loc_600><loc_700>x</text></doctag>",
                      "<doctag><unknown>x</unknown></doctag>",
                      "<doctag><text>unfinished"]
            files = []
            for i, label in enumerate(labels):
                path = folder / f"{i:03d}.doctags"
                path.write_text(label)
                files.append(path)
            results = native_probe(files)
            self.assertTrue(results[0]["markdown_renderable"])
            self.assertIn("location", results[1]["diagnostic_codes"])
            self.assertFalse(results[2]["markdown_renderable"])
            self.assertFalse(results[3]["markdown_renderable"])


if __name__ == "__main__":
    unittest.main()
