"""Offline, pinned target-format audit. Not a converter or training loader."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = re.compile(r"(doc_[0-9a-f]{40})_p([0-9]{5,})")


def verify_file(path, sha256, size):
    if path.stat().st_size != size:
        raise ValueError(f"Wrong size: {path.name}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != sha256:
        raise ValueError(f"Wrong hash: {path.name}")
    return actual


def select_rows(rows, count):
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Sample count must be 1..100")
    selected, seen = [], set()
    for index, row in enumerate(rows):
        if row.get("language") != "en":
            continue
        identifier = row.get("id", "")
        if not isinstance(identifier, str) or not IDENTITY.fullmatch(identifier):
            raise ValueError("Unsupported source/page identity")
        if identifier in seen:
            raise ValueError("Duplicate selected page ID")
        seen.add(identifier)
        selected.append((index, row))
        if len(selected) == count:
            return selected
    raise ValueError("Insufficient English rows in the pinned shard")


def inspect_label(row):
    raw = row.get("doctags")
    if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > 2_000_000:
        raise ValueError("Missing, nontext or oversized target")
    identity = IDENTITY.fullmatch(row["id"])
    if identity is None:
        raise ValueError("Unsupported source/page identity")
    locations = [int(x) for x in re.findall(r"<loc_([0-9]+)>", raw)]
    tags = Counter(re.findall(r"</?([^<>]+)>", raw))
    return dict(id=row["id"], source_id=identity[1], page=int(identity[2]),
                target_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                target_characters=len(raw),
                location_count=len(locations),
                maximum_location=max(locations, default=None),
                out_of_range_locations=sum(x >= 500 for x in locations),
                structural_tags=sorted(t for t in tags if not t.startswith("loc_")))


def native_probe(files):
    completed = subprocess.run(
        ["sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script",
         str(ROOT / "scripts/audit-doctags.lisp"), *map(str, files)],
        capture_output=True, text=True, check=True, timeout=60)
    results = []
    for line in completed.stdout.splitlines():
        if not line.startswith("AUDIT\t"):
            continue
        _, index, renderable, codes = line.split("\t")
        if int(index) != len(results) or renderable not in ("0", "1"):
            raise ValueError("Invalid native audit record")
        results.append(dict(markdown_renderable=renderable == "1",
                            diagnostic_codes=sorted(codes.split(",")) if codes else []))
    if len(results) != len(files):
        raise ValueError("Incomplete native audit")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Optional tools are isolated from the Lisp runtime and offline unit tests.
    import pyarrow
    import pyarrow.parquet as pq
    import tokenizers
    from tokenizers import Tokenizer

    lock_path = ROOT / "references/dataset-audit.lock.json"
    lock = json.loads(lock_path.read_text())
    if {"pyarrow": pyarrow.__version__, "tokenizers": tokenizers.__version__} != lock["tools"]:
        raise ValueError("Use the exact audit tool versions in the lock")
    verify_file(args.parquet, lock["sha256"], lock["bytes"])
    model_lock = json.loads((ROOT / "references/smoldocling.lock.json").read_text())
    model_hashes = {}
    for name in ("tokenizer.json", "config.json"):
        spec = model_lock["files"][name]
        model_hashes[name] = verify_file(args.model / name, spec["sha256"], spec["bytes"])
    config = json.loads((args.model / "config.json").read_text())
    context = config["text_config"]["max_position_embeddings"]
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    table = pq.read_table(args.parquet, columns=["id", "language", "doctags"])
    selected = select_rows(table.to_pylist(), lock["sample_count"])
    # Exclusive output directory prevents silently replacing earlier evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    records, files = [], []
    for ordinal, (index, row) in enumerate(selected):
        record = inspect_label(row)
        path = args.output / f"{ordinal:03d}.doctags"
        path.write_text(row["doctags"], encoding="utf-8")
        files.append(path)
        tokens = len(tokenizer.encode(row["doctags"], add_special_tokens=False).ids)
        record.update(shard_row=index, answer_tokens_without_eos=tokens,
                      answer_alone_exceeds_context=tokens >= context)
        records.append(record)
    for record, native in zip(records, native_probe(files), strict=True):
        record.update(native)
    summary = dict(pages=len(records), sources=len({r["source_id"] for r in records}),
                   native_renderable=sum(r["markdown_renderable"] for r in records),
                   pages_with_out_of_range_locations=sum(r["out_of_range_locations"] > 0 for r in records),
                   answer_alone_exceeds_context=sum(r["answer_alone_exceeds_context"] for r in records),
                   min_answer_tokens=min(r["answer_tokens_without_eos"] for r in records),
                   max_answer_tokens=max(r["answer_tokens_without_eos"] for r in records))
    report = dict(schema_version=1, protocol=lock,
                  protocol_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                  model_revision=model_lock["revision"], model_files=model_hashes,
                  model_context=context,
                  limitations=["No image/annotation accuracy audit or model execution",
                               "No coordinate conversion; unchanged targets only",
                               "Answer length excludes prompt, image tokens and EOS",
                               "Stored-order sample, not representative corpus statistics",
                               "Source ID grouping inferred from dataset IDs, not duplicate-proof"],
                  implementation_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in [Path(__file__), ROOT / "scripts/audit-doctags.lisp",
                                                  *sorted((ROOT / "src").glob("*.lisp"))]},
                  summary=summary, cases=records)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
