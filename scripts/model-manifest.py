"""Record/verify the pinned official model bytes without copying weights into Git."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "docling-project/SmolDocling-256M-preview"
REVISION = "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8"
FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
         "processor_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", "chat_template.json",
         "merges.txt", "vocab.json", "model.safetensors", "README.md"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--download", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=args.directory, allow_patterns=FILES)
    manifest = {"schema_version": 1, "model_id": MODEL, "revision": REVISION,
                "license": "CDLA-Permissive-2.0 (see upstream README model card)",
                "files": {name: {"sha256": digest(args.directory / name),
                                  "bytes": (args.directory / name).stat().st_size} for name in FILES}}
    if args.check:
        if json.loads(args.check.read_text()) != manifest:
            raise SystemExit("FAIL: model snapshot differs from pinned manifest")
        print("PASS: all pinned model/processor/tokenizer bytes match")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()
