"""Foreground-only full-page native training and fresh Python export verification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="New output or an existing independently generated reference")
    parser.add_argument("--output", type=Path, required=True, help="New native adapter directory")
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--reuse-reference", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    checkpoint, reference, output = (p.resolve() for p in (args.checkpoint, args.reference, args.output))

    def run(*command, env=None):
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)

    if not args.reuse_reference:
        run(sys.executable, "scripts/page-training-reference.py", "--checkpoint", checkpoint, "--output", reference)
    env = {**os.environ, "DOCLING_MODEL": str(checkpoint), "DOCLING_PAGE_REFERENCE": str(reference),
           "DOCLING_PAGE_OUTPUT": str(output), "TB_DEVICE": args.device}
    run("sbcl", "--dynamic-space-size", "4096", "--noinform", "--no-sysinit", "--no-userinit",
        "--script", "scripts/test-page-training.lisp", env=env)
    run(sys.executable, "scripts/page-training-reference.py", "--checkpoint", checkpoint,
        "--output", reference, "--verify-adapter", output)
    print(f"PASS: full-page native/Python training qualification on {args.device}", flush=True)


if __name__ == "__main__":
    main()
