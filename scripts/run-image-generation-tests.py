"""Offline oracle/native qualification runner. All children are foreground and joined."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--engine", type=Path, default=ROOT.parent / "cl-transformer-blocks")
    p.add_argument("--checkpoint", type=Path, default=ROOT / ".build/smoldocling")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--reuse-reference", action="store_true")
    args = p.parse_args()
    fixture = ROOT / ".build" / ("image-tiny" if args.tiny else "image-page")
    checkpoint = ROOT / "tests/fixtures/model-tiny" if args.tiny else args.checkpoint.resolve()
    if not args.reuse_reference:
        command = [sys.executable, "scripts/image-generation-reference.py", "--checkpoint", str(checkpoint),
                   "--output", str(fixture)] + (["--tiny"] if args.tiny else [])
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(["make", "build-images"], cwd=ROOT, check=True)
    env = {**os.environ, "DOCLING_ENGINE": str(args.engine.resolve()), "DOCLING_FIXTURE": str(fixture),
           "DOCLING_MODEL": str(checkpoint), "TB_DEVICE": args.device}
    subprocess.run(["sbcl", "--noinform", "--no-userinit", "--no-sysinit", "--script", "scripts/test-image-generation.lisp"],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
