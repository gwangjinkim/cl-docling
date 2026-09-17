"""Generate independent fixtures and run native Lisp tests; all children are joined."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--engine", type=Path, default=ROOT.parent / "cl-transformer-blocks")
    parser.add_argument("--real", action="store_true", help="Requires previously downloaded pinned snapshot")
    parser.add_argument("--padded", action="store_true")
    args = parser.parse_args()
    if args.padded and not args.real:
        parser.error("--padded is for the real checkpoint; tiny fixtures already contain padding")
    fixture = ROOT / ".build" / ("vision-real-padded" if args.padded else "vision-real" if args.real else "vision-tiny")
    reference = [sys.executable, "scripts/vision-reference.py", "--output", str(fixture)]
    if args.real:
        reference += ["--checkpoint", str(ROOT / ".build/smoldocling")]
    if args.padded:
        reference += ["--padded"]
    subprocess.run(reference, cwd=ROOT, check=True)
    environment = {**os.environ, "DOCLING_ENGINE": str(args.engine.resolve()),
                   "DOCLING_FIXTURE": str(fixture), "TB_DEVICE": args.device,
                   "DOCLING_MODEL": str(ROOT / ".build/smoldocling") if args.real else str(fixture)}
    subprocess.run(["sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script",
                    "scripts/test-vision.lisp"], cwd=ROOT, env=environment, check=True)
    if not args.real:
        subprocess.run(["sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script",
                        "scripts/test-vision-lifetimes.lisp"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
