"""Run the independent reference, native gate, and fresh Python reload; join children."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--kind", choices=["tiny", "tied", "real"], default="tiny")
    p.add_argument("--engine", type=Path, default=ROOT.parent / "cl-transformer-blocks")
    p.add_argument("--checkpoint", type=Path, default=ROOT / ".build/smoldocling")
    p.add_argument("--reuse-reference", action="store_true")
    args = p.parse_args()
    fixture = ROOT / ".build" / ("model-" + args.kind)
    source = args.checkpoint.resolve() if args.kind == "real" else fixture
    if not args.reuse_reference:
        command = [sys.executable, "scripts/model-reference.py", "--output", str(fixture)]
        if args.kind == "real":
            command += ["--checkpoint", str(source)]
        if args.kind == "tied":
            command += ["--tied"]
        subprocess.run(command, cwd=ROOT, check=True)
    stage = Path(tempfile.mkdtemp(prefix=f"export-{args.kind}-{args.device}-", dir=ROOT / ".build"))
    export = stage / "checkpoint"
    env = {**os.environ, "DOCLING_ENGINE": str(args.engine.resolve()), "DOCLING_FIXTURE": str(fixture),
           "DOCLING_MODEL": str(source), "DOCLING_EXPORT": str(export), "TB_DEVICE": args.device}
    subprocess.run(["sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script", "scripts/test-model.lisp"],
                   cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "scripts/check-model-export.py", "--source", str(source),
                    "--export", str(export), "--fixture", str(fixture)], cwd=ROOT, check=True)
    shutil.copyfile(export / "reload-evidence.json", fixture / f"reload-{args.device}.json")
    print(f"Kept local export at {export}", flush=True)


if __name__ == "__main__":
    main()
