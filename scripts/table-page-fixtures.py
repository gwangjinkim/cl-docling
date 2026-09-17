"""Two predetermined synthetic table pages; no inference, selection or OCR labels."""
import argparse
import hashlib
import json
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert PIL.__version__ == "12.3.0", "Use the pinned engine reference environment"
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {
        "grid": {"rows": 4, "columns": 3, "cells": [
            [0, 0, 1, 1, "Item"], [0, 1, 1, 1, "Count"], [0, 2, 1, 1, "Price"],
            [1, 0, 1, 1, "Books"], [1, 1, 1, 1, "3"], [1, 2, 1, 1, "12"],
            [2, 0, 1, 1, "Pens"], [2, 1, 1, 1, "5"], [2, 2, 1, 1, "2"],
            [3, 0, 1, 1, "Folders"], [3, 1, 1, 1, "2"], [3, 2, 1, 1, "8"]]},
        "merged": {"rows": 4, "columns": 3, "cells": [
            [0, 0, 1, 2, "Office supplies"], [0, 2, 1, 1, "Count"],
            [1, 0, 2, 1, "Paper"], [1, 1, 1, 1, "Books"], [1, 2, 1, 1, "3"],
            [2, 1, 1, 1, "Folders"], [2, 2, 1, 1, "2"],
            [3, 0, 1, 1, "Writing"], [3, 1, 1, 1, "Pens"], [3, 2, 1, 1, "5"]]},
    }
    for name, expected in cases.items():
        image = Image.new("RGB", (600, 360), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=24)
        for r, c, rs, cs, text in expected["cells"]:
            box = (30 + 180*c, 40 + 70*r, 30 + 180*(c+cs), 40 + 70*(r+rs))
            draw.rectangle(box, outline="black", width=2)
            draw.text((box[0]+12, (box[1]+box[3])/2), text, font=font, fill="black", anchor="lm")
        image.save(args.output / f"{name}.png")
        (args.output / f"{name}.json").write_text(json.dumps(expected, indent=2) + "\n")
    manifest = {"license": "MIT; synthetic project-authored tables", "pillow": PIL.__version__,
                "font": "Pillow bundled default Aileron, 24px", "size": [600, 360],
                "policy": "Both cases frozen before inference; no prompt tuning or case selection",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
