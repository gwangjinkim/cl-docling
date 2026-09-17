"""Independent wrapper gate: pypdf geometry and direct Poppler/Pillow pixel equality."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pypdf
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "tests/fixtures/pdf/native-pages.pdf"
    document = pypdf.PdfReader(source)
    assert len(document.pages) == 2
    assert [p.rotation for p in document.pages] == [0, 90]
    total = 0
    source_changed_bytes = 0
    with tempfile.TemporaryDirectory(prefix="pdf-oracle-") as temporary:
        for number, page in enumerate(document.pages, 1):
            prefix = Path(temporary) / f"page-{number}"
            subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number), "-singlefile",
                            "-png", "-r", "72", str(source), str(prefix)],
                           check=True, timeout=15, capture_output=True)
            expected = Image.open(prefix.with_suffix(".png")).convert("RGB")
            actual = Image.open(args.output_directory / f"page-{number}.png").convert("RGB")
            width, height = int(page.mediabox.width), int(page.mediabox.height)
            if page.rotation in (90, 270):
                width, height = height, width
            assert actual.size == expected.size == (width, height)
            assert actual.tobytes() == expected.tobytes()
            if number == 1:
                original = Image.open(root / "examples/data/native-page.png").convert("RGB")
                assert original.size == actual.size
                source_changed_bytes = sum(a != b for a, b in zip(original.tobytes(), actual.tobytes()))
            total += len(actual.tobytes())
    print(json.dumps({"pages": 2, "exact_rgb_bytes": total, "geometry": "pypdf",
                      "pixels": "direct pdftoppm invocation, no Lisp wrapper",
                      "page1_rgb_bytes_different_from_original_png": source_changed_bytes}))


if __name__ == "__main__":
    main()
