"""Create deterministic public PDF fixtures; reportlab/pypdf are test tools only."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import reportlab
import pypdf
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(320, 448), invariant=1)
    pdf.setTitle("Native Common Lisp document fixture")
    pdf.drawImage(str(root / "examples/data/native-page.png"), 0, 0, width=320, height=448)
    pdf.showPage()
    pdf.setPageSize((448, 320))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(28, 268, "BUILD TOGETHER")
    pdf.setFont("Helvetica", 15)
    for y, line in [(220, "1. Read the page."), (190, "2. Preserve the raw output."),
                    (160, "3. Share a small failing example.")]:
        pdf.drawString(28, y, line)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(28, 30, "Public synthetic fixture - rotation metadata is intentional.")
    pdf.showPage()
    pdf.save()
    reader = PdfReader(io.BytesIO(stream.getvalue()))
    writer = PdfWriter()
    writer.append(reader)
    writer.pages[1].rotate(90)
    # No timestamps or random document ID in the unencrypted fixture.
    with (args.output / "native-pages.pdf").open("wb") as out:
        writer.write(out)
    encrypted = PdfWriter()
    encrypted.append(reader)
    encrypted.encrypt("fixture-only", algorithm="RC4-40")
    # Encryption contains a generated file ID: metadata marks this fixture non-reproducible.
    with (args.output / "encrypted.pdf").open("wb") as out:
        encrypted.write(out)
    manifest = {"reportlab": reportlab.Version, "pypdf": pypdf.__version__,
                "page_sizes_points": [[320, 448], [448, 320]], "rotation": [0, 90],
                "encryption": "Public rejection fixture only; RC4-40 is NOT a security recommendation",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.glob("*.pdf"))}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
