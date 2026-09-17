"""Independent supported-subset fixtures from docling-core, never a runtime fallback.

Run with the isolated environment in references/doctags/uv.lock. Output is generated
data; this script neither imports Lisp results nor encodes their Markdown expectations.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from docling_core.types.doc.tokens import DocumentToken
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer


def lisp_string(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    assert importlib.metadata.version("docling-core") == "2.97.0"
    evidence = json.loads((root / "docs/evidence/m3-generation-cpu.json").read_text())
    cases = {
        "native-page": evidence["raw"],
        "unicode": "<doctag><title>Research notes</title><section_header_level_2>Überblick 日本語</section_header_level_2><text>Read carefully.</text><text>Build together.</text></doctag>",
        "lists": "<doctag><ordered_list><list_item>First</list_item><list_item>Second</list_item></ordered_list><unordered_list><list_item>Third</list_item><list_item>Fourth</list_item></unordered_list></doctag>",
        "multiline": "<doctag><section_header_level_1>Two\nlines</section_header_level_1></doctag>",
    }
    entries = []
    for name, raw in cases.items():
        document = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], None))
        # Normalize only final newline for text files; preserve all other serializer bytes.
        markdown = document.export_to_markdown().rstrip("\n") + "\n"
        (args.output / f"{name}.doctags").write_text(raw, encoding="utf-8")
        (args.output / f"{name}.md").write_text(markdown, encoding="utf-8")
        texts = [item.text for item in document.texts]
        entries.append(f"(:input {lisp_string(name + '.doctags')} :markdown {lisp_string(name + '.md')} :texts ({' '.join(map(lisp_string, texts))}))")
        print(name, repr(markdown))
    (args.output / "cases.sexp").write_text("(\n" + "\n".join(entries) + "\n)\n", encoding="utf-8")
    source_hashes = {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                     for cls in (DoclingDocument, DocumentToken, MarkdownDocSerializer)}
    manifest = {"docling-core": "2.97.0", "source_sha256": source_hashes,
                "source": "https://github.com/docling-project/docling-core/tree/v2.97.0",
                "license": "MIT", "normalization": "Exactly one terminal newline in Markdown only",
                "native_page_evidence_sha256": hashlib.sha256((root / "docs/evidence/m3-generation-cpu.json").read_bytes()).hexdigest(),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.name != "manifest.json"}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
