"""Authored, independent all-layer Docling oracle; no model output imported."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from docling_core.types.doc.document import ContentLayer, DocTagsDocument, DoclingDocument
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer


def lisp_string(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert importlib.metadata.version('docling-core') == '2.97.0'
    cases = {
        'ordered': '<doctag><page_header><loc_10><loc_0><loc_490><loc_20>Research notes 日本語</page_header><text>Body text.</text><footnote><loc_10><loc_420><loc_490><loc_450>1 Additional details.</footnote><page_footer>17</page_footer></doctag>',
        'leaves': '<doctag><footnote>First note.</footnote><page_header>Later header.</page_header><footnote>Second note.</footnote></doctag>',
        'header-only': '<doctag><page_header>Build together.</page_header></doctag>',
        'footnote-only': '<doctag><footnote>Unnumbered note.</footnote></doctag>',
    }
    entries = []
    for name, raw in cases.items():
        document = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], None))
        markdown = document.export_to_markdown(included_content_layers=set(ContentLayer)).rstrip('\n') + '\n'
        (args.output / (name + '.doctags')).write_text(raw, encoding='utf-8')
        (args.output / (name + '.md')).write_text(markdown, encoding='utf-8')
        entries.append(f'(:input {lisp_string(name + ".doctags")} :markdown {lisp_string(name + ".md")} :texts ({" ".join(lisp_string(i.text) for i in document.texts)}))')
        print(name, repr(markdown), [(i.label.value, i.text) for i in document.texts])
    (args.output / 'cases.sexp').write_text('(\n' + '\n'.join(entries) + '\n)\n', encoding='utf-8')
    manifest = {
        'docling-core': '2.97.0',
        'source': 'https://github.com/docling-project/docling-core/tree/v2.97.0',
        'license': 'MIT',
        'policy': 'Authored examples; all content layers; one terminal Markdown newline. No inferred footnote links or furniture removal.',
        'source_sha256': {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                          for cls in (DoclingDocument, MarkdownDocSerializer)},
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
