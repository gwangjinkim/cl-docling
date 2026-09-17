"""Independent footer oracle; no Lisp output is imported or repaired."""
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
        'located': '<doctag><text>Body text.</text><page_footer><loc_10><loc_460><loc_490><loc_480>Research notes 日本語</page_footer><page_footer>17</page_footer></doctag>',
        'footer-only': '<doctag><page_footer>Build together.</page_footer></doctag>',
    }
    entries = []
    for name, raw in cases.items():
        document = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], None))
        markdown = document.export_to_markdown(included_content_layers=set(ContentLayer)).rstrip('\n') + '\n'
        (args.output / (name + '.doctags')).write_text(raw, encoding='utf-8')
        (args.output / (name + '.md')).write_text(markdown, encoding='utf-8')
        entries.append(f'(:input {lisp_string(name + ".doctags")} :markdown {lisp_string(name + ".md")} :texts ({" ".join(lisp_string(i.text) for i in document.texts)}))')
        print(name, repr(markdown), [(i.label.value, i.content_layer.value, i.text) for i in document.texts])
    (args.output / 'cases.sexp').write_text('(\n' + '\n'.join(entries) + '\n)\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    replay, input_hashes = [], {}
    for page in (27, 28):
        name = f'page-{page}'
        base = root / 'tests/fixtures/public-pdf/results/native'
        result = json.loads((base / (name + '.json')).read_text())
        raw = (base / (name + '.doctags')).read_text()
        assert raw == result['raw'] and result['stop_reason'] == 'eos'
        document = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], None))
        footers = [i.text for i in document.texts if i.label.value == 'page_footer']
        assert len(footers) == 2
        cells = ['(' + ' '.join(map(str, c[:4])) + ' ' + lisp_string(c[4]) + ')'
                 for c in result['tables'][0]['cells']]
        replay.append(f'(:page {page} :footers ({" ".join(map(lisp_string, footers))}) :cells ({" ".join(cells)}))')
        for suffix in ('json', 'doctags'):
            path = base / (name + '.' + suffix)
            input_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'replay.sexp').write_text('(\n' + '\n'.join(replay) + '\n)\n', encoding='utf-8')
    manifest = {
        'docling-core': '2.97.0',
        'source': 'https://github.com/docling-project/docling-core/tree/v2.97.0',
        'license': 'MIT',
        'policy': 'Explicitly include all content layers, including furniture; one terminal Markdown newline.',
        'replay_policy': 'Cell records copied unchanged from M6.9 JSON; footer texts independently parsed by docling-core.',
        'replay_source_sha256': input_hashes,
        'source_sha256': {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                          for cls in (DoclingDocument, MarkdownDocSerializer)},
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
