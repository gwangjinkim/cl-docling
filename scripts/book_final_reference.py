"""Prepare complete post-selection final references; never run a model or select."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil

from book_baseline import project
from book_final_guard import POLICY as WORKFLOW, validate_selection_seal
from book_picture_export import export as export_picture, verify_asset
from dataset_geometry import convert_pixel_locations
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'references/book-final-review.json'


def validate_review(review):
    expected = [('final-008', 8, 90, 'picture-reference'),
                ('final-009', 9, 121, 'picture-reference'),
                ('final-010', 10, 40, 'text-reference')]
    if (review.get('schema_version') != 1 or review.get('reference_complete') is not True
            or review.get('training_approved') is not False or len(review.get('cases', [])) != 3
            or [(c.get('name'), c.get('row'), c.get('page'), c.get('decision'))
                for c in review['cases']] != expected):
        raise ValueError('Changed or incomplete fixed final review')
    for case in review['cases']:
        width, height = case.get('image_size', [None, None])
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError('Invalid reviewed final image size')
        elements = case.get('elements')
        if case['decision'] == 'picture-reference':
            box = case.get('picture_box', [])
            if (elements != [['picture', None]] or len(box) != 4
                    or any(type(value) is not int for value in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError('Invalid complete final picture reference')
        elif (not isinstance(elements, list) or len(elements) != 12
              or [item[0] for item in elements[:5]] !=
              ['page_header', 'page_header', 'text', 'section_header_level_1', 'title']
              or any(not isinstance(item, list) or len(item) != 2 or not item[1] for item in elements)):
            raise ValueError('Invalid complete final text reference')
    return True


def make_target(case):
    if case['decision'] == 'text-reference':
        allowed = {'text', 'page_header', 'page_footer', 'section_header_level_1', 'title'}
        pieces = ['<doctag>']
        for tag, text in case['elements']:
            if (tag not in allowed or not isinstance(text, str) or not text.strip()
                    or '<' in text or '>' in text or re.search(r'&(?:#\w+|\w+);', text)
                    or any(ord(character) < 32 for character in text)):
                raise ValueError('Unsafe or unsupported reviewed final text')
            pieces.append(f'<{tag}>{text}</{tag}>')
        return ''.join([*pieces, '</doctag>'])
    pixels = ''.join(f'<loc_{value}>' for value in case['picture_box'])
    raw = f'<doctag><picture>{pixels}</picture></doctag>'
    return convert_pixel_locations(raw, *case['image_size'], convention='top-left-pixels')


def independent_reference(raw, image_path, case):
    from PIL import Image
    from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
    from docling_core.types.doc.tokens import DocumentToken
    if importlib.metadata.version('docling-core') != '2.97.0':
        raise ValueError('Use pinned docling-core 2.97.0')
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    if list(image.size) != case['image_size']:
        raise ValueError('Changed reviewed final image dimensions')
    document = DoclingDocument.load_from_doctags(
        DocTagsDocument.from_doctags_and_image_pairs([raw], [image]))
    if case['decision'] == 'picture-reference':
        locations = DocumentToken.get_location(tuple(case['picture_box']), *image.size)
        if f'<picture>{locations}</picture>' not in raw or len(document.pictures) != 1 or document.texts:
            raise ValueError('Independent final picture structure/location differs')
        quantized = [int(value) for value in re.findall(r'<loc_(\d+)>', locations)]
        box = [value * dimension // 500
               for value, dimension in zip(quantized, [image.width, image.height] * 2, strict=True)]
        expected = image.crop(tuple(box))
        actual = document.pictures[0].image.pil_image.convert('RGB')
        if actual.size != expected.size or actual.tobytes() != expected.tobytes():
            raise ValueError('Independent final picture pixels differ')
        return dict(pictures=1, texts=0, quantized_location=quantized,
                    decoded_crop_box=box, picture_size=list(actual.size),
                    rgb_pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), pixels_exact=True)
    wanted = [(('section_header' if tag == 'section_header_level_1' else tag), text)
              for tag, text in case['elements']]
    actual = [(item.label.value, item.text) for item in document.texts]
    if actual != wanted or document.pictures:
        raise ValueError('Independent final text roles/content differ')
    return dict(pictures=0, texts=len(actual), text_and_roles_exact=True)


def prepare(render_root, seal_path, selection_protocol, selection_root, selected_root, output):
    review = json.loads(REVIEW.read_text())
    validate_review(review)
    seal = json.loads(seal_path.read_text())
    validate_selection_seal(seal, selection_protocol, selection_root, selected_root)
    if digest(seal_path) != review['selection_seal_sha256']:
        raise ValueError('Review targets another selection seal')
    render_report = json.loads((render_root / 'render-report.json').read_text())
    if (render_report.get('selection_seal_sha256') != digest(seal_path)
            or render_report.get('final_content_opened') is not True
            or render_report.get('reference_complete') is not False
            or render_report.get('model_inference') is not False):
        raise ValueError('Changed or invalid protected render report')
    verify_files(render_root, render_report['images'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'expected').mkdir()
    (output / 'reference-export').mkdir()
    cases = []
    for case in review['cases']:
        image = render_root / case['image']
        destination_image = output / f"{case['name']}.png"
        shutil.copyfile(image, destination_image)
        raw = make_target(case)
        target = output / f"{case['name']}.doctags"
        target.write_text(raw)
        independent = independent_reference(raw, destination_image, case)
        if case['decision'] == 'picture-reference':
            export_root = output / 'reference-export' / case['name']
            native = export_picture(target, destination_image, export_root)
            verification = verify_asset(destination_image, export_root / native['asset'],
                                        native['picture_boxes'][0], native['picture_size'])
            if not verification['pixels_exact']:
                raise ValueError('Native final reference asset differs')
            expected = dict(markdown=native['markdown'], signature=native['signature'],
                            picture_boxes=native['picture_boxes'], picture_iou_threshold=0.8)
        else:
            native = project([target], output / 'reference-export' / case['name'])[0]
            if native['markdown'] is None or native['diagnostics']:
                raise ValueError('Final text reference does not render strictly')
            expected = dict(markdown=native['markdown'], signature=native['signature'], picture_boxes=[])
        expected_path = output / 'expected' / f"{case['name']}.json"
        expected_path.write_text(json.dumps(expected, indent=2) + '\n')
        cases.append(dict(name=case['name'], row=case['row'], page=case['page'],
                          decision=case['decision'], image=destination_image.name,
                          target=target.name, expected=str(expected_path.relative_to(output)),
                          independent=independent))
    validate_selection_seal(seal, selection_protocol, selection_root, selected_root)
    verify_files(render_root, render_report['images'])
    manifest = dict(schema_version=1, scope=review['scope'], review_sha256=digest(REVIEW),
                    workflow_sha256=digest(WORKFLOW), selection_seal_sha256=digest(seal_path),
                    render_report_sha256=digest(render_root / 'render-report.json'),
                    reference_complete=True, training_ready=False, model_inference=False,
                    cases=cases, files={str(path.relative_to(output)): digest(path)
                                        for path in sorted(output.rglob('*')) if path.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared complete post-selection final references; no model inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('render-root', 'selection-seal', 'selection-protocol', 'selection-root',
                  'selected-root', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.render_root, args.selection_seal, args.selection_protocol,
            args.selection_root, args.selected_root, args.output)


if __name__ == '__main__':
    main()
