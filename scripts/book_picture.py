"""Prepare an evaluation-only illustrated-page reference; no model or final data."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import shutil

from book_pilot import make_target
from dataset_geometry import convert_pixel_locations
from dataset_audit import ROOT, native_probe
from replay_dpbench import digest, verify_files

REVIEW = ROOT / 'references/book-picture-review.json'


def make_reference(review):
    if (review['split'] != 'validation' or review['row'] != 5 or review['page'] != 10 or
            review['source'] != 'inventorsmechani00peck' or review['training_approved'] is not False):
        raise ValueError('Only the fixed evaluation-only illustrated page is allowed')
    elements = review['elements']
    if [e[0] for e in elements] != ['page_header', 'section_header_level_1', 'picture', *(['text']*5)]:
        raise ValueError('Missing/reordered picture or complete text inventory')
    width, height = review['image_size']
    left, top, right, bottom = review['picture_box']
    if (any(type(v) is not int for v in (width, height, left, top, right, bottom)) or
            not 0 <= left < right <= width or not 0 <= top < bottom <= height):
        raise ValueError('Invalid picture rectangle')
    pieces = []
    for tag, value in elements:
        if tag == 'picture':
            if value is not None:
                raise ValueError('No invented caption or classification')
            locations = ''.join(f'<loc_{v}>' for v in review['picture_box'])
            pieces.append(convert_pixel_locations(f'<picture>{locations}</picture>', width, height,
                                                   convention=review['coordinates']))
        else:
            # Reuse the text safety contract without widening training eligibility.
            pieces.append(make_target([[tag, value]])[len('<doctag>'):-len('</doctag>')])
    return '<doctag>' + ''.join(pieces) + '</doctag>'


def independent_picture(raw, image_path, review):
    from PIL import Image
    from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
    from docling_core.types.doc.tokens import DocumentToken
    if importlib.metadata.version('docling-core') != '2.97.0':
        raise ValueError('Use pinned docling-core 2.97.0')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    if list(image.size) != review['image_size']:
        raise ValueError('Changed reviewed image dimensions')
    locations = DocumentToken.get_location(tuple(review['picture_box']), *image.size)
    if '<picture>' + locations + '</picture>' not in raw:
        raise ValueError('Independent location quantization differs')
    document = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], [image]))
    expected_text = [(('section_header' if tag == 'section_header_level_1' else tag), text)
                     for tag, text in review['elements'] if tag != 'picture']
    if [(item.label.value, item.text) for item in document.texts] != expected_text or len(document.pictures) != 1:
        raise ValueError('Independent parser lost text, roles or the picture')
    picture = document.pictures[0]
    if picture.captions:
        raise ValueError('Unexpected caption')
    quantized = [int(v) for v in re.findall(r'<loc_(\d+)>', locations)]
    crop_box = [int(v/500*d) for v, d in zip(quantized, [image.width, image.height]*2, strict=True)]
    expected = image.crop(tuple(crop_box))
    actual = picture.image.pil_image.convert('RGB')
    if actual.size != expected.size or actual.tobytes() != expected.tobytes():
        raise ValueError('Independent picture pixels differ from original page region')
    return dict(docling_core='2.97.0', pillow=importlib.metadata.version('pillow'),
                source='https://github.com/docling-project/docling-core/tree/v2.97.0', license='MIT',
                source_sha256={cls.__name__: digest(Path(inspect.getfile(cls))) for cls in (DoclingDocument, DocumentToken)},
                picture_count=1, text_elements=len(document.texts), text_and_roles_exact=True,
                quantized_location=quantized, decoded_crop_box=crop_box, picture_size=list(actual.size),
                rgb_pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), picture_pixels_exact=True,
                policy='Crop exists only in reference analysis; full PNG remains unchanged input. Manual padded bounds are not precision geometry gold; no caption/signature OCR inferred.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('validation', 'pdf', 'review-render', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(REVIEW.read_text())
    split_path = ROOT / 'tests/fixtures/book-splits/report.json'
    split = json.loads(split_path.read_text())
    case = next(c for c in split['validation'] if c['row'] == 5)
    if case['image_sha256'] != review['image_sha256'] or case['page'] != review['page']:
        raise ValueError('Changed frozen validation identity')
    original_files = ['references/book-validation-review.json', 'references/book-splits.lock.json',
                      'references/book-baseline.lock.json', 'tests/fixtures/book-splits/report.json',
                      'tests/fixtures/book-baseline/manifest.json', 'tests/fixtures/book-baseline/report.json']
    original = {name: digest(ROOT / name) for name in original_files}
    def verify():
        verify_files(args.validation, {'report.json': digest(split_path), case['image']: review['image_sha256']})
        if digest(args.pdf) != review['pdf_sha256'] or digest(args.review_render) != review['review_render_sha256']:
            raise ValueError('Changed original PDF or visually reviewed second-pass render')
        verify_files(ROOT, original)
    verify()
    raw = make_reference(review)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.validation / case['image'], args.output / case['image'])
    target = args.output / 'validation-005.doctags'
    target.write_text(raw, encoding='utf-8')
    oracle = independent_picture(raw, args.output / case['image'], review)
    native = native_probe([target])[0]
    if native != dict(markdown_renderable=False, diagnostic_codes=['unsupported-tag']):
        raise ValueError('Native picture support changed; requalify explicitly, do not silently rescore')
    verify()
    report = dict(schema_version=1, scope=review['scope'], training_ready=False,
                  native_scoring_ready=False, supplemental_content_reviewed=True,
                  original_baseline_unchanged=True, final_content_opened=False,
                  row=5, page=10, split='validation', review_sha256=digest(REVIEW),
                  pdf_sha256=review['pdf_sha256'], review_render_sha256=review['review_render_sha256'],
                  source_report_sha256=digest(split_path), original_sha256=original,
                  implementation_sha256={name: digest(ROOT / name) for name in ('scripts/book_picture.py', 'scripts/dataset_geometry.py')},
                  oracle=oracle, native=native,
                  retained_sha256={p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(oracle=oracle, native=native, training_ready=False), indent=2))


if __name__ == '__main__':
    main()
