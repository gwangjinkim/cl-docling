"""Replay the frozen illustrated validation page through native asset export; no model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from book_score import normalized
from picture_score import score_picture
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-picture-export.lock.json'
PICTURE_REPORT = ROOT / 'tests/fixtures/book-picture/report.json'
BASELINE_REPORT = ROOT / 'tests/fixtures/book-baseline/report.json'


def parse_record(stdout):
    lines = [line for line in stdout.splitlines() if line.startswith('PICTURE\t')]
    if len(lines) != 1:
        raise ValueError('Expected exactly one native picture record')
    fields = lines[0].split('\t')
    if len(fields) != 6:
        raise ValueError('Malformed native picture record')
    _, location, classification, width, height, signature = fields
    try:
        location = [int(value) for value in location.split(',')]
        dimensions = [int(width), int(height)]
    except ValueError as error:
        raise ValueError('Non-integer native picture record') from error
    if len(location) != 4 or any(value < 0 or value > 499 for value in location):
        raise ValueError('Invalid native picture location')
    return dict(picture_boxes=[location], classification=None if classification == 'none' else classification,
                picture_size=dimensions, signature=signature.split(',') if signature else [])


def export(raw, image, destination):
    command = ['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
               'scripts/picture-export.lisp', str(raw.resolve()), str(image.resolve()),
               str(destination.resolve())]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=60, check=True)
    record = parse_record(completed.stdout)
    record['markdown'] = (destination / 'document.md').read_text()
    record['asset'] = 'assets/picture-1.png'
    return record


def verify_asset(source_path, asset_path, location, expected_size):
    from PIL import Image
    with Image.open(source_path) as opened:
        source = opened.convert('RGB')
    box = [value * dimension // 500
           for value, dimension in zip(location, [source.width, source.height] * 2, strict=True)]
    expected = source.crop(tuple(box))
    with Image.open(asset_path) as opened:
        actual = opened.convert('RGB')
    exact = actual.size == expected.size and actual.tobytes() == expected.tobytes()
    if list(actual.size) != expected_size or not exact:
        raise ValueError('Native picture asset differs from the exact source region')
    return dict(decoded_crop_box=box, picture_size=list(actual.size),
                rgb_pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), pixels_exact=True)


def verify_sources(picture, baseline):
    picture_report = json.loads(PICTURE_REPORT.read_text())
    baseline_report = json.loads(BASELINE_REPORT.read_text())
    verify_files(ROOT, picture_report['original_sha256'])
    verify_files(picture, {'report.json': digest(PICTURE_REPORT), **picture_report['retained_sha256']})
    verify_files(baseline, {'run.json': baseline_report['run_sha256'],
                            'scores/report.json': digest(BASELINE_REPORT),
                            **baseline_report['retained_sha256']})
    return picture_report, baseline_report


def replay(picture, baseline, output):
    policy = json.loads(POLICY.read_text())
    picture_report, baseline_report = verify_sources(picture, baseline)
    if (policy['case'] != 'validation-005' or policy['input_image_sha256'] != picture_report['retained_sha256']['validation-005.png']
            or policy.get('training_ready') is not False):
        raise ValueError('Changed frozen picture-export protocol')
    source_image = picture / 'validation-005.png'
    reference_raw = picture / 'validation-005.doctags'
    actual_root = baseline / 'native/base'
    actual_raw = actual_root / 'validation-005.doctags'
    actual_json = json.loads((actual_root / 'validation-005.json').read_text())
    if actual_json['raw'] != actual_raw.read_text() or actual_json['case'] != policy['case']:
        raise ValueError('Retained base JSON/DocTags mismatch')

    output.mkdir(parents=True, exist_ok=False)
    reference = export(reference_raw, source_image, output / 'reference')
    actual = export(actual_raw, source_image, output / 'base')
    if reference['picture_boxes'] != [policy['reference_location']] or reference['signature'] != policy['reference_signature']:
        raise ValueError('Native reference projection changed')
    if reference['classification'] is not None:
        raise ValueError('Reference must not invent a picture classification')
    for record, directory in ((reference, 'reference'), (actual, 'base')):
        record['asset_verification'] = verify_asset(
            source_image, output / directory / record['asset'], record['picture_boxes'][0], record['picture_size'])
        record['asset_pixels_verified'] = record['asset_verification']['pixels_exact']

    expected = dict(markdown=reference['markdown'], signature=reference['signature'],
                    picture_boxes=reference['picture_boxes'])
    observed = {**actual, 'stop_reason': actual_json['stop_reason'],
                'diagnostics': [], 'error': False}
    score = score_picture(expected, observed, policy['picture_iou_threshold'])
    if normalized(reference['markdown']).count('![Picture](assets/picture-1.png)') != 1:
        raise ValueError('Reference Markdown lost its stable source-backed picture link')
    verify_sources(picture, baseline)
    retained = {str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob('*')) if path.is_file()}
    implementation = ('scripts/picture-export.lisp', 'scripts/picture_score.py',
                      'scripts/book_picture_export.py', 'native/images.c',
                      'src/images.lisp', 'src/doctags.lisp', 'src/document.lisp')
    report = dict(schema_version=1, scope=policy['scope'], protocol=policy,
                  protocol_sha256=digest(POLICY), model_inference=False, training_ready=False,
                  final_content_opened=False, original_baseline_unchanged=True,
                  development_exposed=True, source_revision=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_evidence_sha256=dict(picture=digest(PICTURE_REPORT), baseline=digest(BASELINE_REPORT)),
                  input_image_sha256=digest(source_image), reference=reference, base=actual,
                  base_stop_reason=actual_json['stop_reason'], score=score,
                  retained_sha256=retained,
                  implementation_sha256={name: digest(ROOT / name) for name in implementation})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(score=score, reference_asset=reference['asset_verification'],
                          base_asset=actual['asset_verification'], final_content_opened=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('picture', 'baseline', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    replay(args.picture, args.baseline, args.output)


if __name__ == '__main__':
    main()
