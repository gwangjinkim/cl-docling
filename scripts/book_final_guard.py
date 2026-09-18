"""Unlock fixed final-page rendering only after validation-only selection is sealed."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-final-workflow.lock.json'
SPLIT_REPORT = ROOT / 'tests/fixtures/book-splits/report.json'


def safe_relative(name):
    if not isinstance(name, str) or not name or len(name) > 240:
        raise ValueError('Expected a bounded relative artifact name')
    path = Path(name)
    if path.is_absolute() or any(part in ('', '.', '..') for part in path.parts):
        raise ValueError('Artifact path must stay beneath its declared root')
    return path


def verified_file(root, name, wanted):
    relative = safe_relative(name)
    path = root / relative
    if root.is_symlink() or path.is_symlink() or not path.is_file() or digest(path) != wanted:
        raise ValueError(f'Changed, missing or linked sealed artifact: {name}')
    return path


def validate_selection_seal(seal, selection_protocol, selection_root, selected_root):
    policy = json.loads(POLICY.read_text())
    required = {'schema_version', 'status', 'selection_criterion', 'final_content_opened',
                'workflow_sha256', 'selection_protocol_sha256', 'selection_report',
                'selection_report_sha256', 'selected_id', 'selected_kind', 'selected_files',
                'final_cases', 'development_evidence', 'source_revision'}
    if not isinstance(seal, dict) or set(seal) != required or seal['schema_version'] != 1:
        raise ValueError('Incomplete or extended selection seal schema')
    if (seal['status'] != 'selection-sealed' or seal['selection_criterion'] != 'validation-only'
            or seal['final_content_opened'] is not False):
        raise ValueError('Selection must be final, validation-only and pre-final')
    if seal['workflow_sha256'] != digest(POLICY):
        raise ValueError('Selection seal targets another final workflow')
    if (selection_protocol.is_symlink() or not selection_protocol.is_file()
            or seal['selection_protocol_sha256'] != digest(selection_protocol)):
        raise ValueError('Changed or linked selection protocol')
    if not re.fullmatch(r'[0-9a-f]{40}', seal['source_revision']):
        raise ValueError('Selection source revision must be an exact commit ID')
    if not isinstance(seal['selected_id'], str) or not seal['selected_id']:
        raise ValueError('Missing selected candidate identity')
    expected_cases = [case['name'] for case in policy['cases']]
    if seal['final_cases'] != expected_cases:
        raise ValueError('Changed fixed final case inventory')
    if seal['development_evidence'] != policy['required_development_evidence']:
        raise ValueError('Selection seal does not bind the frozen development evidence')
    verify_files(ROOT, policy['required_development_evidence'])
    verified_file(selection_root, seal['selection_report'], seal['selection_report_sha256'])
    files = seal['selected_files']
    if not isinstance(files, dict) or seal['selected_kind'] not in ('base', 'adapter'):
        raise ValueError('Unsupported selected artifact kind')
    if (seal['selected_kind'] == 'base') != (files == {}):
        raise ValueError('Base fallback has no adapter payload; adapter selection must bind files')
    for name, wanted in files.items():
        if not re.fullmatch(r'[0-9a-f]{64}', wanted):
            raise ValueError('Invalid selected artifact digest')
        verified_file(selected_root, name, wanted)
    return dict(seal)


def verify_final_inputs(pdf):
    policy = json.loads(POLICY.read_text())
    split = json.loads(SPLIT_REPORT.read_text())
    if digest(pdf) != policy['pdf_sha256']:
        raise ValueError('Changed reserved final PDF')
    inventory = {case['row']: case for case in split['inventory']}
    for expected in policy['cases']:
        actual = inventory.get(expected['row'])
        if not actual or (actual['source'], actual['page'], actual['split']) != (
                policy['source'], expected['page'], 'final'):
            raise ValueError('Changed reserved final case metadata')
    if split['final_status'] != 'reserved-unreviewed':
        raise ValueError('Expected the original metadata-only unopened reservation')


def render(pdf, seal_path, selection_protocol, selection_root, selected_root, output):
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError('Selection seal must be a regular file')
    seal = json.loads(seal_path.read_text())
    validate_selection_seal(seal, selection_protocol, selection_root, selected_root)
    verify_final_inputs(pdf)
    policy = json.loads(POLICY.read_text())
    version = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True,
                             timeout=10, check=True)
    version_text = version.stdout + version.stderr
    if f"pdftoppm version {policy['poppler_version']}" not in version_text:
        raise ValueError('Use the frozen Poppler version')
    output.mkdir(parents=True, exist_ok=False)
    images = {}
    for case in policy['cases']:
        for dpi in policy['render_dpi']:
            name = f"{case['name']}-{dpi}.png"
            prefix = output / name.removesuffix('.png')
            subprocess.run(['pdftoppm', '-f', str(case['page']), '-l', str(case['page']),
                            '-singlefile', '-r', str(dpi), '-png', str(pdf.resolve()), str(prefix)],
                           cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
            path = output / name
            if not path.is_file():
                raise ValueError('Poppler did not create the expected final-page render')
            images[name] = digest(path)
    validate_selection_seal(seal, selection_protocol, selection_root, selected_root)
    verify_final_inputs(pdf)
    report = dict(schema_version=1, scope=policy['scope'], workflow_sha256=digest(POLICY),
                  selection_seal_sha256=digest(seal_path), selected_id=seal['selected_id'],
                  selected_kind=seal['selected_kind'], final_content_opened=True,
                  reference_complete=False, model_inference=False, training_ready=False,
                  pdf_sha256=digest(pdf), poppler=policy['poppler_version'], images=images)
    (output / 'render-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(final_content_opened=True, reference_complete=False,
                          model_inference=False, pages=len(policy['cases'])), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--selection-seal', type=Path, required=True)
    parser.add_argument('--selection-protocol', type=Path, required=True)
    parser.add_argument('--selection-root', type=Path, required=True)
    parser.add_argument('--selected-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-open-final', choices=['selection-is-sealed'], required=True)
    args = parser.parse_args()
    render(args.pdf, args.selection_seal, args.selection_protocol,
           args.selection_root, args.selected_root, args.output)


if __name__ == '__main__':
    main()
