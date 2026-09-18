"""Prepare and run the frozen real-book LoRA pilot; validation selects, final stays sealed."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess

from book_baseline import project
from book_final_guard import validate_selection_seal
from book_gradients import run_child
from book_picture_export import export as export_picture, verify_asset
from book_score import score_case
from picture_score import score_picture
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-finetune.lock.json'
BASELINE_REPORT = ROOT / 'tests/fixtures/book-baseline/report.json'
PICTURE_REPORT = ROOT / 'tests/fixtures/book-picture/report.json'
PICTURE_EXPORT_REPORT = ROOT / 'tests/fixtures/book-picture-export/report.json'
FROZEN_MANIFEST = ROOT / 'tests/fixtures/book-finetune/manifest.json'
FINAL_WORKFLOW = ROOT / 'references/book-final-workflow.lock.json'


def candidate_summary(pages):
    expected = json.loads(POLICY.read_text())['validation_cases']
    if not isinstance(pages, dict) or set(pages) != set(expected):
        raise ValueError('Candidate must retain every fixed validation page exactly once')
    for name in expected:
        row = pages[name]
        if (not isinstance(row, dict) or type(row.get('accepted')) is not bool
                or type(row.get('strict_export')) is not bool
                or type(row.get('character_edits')) is not int or row['character_edits'] < 0
                or type(row.get('word_edits')) is not int or row['word_edits'] < 0):
            raise ValueError('Invalid or incomplete validation score')
    return dict(pages=len(expected), strict_exports=sum(pages[name]['strict_export'] for name in expected),
                accepted_pages=sum(pages[name]['accepted'] for name in expected),
                character_edits=sum(pages[name]['character_edits'] for name in expected),
                word_edits=sum(pages[name]['word_edits'] for name in expected))


def choose_candidate(reports):
    order = ['base', 'step-3', 'step-6']
    if not isinstance(reports, dict) or list(reports) != order:
        raise ValueError('Expected the exact ordered base/step-3/step-6 inventory')
    summaries = {name: candidate_summary(reports[name]) for name in order}
    selected = min(order, key=lambda name: (-summaries[name]['accepted_pages'],
                                             summaries[name]['character_edits'],
                                             summaries[name]['word_edits'], order.index(name)))
    return selected, summaries


def validate_training(training, policy):
    steps = training.get('steps') if isinstance(training, dict) else None
    if (training.get('device') != policy['device'] or training.get('dtype') != policy['dtype']
            or training.get('final_step') != len(policy['schedule'])
            or training.get('remaining_handles') != 0 or not isinstance(steps, list)
            or len(steps) != len(policy['schedule'])):
        raise ValueError('Incomplete training trajectory or native cleanup')
    handles = steps[0].get('handles') if steps else None
    for number, (name, row) in enumerate(zip(policy['schedule'], steps, strict=True), 1):
        expected = policy['training_cases'][name]
        if (row.get('step') != number or row.get('name') != name or row.get('tiles') != expected['tiles']
                or row.get('sequence_tokens') != expected['sequence_tokens']
                or type(row.get('loss')) not in (int, float) or not math.isfinite(row['loss']) or row['loss'] < 0
                or row.get('base_unchanged') is not True or row.get('adapter_changed') is not True
                or type(handles) is not int or handles <= 0 or row.get('handles') != handles):
            raise ValueError('Changed input, loss, base, adapter or handle trajectory')
        for field in ('seconds', 'mlx_peak_bytes', 'mlx_active_bytes', 'mlx_cache_bytes'):
            if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError('Invalid training resource measurement')
        if (row['mlx_active_bytes'] > row['mlx_peak_bytes']
                or row['mlx_peak_bytes'] > policy['max_stage_mlx_bytes']
                or row['mlx_cache_bytes'] > policy['max_cache_bytes']):
            raise ValueError('Training resource ceiling exceeded')
    return True


def validate_manifest(manifest):
    policy = json.loads(POLICY.read_text())
    expected_files = {f'{name}.{suffix}' for name in policy['training_cases'] for suffix in ('png', 'doctags')}
    expected_files |= {f'{name}.png' for name in policy['validation_cases']}
    expected_files |= {'validation-005.doctags', 'validation-006.expected.json',
                       'validation-007.expected.json'}
    expected_files |= {f'base/{name}.{suffix}' for name in policy['validation_cases']
                       for suffix in ('json', 'doctags')}
    expected_evidence = dict(baseline_inputs='6eef9e6fcb29b469e900856b388824be9979b12d60f3aee062915dbc882e7f62',
                             baseline=digest(BASELINE_REPORT), picture=digest(PICTURE_REPORT),
                             picture_export=digest(PICTURE_EXPORT_REPORT))
    if (not isinstance(manifest, dict) or manifest.get('protocol') != policy
            or manifest.get('protocol_sha256') != digest(POLICY)
            or manifest.get('training_cases') != list(policy['training_cases'])
            or manifest.get('validation_cases') != policy['validation_cases']
            or set(manifest.get('files', {})) != expected_files
            or manifest.get('source_evidence_sha256') != expected_evidence
            or manifest.get('final_content_opened') is not False):
        raise ValueError('Changed or incomplete frozen fine-tune manifest')
    return True


def prepare(baseline_inputs, picture, baseline_run, output):
    baseline_manifest = json.loads((baseline_inputs / 'manifest.json').read_text())
    baseline_report = json.loads(BASELINE_REPORT.read_text())
    picture_report = json.loads(PICTURE_REPORT.read_text())
    verify_files(baseline_inputs, baseline_manifest['files'])
    verify_files(picture, {'report.json': digest(PICTURE_REPORT), **picture_report['retained_sha256']})
    verify_files(baseline_run, {'run.json': baseline_report['run_sha256'], **baseline_report['retained_sha256']})
    policy = json.loads(POLICY.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'base').mkdir()
    files = []
    for name in policy['training_cases']:
        files.extend((f'{name}.png', f'{name}.doctags'))
    for name in policy['validation_cases']:
        files.append(f'{name}.png')
    files.extend(('validation-006.expected.json', 'validation-007.expected.json'))
    for name in files:
        shutil.copyfile(baseline_inputs / name, output / name)
    shutil.copyfile(picture / 'validation-005.doctags', output / 'validation-005.doctags')
    for name in policy['validation_cases']:
        for suffix in ('json', 'doctags'):
            relative = Path('base') / f'{name}.{suffix}'
            shutil.copyfile(baseline_run / 'native/base' / f'{name}.{suffix}', output / relative)
    manifest = dict(schema_version=1, protocol=policy, protocol_sha256=digest(POLICY),
                    training_cases=list(policy['training_cases']),
                    validation_cases=policy['validation_cases'], final_content_opened=False,
                    source_evidence_sha256=dict(baseline_inputs=digest(baseline_inputs / 'manifest.json'),
                                                baseline= digest(BASELINE_REPORT), picture=digest(PICTURE_REPORT),
                                                picture_export=digest(PICTURE_EXPORT_REPORT)),
                    files={str(path.relative_to(output)): digest(path)
                           for path in sorted(output.rglob('*')) if path.is_file()})
    validate_manifest(manifest)
    verify_files(baseline_inputs, baseline_manifest['files'])
    verify_files(picture, {'report.json': digest(PICTURE_REPORT), **picture_report['retained_sha256']})
    verify_files(baseline_run, {'run.json': baseline_report['run_sha256'], **baseline_report['retained_sha256']})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared three train + three complete validation cases; final content remains unopened.')


def load_output(directory, name, max_tokens):
    metadata, raw_path = directory / f'{name}.json', directory / f'{name}.doctags'
    if not metadata.is_file() or not raw_path.is_file():
        return None, 'missing-output'
    value = json.loads(metadata.read_text())
    raw = raw_path.read_text()
    if value.get('raw') != raw or not isinstance(value.get('tokens'), list) or len(value['tokens']) > max_tokens:
        raise ValueError('Inconsistent retained candidate output')
    if value.get('stop_reason') not in ('eos', 'length', 'unknown'):
        raise ValueError('Invalid candidate stop reason')
    return value, None


def score_candidate(inputs, directory, destination):
    policy = json.loads(POLICY.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    reports = {}
    picture_reference = json.loads(PICTURE_EXPORT_REPORT.read_text())['reference']
    name = 'validation-005'
    result, error = load_output(directory, name, policy['generation']['max_new_tokens'])
    picture_actual = None
    if result is not None and result['stop_reason'] == 'eos':
        try:
            record = export_picture(directory / f'{name}.doctags', inputs / f'{name}.png',
                                    destination / 'validation-005-picture')
            record['asset_verification'] = verify_asset(
                inputs / f'{name}.png', destination / 'validation-005-picture' / record['asset'],
                record['picture_boxes'][0], record['picture_size'])
            record['asset_pixels_verified'] = record['asset_verification']['pixels_exact']
            picture_actual = {**record, 'stop_reason': 'eos', 'diagnostics': [], 'error': False}
        except (OSError, ValueError, subprocess.SubprocessError) as exception:
            error = f'{type(exception).__name__}: {exception}'
    reports[name] = score_picture(picture_reference, picture_actual,
                                  policy['picture_iou_threshold'] if 'picture_iou_threshold' in policy
                                  else json.loads((ROOT / 'references/book-picture-export.lock.json').read_text())['picture_iou_threshold'])
    reports[name]['execution_error'] = error

    for name in ('validation-006', 'validation-007'):
        expected = json.loads((inputs / f'{name}.expected.json').read_text())
        result, error = load_output(directory, name, policy['generation']['max_new_tokens'])
        projected = None
        if result is not None:
            try:
                projected = project([directory / f'{name}.doctags'], destination / f'{name}-projection',
                                    [result['stop_reason']])[0]
                projected.update(stop_reason=result['stop_reason'], tokens=len(result['tokens']), error=False)
            except (OSError, ValueError, subprocess.SubprocessError) as exception:
                error = f'{type(exception).__name__}: {exception}'
        reports[name] = score_case(expected, projected)
        reports[name]['execution_error_detail'] = error
    candidate_summary(reports)
    (destination / 'scores.json').write_text(json.dumps(reports, indent=2) + '\n')
    return reports


def run(inputs, checkpoint, output):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    if not FROZEN_MANIFEST.is_file() or digest(inputs / 'manifest.json') != digest(FROZEN_MANIFEST):
        raise ValueError('Commit the exact prepared M5.10 manifest before training')
    verify_files(inputs, manifest['files'])
    policy = json.loads(POLICY.read_text())
    model_lock = ROOT / 'references/smoldocling.lock.json'
    model = json.loads(model_lock.read_text())
    model_files = {name: info['sha256'] for name, info in model['files'].items()}
    verify_files(checkpoint, model_files)
    engine = ROOT.parent / 'cl-transformer-blocks'
    libraries = {'TB_MLX_LIBRARY': engine / '.build/native/libtb_mlx.dylib',
                 'DOCLING_IMAGE_LIBRARY': ROOT / '.build/native/libdocling_images.dylib'}
    for library in libraries.values():
        if not library.is_file():
            raise ValueError('Build the pinned native libraries before the pilot')
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    implementations = ['scripts/book_finetune.py', 'scripts/book-finetune.lisp',
                       'scripts/experiment-common.lisp', 'scripts/picture_score.py',
                       'scripts/book_picture_export.py']
    provenance = dict(protocol=policy, protocol_sha256=digest(POLICY),
                      manifest_sha256=digest(inputs / 'manifest.json'), model_revision=model['revision'],
                      model_lock_sha256=digest(model_lock), source_revision=revision,
                      engine_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=engine,
                                                              text=True).strip(),
                      library_sha256={name: digest(path) for name, path in libraries.items()},
                      implementation_sha256={name: digest(ROOT / name) for name in implementations},
                      platform=platform.platform())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    env = {**os.environ, **{name: str(path) for name, path in libraries.items()},
           'DOCLING_ENGINE': str(engine), 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': 'gpu'}
    try:
        command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                   '--script', 'scripts/book-finetune.lisp', str(inputs.resolve()),
                   str((output / 'training').resolve())]
        process = run_child(command, env, output / 'training.log', policy['child_timeout_seconds'])
        if process['whole_process_peak_rss_bytes'] > policy['max_child_rss_bytes']:
            raise ValueError('Training process RSS ceiling exceeded')
        (output / 'training-process.json').write_text(json.dumps(process, indent=2) + '\n')
        training = json.loads((output / 'training/training.json').read_text())
        validate_training(training, policy)
        candidate_dirs = {'base': inputs / 'base',
                          'step-3': output / 'training/step-3',
                          'step-6': output / 'training/step-6'}
        reports = {name: score_candidate(inputs, directory, output / 'scores' / name)
                   for name, directory in candidate_dirs.items()}
        selected, summaries = choose_candidate(reports)
        selected_root = output / 'selected'
        if selected == 'base':
            selected_root.mkdir()
            selected_files = {}
        else:
            shutil.copytree(output / 'training' / f"adapter-{selected.removeprefix('step-')}", selected_root)
            selected_files = {str(path.relative_to(selected_root)): digest(path)
                              for path in sorted(selected_root.rglob('*')) if path.is_file()}
        selection = dict(schema_version=1, selection_criterion='validation-only', selected_id=selected,
                         selected_kind='base' if selected == 'base' else 'adapter', candidates=reports,
                         summaries=summaries, losses=training['steps'], process_memory=process, **provenance)
        (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        final_policy = json.loads(FINAL_WORKFLOW.read_text())
        seal = dict(schema_version=1, status='selection-sealed', selection_criterion='validation-only',
                    final_content_opened=False, workflow_sha256=digest(FINAL_WORKFLOW),
                    selection_protocol_sha256=digest(POLICY), selection_report='selection.json',
                    selection_report_sha256=digest(output / 'selection.json'), selected_id=selected,
                    selected_kind=selection['selected_kind'], selected_files=selected_files,
                    final_cases=[case['name'] for case in final_policy['cases']],
                    development_evidence=final_policy['required_development_evidence'],
                    source_revision=revision)
        (output / 'selection-seal.json').write_text(json.dumps(seal, indent=2) + '\n')
        validate_selection_seal(seal, POLICY, output, selected_root)
        report = dict(schema_version=1, training_ready=False, final_content_opened=False,
                      selection_sealed=True, selected_id=selected, selected_kind=selection['selected_kind'],
                      summaries=summaries, selection_sha256=digest(output / 'selection.json'),
                      seal_sha256=digest(output / 'selection-seal.json'), **provenance)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(selected=selected, summaries=summaries, final_content_opened=False), indent=2))
    except BaseException as exception:
        (output / 'failure.json').write_text(json.dumps(
            dict(type=type(exception).__name__, message=str(exception)), indent=2) + '\n')
        raise
    finally:
        verify_files(inputs, manifest['files'])
        verify_files(checkpoint, model_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    for field in ('baseline-inputs', 'picture', 'baseline-run', 'output'):
        prepare_parser.add_argument('--' + field, type=Path, required=True)
    run_parser = sub.add_parser('run')
    for field in ('inputs', 'checkpoint', 'output'):
        run_parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.baseline_inputs, args.picture, args.baseline_run, args.output)
    else:
        run(args.inputs, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
