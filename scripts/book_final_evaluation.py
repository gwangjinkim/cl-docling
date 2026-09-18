"""Run, score, or replay the sealed one-way M5.11 final evaluation."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

from book_baseline import project
from book_final_guard import validate_selection_seal
from book_gradients import run_child
from book_picture_export import export as export_picture, verify_asset
from book_score import score_case
from picture_score import score_picture
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-final-evaluation.lock.json'
FROZEN_MANIFEST = ROOT / 'tests/fixtures/book-final/manifest.json'
SELECTION_PROTOCOL = ROOT / 'references/book-finetune.lock.json'
MODEL_LOCK = ROOT / 'references/smoldocling.lock.json'


def validate_policy(policy):
    expected = ['final-008', 'final-009', 'final-010']
    regressions = ['validation-005', 'validation-006', 'validation-007']
    if (policy.get('schema_version') != 1 or policy.get('final_cases') != expected
            or policy.get('regression_cases') != regressions
            or policy.get('selected_id') != 'base' or policy.get('selected_kind') != 'base'
            or policy.get('selection_seal_sha256') !=
            '0e46c464426726119b291cde844c8b342b8c33757e301b2ac2481b0674e1bf80'
            or policy.get('final_manifest_sha256') !=
            '07c62596e69ae63cc6e834e29327cf2c403c48e522f9ced9c41c0db0f17d36df'
            or policy.get('generation') != {'task': 'Convert this page to docling.',
                                             'max_new_tokens': 2048}
            or policy.get('native_runs') != 1 or policy.get('parity_cases') != expected
            or policy.get('picture_iou_threshold') != 0.8
            or policy.get('training_ready') is not False):
        raise ValueError('Changed or incomplete final-evaluation protocol')
    return True


def validate_final_manifest(manifest, policy):
    expected = policy['final_cases']
    cases = manifest.get('cases', [])
    wanted_files = {f'{name}.{suffix}' for name in expected for suffix in ('png', 'doctags')}
    wanted_files |= {f'expected/{name}.json' for name in expected}
    if (manifest.get('schema_version') != 1 or manifest.get('reference_complete') is not True
            or manifest.get('training_ready') is not False or manifest.get('model_inference') is not False
            or manifest.get('selection_seal_sha256') != policy['selection_seal_sha256']
            or [case.get('name') for case in cases] != expected
            or [case.get('decision') for case in cases] !=
            ['picture-reference', 'picture-reference', 'text-reference']
            or not wanted_files.issubset(manifest.get('files', {}))):
        raise ValueError('Changed or incomplete frozen final manifest')
    return True


def summarize(names, rows):
    if set(rows) != set(names) or len(names) != len(set(names)):
        raise ValueError('Missing, duplicate or extra score')
    values = [rows[name] for name in names]
    characters = sum(row['reference_characters'] for row in values)
    words = sum(row['reference_words'] for row in values)
    character_edits = sum(row['character_edits'] for row in values)
    word_edits = sum(row['word_edits'] for row in values)
    return dict(pages=len(values), parser_valid_pages=sum(row['strict_export'] for row in values),
                exact_recognition_pages=sum(row['accepted'] for row in values),
                character_edits=character_edits, reference_characters=characters,
                markdown_cer=character_edits / characters, word_edits=word_edits,
                reference_words=words, markdown_wer=word_edits / words)


def verify_inputs(inputs, selection_root, selected_root):
    policy = json.loads(POLICY.read_text())
    validate_policy(policy)
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_final_manifest(manifest, policy)
    frozen = json.loads(FROZEN_MANIFEST.read_text())
    if (digest(inputs / 'manifest.json') != policy['final_manifest_sha256']
            or manifest != frozen):
        raise ValueError('Use the exact hash-bound and semantically frozen final manifest')
    verify_files(inputs, manifest['files'])
    seal_path = selection_root / 'selection-seal.json'
    if digest(seal_path) != policy['selection_seal_sha256']:
        raise ValueError('Final protocol targets another selection seal')
    validate_selection_seal(json.loads(seal_path.read_text()), SELECTION_PROTOCOL,
                            selection_root, selected_root)
    return policy, manifest


def load_output(directory, name, maximum):
    metadata, raw_path = directory / f'{name}.json', directory / f'{name}.doctags'
    error_path = directory / f'{name}.error.json'
    if error_path.is_file() or not metadata.is_file() or not raw_path.is_file():
        return None, ('recorded-execution-error' if error_path.is_file() else 'missing-output')
    value = json.loads(metadata.read_text())
    if (value.get('raw') != raw_path.read_text() or not isinstance(value.get('tokens'), list)
            or len(value['tokens']) > maximum or value.get('stop_reason') not in ('eos', 'length', 'unknown')):
        raise ValueError('Inconsistent retained final output')
    return value, None


def score_final(inputs, native, destination, policy, manifest):
    destination.mkdir(parents=True, exist_ok=False)
    scores, identities = {}, {}
    for case in manifest['cases']:
        name = case['name']
        result, error = load_output(native / 'base', name, policy['generation']['max_new_tokens'])
        expected = json.loads((inputs / case['expected']).read_text())
        actual = None
        if result is not None:
            try:
                if case['decision'] == 'picture-reference' and result['stop_reason'] == 'eos':
                    exported = export_picture(native / 'base' / f'{name}.doctags', inputs / case['image'],
                                              destination / f'{name}-picture')
                    verification = verify_asset(inputs / case['image'],
                                                destination / f'{name}-picture' / exported['asset'],
                                                exported['picture_boxes'][0], exported['picture_size'])
                    actual = {**exported, 'asset_pixels_verified': verification['pixels_exact'],
                              'asset_verification': verification, 'stop_reason': 'eos',
                              'diagnostics': [], 'error': False}
                elif case['decision'] == 'text-reference':
                    actual = project([native / 'base' / f'{name}.doctags'],
                                     destination / f'{name}-projection', [result['stop_reason']])[0]
                    actual.update(stop_reason=result['stop_reason'], tokens=len(result['tokens']), error=False)
            except (OSError, ValueError, subprocess.SubprocessError) as exception:
                error = f'{type(exception).__name__}: {exception}'
        if case['decision'] == 'picture-reference':
            score = score_picture(expected, actual, policy['picture_iou_threshold'])
            score['execution_error_detail'] = error
        else:
            score = score_case(expected, actual)
            score['execution_error_detail'] = error
        scores[name] = score
        identities[name] = None if result is None else dict(
            tokens_sha256=digest(native / 'base' / f'{name}.json'),
            raw_sha256=digest(native / 'base' / f'{name}.doctags'), token_count=len(result['tokens']))
    return scores, identities


def write_report(inputs, run, selection_root, selected_root, destination):
    policy, manifest = verify_inputs(inputs, selection_root, selected_root)
    provenance = json.loads((run / 'provenance.json').read_text())
    if provenance['manifest_sha256'] != digest(inputs / 'manifest.json'):
        raise ValueError('Final run used another manifest')
    final_scores, identities = score_final(inputs, run / 'native', destination / 'exports', policy, manifest)
    selection = json.loads((selection_root / 'selection.json').read_text())
    regressions = selection['candidates']['base']
    if set(regressions) != set(policy['regression_cases']):
        raise ValueError('Incomplete regression evidence')
    final_summary = summarize(policy['final_cases'], final_scores)
    regression_summary = summarize(policy['regression_cases'], regressions)
    completion = json.loads((run / 'native/completion.json').read_text())
    if completion != {'remaining_handles': 0, 'device': 'gpu', 'updates': 0,
                       'native_runs': 1, 'selected_kind': 'base'}:
        raise ValueError('Incomplete native final cleanup')
    report = dict(schema_version=1, selection_immutable=True, selected_id='base', selected_kind='base',
                  base_selected_identity=True, native_inference_runs=1, final_content_opened=True,
                  training_updates=0, protocol_sha256=digest(POLICY),
                  final_manifest_sha256=digest(inputs / 'manifest.json'),
                  selection_seal_sha256=digest(selection_root / 'selection-seal.json'),
                  provenance=provenance, process_memory=json.loads((run / 'native-process.json').read_text()),
                  completion=completion,
                  final={name: {'base': row, 'selected': row, 'same_native_output': True,
                                'identity': identities[name]} for name, row in final_scores.items()},
                  regression={name: {'base': row, 'selected': row, 'same_retained_output': True}
                              for name, row in regressions.items()},
                  summary={'final': {'base': final_summary, 'selected': final_summary},
                           'regression': {'base': regression_summary, 'selected': regression_summary}},
                  retained_sha256={str(path.relative_to(run)): digest(path)
                                   for path in sorted((run / 'native').rglob('*')) if path.is_file()},
                  scorer_sha256={name: digest(ROOT / name) for name in
                    ('scripts/book_final_evaluation.py', 'scripts/book-final-evaluation.lisp',
                     'scripts/book-project.lisp', 'scripts/book_picture_export.py',
                     'scripts/picture_score.py', 'scripts/book_score.py')})
    verify_inputs(inputs, selection_root, selected_root)
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


def run(inputs, checkpoint, selection_root, selected_root, output):
    policy, manifest = verify_inputs(inputs, selection_root, selected_root)
    model = json.loads(MODEL_LOCK.read_text())
    model_files = {name: info['sha256'] for name, info in model['files'].items()}
    verify_files(checkpoint, model_files)
    engine = ROOT.parent / 'cl-transformer-blocks'
    libraries = {'TB_MLX_LIBRARY': engine / '.build/native/libtb_mlx.dylib',
                 'DOCLING_IMAGE_LIBRARY': ROOT / '.build/native/libdocling_images.dylib'}
    if any(not path.is_file() for path in libraries.values()):
        raise ValueError('Build the pinned native libraries before final evaluation')
    output.mkdir(parents=True, exist_ok=False)
    implementation = ('scripts/book_final_evaluation.py', 'scripts/book-final-evaluation.lisp',
                      'scripts/experiment-common.lisp')
    provenance = dict(protocol_sha256=digest(POLICY), manifest_sha256=digest(inputs / 'manifest.json'),
                      selection_seal_sha256=digest(selection_root / 'selection-seal.json'),
                      selected_id='base', selected_kind='base', model_revision=model['revision'],
                      model_lock_sha256=digest(MODEL_LOCK), source_revision=subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      engine_revision=subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=engine, text=True).strip(),
                      library_sha256={name: digest(path) for name, path in libraries.items()},
                      implementation_sha256={name: digest(ROOT / name) for name in implementation},
                      platform=platform.platform())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    env = {**os.environ, **{name: str(path) for name, path in libraries.items()},
           'DOCLING_ENGINE': str(engine), 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': 'gpu',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
               '--script', 'scripts/book-final-evaluation.lisp', str(inputs.resolve()),
               str((output / 'native').resolve())]
    process = run_child(command, env, output / 'native.log', policy['child_timeout_seconds'])
    (output / 'native-process.json').write_text(json.dumps(process, indent=2) + '\n')
    if process['whole_process_peak_rss_bytes'] > policy['max_child_rss_bytes']:
        raise ValueError('Final process RSS ceiling exceeded')
    verify_files(checkpoint, model_files)
    write_report(inputs, output, selection_root, selected_root, output / 'scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command, fields in [('run', ('inputs', 'checkpoint', 'selection-root', 'selected-root', 'output')),
                            ('replay', ('inputs', 'run', 'selection-root', 'selected-root', 'output'))]:
        item = sub.add_parser(command)
        for field in fields:
            item.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.inputs.resolve(), args.checkpoint.resolve(), args.selection_root.resolve(),
            args.selected_root.resolve(), args.output.resolve())
    else:
        write_report(args.inputs.resolve(), args.run.resolve(), args.selection_root.resolve(),
                     args.selected_root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
