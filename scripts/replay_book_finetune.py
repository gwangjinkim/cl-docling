"""Hash-bound replay of the completed M5.10 run; no model, update or final data."""
import argparse
import json
from pathlib import Path

from book_final_guard import validate_selection_seal
from book_finetune import (FINAL_WORKFLOW, FROZEN_MANIFEST, POLICY, ROOT, choose_candidate,
                           score_candidate, validate_manifest, validate_training)
from replay_dpbench import digest, verify_files


def replay(inputs, checkpoint, run, output):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    if digest(inputs / 'manifest.json') != digest(FROZEN_MANIFEST):
        raise ValueError('Changed frozen fine-tune manifest')
    verify_files(inputs, manifest['files'])
    policy = json.loads(POLICY.read_text())
    model_lock = ROOT / 'references/smoldocling.lock.json'
    model = json.loads(model_lock.read_text())
    model_files = {name: info['sha256'] for name, info in model['files'].items()}
    verify_files(checkpoint, model_files)
    provenance = json.loads((run / 'provenance.json').read_text())
    if (provenance['protocol'] != policy or provenance['protocol_sha256'] != digest(POLICY)
            or provenance['manifest_sha256'] != digest(FROZEN_MANIFEST)
            or provenance['model_lock_sha256'] != digest(model_lock)):
        raise ValueError('Changed run provenance')
    for name, wanted in provenance['implementation_sha256'].items():
        if digest(ROOT / name) != wanted:
            raise ValueError('Run implementation changed: ' + name)
    training = json.loads((run / 'training/training.json').read_text())
    validate_training(training, policy)
    selection_path, seal_path = run / 'selection.json', run / 'selection-seal.json'
    selection, seal = json.loads(selection_path.read_text()), json.loads(seal_path.read_text())
    if seal['selection_report_sha256'] != digest(selection_path):
        raise ValueError('Selection report changed after sealing')
    validate_selection_seal(seal, POLICY, run, run / 'selected')
    output.mkdir(parents=True, exist_ok=False)
    candidate_dirs = {'base': inputs / 'base', 'step-3': run / 'training/step-3',
                      'step-6': run / 'training/step-6'}
    reports = {name: score_candidate(inputs, directory, output / 'scores' / name)
               for name, directory in candidate_dirs.items()}
    selected, summaries = choose_candidate(reports)
    if (reports != selection['candidates'] or summaries != selection['summaries']
            or selected != selection['selected_id'] or selected != seal['selected_id']):
        raise ValueError('Retained outputs do not reproduce the sealed selection')
    original_report = json.loads((run / 'report.json').read_text())
    if (original_report['selected_id'] != selected or original_report['summaries'] != summaries
            or original_report['selection_sha256'] != digest(selection_path)
            or original_report['seal_sha256'] != digest(seal_path)
            or original_report['final_content_opened'] is not False):
        raise ValueError('Run report differs from sealed selection')
    verify_files(inputs, manifest['files'])
    verify_files(checkpoint, model_files)
    report = dict(schema_version=1, scope=policy['scope'], replay_only=True,
                  model_inference=False, updates=0, final_content_opened=False,
                  selected_id=selected, selected_kind=selection['selected_kind'], summaries=summaries,
                  protocol_sha256=digest(POLICY), manifest_sha256=digest(FROZEN_MANIFEST),
                  selection_sha256=digest(selection_path), seal_sha256=digest(seal_path),
                  source_revision=provenance['source_revision'],
                  retained_sha256={str(path.relative_to(run)): digest(path)
                                   for path in sorted(run.rglob('*')) if path.is_file()},
                  replay_sha256={str(path.relative_to(output)): digest(path)
                                 for path in sorted(output.rglob('*')) if path.is_file()},
                  implementation_sha256={'scripts/replay_book_finetune.py': digest(Path(__file__))})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: retained M5.10 outputs reproduce the sealed base selection; final remains unopened.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('inputs', 'checkpoint', 'run', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    replay(args.inputs, args.checkpoint, args.run, args.output)


if __name__ == '__main__':
    main()
