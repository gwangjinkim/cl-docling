"""Prepare frozen development-only inputs, run one bounded baseline, or replay it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from replay_dpbench import digest, verify_files
from book_score import score_case, summarize

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-baseline.lock.json'


def project(files, output, stop_reasons=None):
    stops = ['eos'] * len(files) if stop_reasons is None else stop_reasons
    if len(stops) != len(files) or any(s not in ('eos', 'length', 'unknown') for s in stops):
        raise ValueError('Invalid projection stop reasons')
    output.mkdir(parents=True, exist_ok=False)
    command = ['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
               'scripts/book-project.lisp', str(output.resolve()),
               *[arg for stop, path in zip(stops, files, strict=True) for arg in (stop, str(path.resolve()))]]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    records = {}
    for line in completed.stdout.splitlines():
        if not line.startswith('BOOK\t'):
            continue
        _, ordinal, rendered, signature, diagnostics = line.split('\t')
        index = int(ordinal)
        if index in records or rendered not in ('0', '1') or not 0 <= index < len(files):
            raise ValueError('Invalid parser projection')
        path = output / f'{index}.md'
        if path.is_file() != (rendered == '1'):
            raise ValueError('Projection artifact mismatch')
        records[index] = dict(markdown=path.read_text() if path.is_file() else None,
                              signature=signature.split(',') if signature else [],
                              diagnostics=diagnostics.split(',') if diagnostics else [])
    if set(records) != set(range(len(files))):
        raise ValueError('Incomplete parser projection')
    return [records[i] for i in range(len(files))]


def validate_manifest(manifest):
    policy = json.loads(POLICY.read_text())
    cases = manifest['cases']
    if (manifest['protocol'] != policy or manifest['protocol_sha256'] != digest(POLICY) or
            [c['name'] for c in cases] != policy['names'] or [c['split'] for c in cases] != policy['splits']):
        raise ValueError('Changed frozen development protocol/selection')
    for case in cases:
        reviewed = case['name'] not in policy['unscorable']
        if case['reference_available'] is not reviewed or case['expected'] != (case['name'] + '.expected.json' if reviewed else None):
            raise ValueError('Changed reference coverage')


def prepare(train, validation, output):
    frozen_train = ROOT / 'tests/fixtures/book-pilot/report.json'
    frozen_validation = ROOT / 'tests/fixtures/book-splits/report.json'
    inputs = []
    files_by_root = {}
    for folder, frozen, key, split in [(train, frozen_train, 'inventory', 'train'),
                                        (validation, frozen_validation, 'validation', 'validation')]:
        report = json.loads(frozen.read_text())
        files = {'report.json': digest(frozen)}
        for case in report[key]:
            if split == 'train' and case['decision'] != 'approved':
                continue
            for field in ('image', 'target'):
                if field in case:
                    files[case[field]] = case[field + '_sha256']
            inputs.append((folder, case, split))
        verify_files(folder, files)
        files_by_root[folder] = files
    output.mkdir(parents=True, exist_ok=False)
    cases, targets = [], []
    for folder, case, split in inputs:
        name = Path(case['image']).stem
        shutil.copyfile(folder / case['image'], output / case['image'])
        if 'target' in case:
            shutil.copyfile(folder / case['target'], output / case['target'])
            targets.append(output / case['target'])
        cases.append(dict(name=name, split=split, id=case['id'], source_id=case['source_id'],
                          image_sha256=case['image_sha256'], reference_available='target' in case,
                          expected=name + '.expected.json' if 'target' in case else None))
    references = iter(project(targets, output / 'reference-projection'))
    for case in cases:
        if case['reference_available']:
            reference = next(references)
            if reference['markdown'] is None or reference['diagnostics']:
                raise ValueError('Reviewed target no longer renders')
            (output / case['expected']).write_text(json.dumps(reference, indent=2) + '\n')
    manifest = dict(schema_version=1, protocol=json.loads(POLICY.read_text()), protocol_sha256=digest(POLICY),
                    source_reports=dict(train=digest(frozen_train), validation=digest(frozen_validation)),
                    cases=cases, files={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    validate_manifest(manifest)
    for folder, files in files_by_root.items():
        verify_files(folder, files)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared six development pages, five reviewed references, one unscorable illustration; no final content.')


def score_run(inputs, run, destination):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    verify_files(inputs, manifest['files'])
    provenance = json.loads((run / 'run.json').read_text())
    if provenance['manifest_sha256'] != digest(inputs / 'manifest.json'):
        raise ValueError('Run input manifest changed')
    destination.mkdir(parents=True, exist_ok=False)
    outputs, paths, names = {}, [], []
    for case in manifest['cases']:
        name = case['name']
        base = run / 'native/base'
        path, raw, error = (base / (name + suffix) for suffix in ('.json', '.doctags', '.error.json'))
        if error.is_file() or not path.is_file() or not raw.is_file():
            outputs[name] = dict(error='recorded-execution-error' if error.is_file() else 'missing-output')
            continue
        result = json.loads(path.read_text())
        if result['raw'] != raw.read_text() or len(result['tokens']) > manifest['protocol']['generation']['max_new_tokens']:
            raise ValueError('Inconsistent raw/token artifacts')
        outputs[name] = result
        paths.append(raw)
        names.append(name)
    projections = project(paths, destination / 'projection', [outputs[n]['stop_reason'] for n in names]) if paths else []
    for name, projected in zip(names, projections, strict=True):
        native = outputs[name]
        md = run / 'native/base' / (name + '.md')
        if (native['markdown_written'] != (projected['markdown'] is not None) or
                (md.read_text() if md.is_file() else None) != projected['markdown'] or
                [d['code'] for d in native['diagnostics']] != projected['diagnostics']):
            raise ValueError('Native/projection parser artifacts differ')
        outputs[name] = {**projected, 'stop_reason': native['stop_reason'], 'tokens': len(native['tokens']), 'error': False}
    reports = {}
    for case in manifest['cases']:
        reference = json.loads((inputs / case['expected']).read_text()) if case['reference_available'] else None
        reports[case['name']] = score_case(reference, outputs[case['name']])
    completion = run / 'native/completion.json'
    report = dict(schema_version=1, training_ready=False, protocol=manifest['protocol'],
                  manifest_sha256=digest(inputs / 'manifest.json'), run_sha256=digest(run / 'run.json'),
                  source_revision=provenance['source_revision'], exit_code=provenance['exit_code'],
                  completion=json.loads(completion.read_text()) if completion.is_file() else None,
                  cases=reports, summary=summarize(manifest['cases'], reports),
                  retained_sha256={str(p.relative_to(run)): digest(p) for p in sorted((run / 'native').rglob('*')) if p.is_file()},
                  scorer_sha256={p: digest(ROOT / p) for p in ('scripts/book_score.py', 'scripts/book-project.lisp', 'scripts/book_baseline.py')})
    verify_files(inputs, manifest['files'])
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


def run_baseline(inputs, model, output):
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    # Require a committed, exact manifest before any model run.
    if digest(inputs / 'manifest.json') != digest(ROOT / 'tests/fixtures/book-baseline/manifest.json'):
        raise ValueError('Seal the exact prepared manifest first')
    verify_files(inputs, manifest['files'])
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    model_files = {n: info['sha256'] for n, info in lock['files'].items()}
    verify_files(model, model_files)
    output.mkdir(parents=True, exist_ok=False)
    engine = ROOT.parent / 'cl-transformer-blocks'
    libraries = {'TB_MLX_LIBRARY': engine / '.build/native/libtb_mlx.dylib',
                 'DOCLING_IMAGE_LIBRARY': ROOT / '.build/native/libdocling_images.dylib'}
    env = {**os.environ, **{k: str(p) for k, p in libraries.items()},
           'DOCLING_MODEL': str(model.resolve()), 'DOCLING_ENGINE': str(engine), 'TB_DEVICE': 'gpu'}
    command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
               '--script', 'scripts/book-baseline.lisp', str(inputs.resolve()), str((output / 'native').resolve())]
    record = dict(manifest_sha256=digest(inputs / 'manifest.json'), model_revision=lock['revision'],
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  engine_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=engine, text=True).strip(),
                  libraries_sha256={k: digest(v) for k, v in libraries.items()},
                  implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                    [ROOT / 'scripts/book-baseline.lisp', ROOT / 'scripts/experiment-common.lisp', *sorted((ROOT / 'src').glob('*.lisp'))]},
                  started=datetime.now(timezone.utc).isoformat())
    with (output / 'native.log').open('x') as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=manifest['protocol']['child_timeout_seconds'])
            record['exit_code'] = completed.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=None, timeout=True)
    record['finished'] = datetime.now(timezone.utc).isoformat()
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    verify_files(model, model_files)
    score_run(inputs, output, output / 'scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command, fields in [('prepare', ('train', 'validation', 'output')),
                             ('run', ('inputs', 'model', 'output')), ('replay', ('inputs', 'run', 'output'))]:
        p = sub.add_parser(command)
        for field in fields:
            p.add_argument('--' + field, required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.train, args.validation, args.output)
    elif args.command == 'run':
        run_baseline(args.inputs, args.model, args.output)
    else:
        # Frozen replay checks original artifacts before reading model-produced text.
        frozen = json.loads((ROOT / 'tests/fixtures/book-baseline/report.json').read_text())
        verify_files(args.run, {'run.json': frozen['run_sha256'], **frozen['retained_sha256']})
        score_run(args.inputs, args.run, args.output)


if __name__ == '__main__':
    main()
