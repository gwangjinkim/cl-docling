"""Bounded foreground development -> sealed validation choice -> fresh final test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from adaptation_data import ROOT
from selection_data import PROTOCOL, STEPS, choose_candidate, verify_manifest
from table_score import score


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_files(folder, files):
    for name, entry in files.items():
        wanted = entry if isinstance(entry, str) else entry['sha256']
        if digest(folder / name) != wanted:
            raise ValueError('Checksum mismatch: ' + str(folder / name))


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    fixtures = ROOT / 'tests/fixtures/selection-pages'
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    verify_manifest(manifest)
    verify_files(fixtures, manifest['files'])
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    verify_files(checkpoint, lock['files'])
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': args.device}
    timeline = []

    def event(name):
        timeline.append(dict(event=name, time=datetime.now(timezone.utc).isoformat()))

    def run(mode, destination, *extra):
        subprocess.run(['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                        '--script', 'scripts/selection-experiment.lisp', mode, str(fixtures), str(destination), *map(str, extra)],
                       cwd=ROOT, env=env, check=True)

    event('development_started')
    run('train', output / 'development')
    event('development_finished')
    validation = [c for c in manifest['cases'] if c['split'] == 'validation']
    reports = {}
    for step in STEPS:
        reports[step] = {}
        for case in validation:
            name = case['name']
            expected = json.loads((fixtures / f'{name}.json').read_text())
            result = json.loads((output / 'development' / f'step-{step}' / f'{name}.json').read_text())
            reports[step][name] = score(expected, result)
    selected = choose_candidate(reports)
    adapter = output / 'development' / f'adapter-{selected}' if selected else None
    hashes = {p.name: digest(p) for p in sorted(adapter.iterdir())} if adapter else {}
    selection = dict(selected_step=selected, validation=reports, adapter_sha256=hashes,
                     manifest_sha256=digest(fixtures / 'manifest.json'), protocol=PROTOCOL)
    write_new(output / 'selection.json', selection)
    selection_hash = digest(output / 'selection.json')
    event('selection_persisted')
    print(f'Validation-only choice sealed: step {selected}; final inference starts now.', flush=True)
    event('final_started')
    run('final', output / 'final', adapter if adapter else 'base')
    event('final_finished')
    if digest(output / 'selection.json') != selection_hash:
        raise ValueError('Selection changed during final evaluation')
    if adapter:
        verify_files(adapter, hashes)
        shutil.copytree(adapter, output / 'final/adapter')
    final = [c for c in manifest['cases'] if c['split'] in ('final', 'regression')]
    report = dict(selected_step=selected, selection_sha256=selection_hash, model_revision=lock['revision'],
                  device=args.device, platform=platform.platform(), protocol=PROTOCOL,
                  repository_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  cases={}, summary={})
    for case in final:
        name = case['name']
        expected = json.loads((fixtures / f'{name}.json').read_text())
        report['cases'][name] = dict(split=case['split'])
        for phase in ('base', 'adapted'):
            result = json.loads((output / 'final' / phase / f'{name}.json').read_text())
            report['cases'][name][phase] = score(expected, result)
    for split in ('final', 'regression'):
        subset = [c for c in report['cases'].values() if c['split'] == split]
        report['summary'][split] = {phase: dict(accepted=sum(c[phase]['accepted'] for c in subset), pages=len(subset),
                                               exact_cells=sum(c[phase]['exact_cell_matches'] for c in subset),
                                               expected_cells=sum(c[phase]['expected_cells'] for c in subset))
                                    for phase in ('base', 'adapted')}
    write_new(output / 'report.json', report)
    write_new(output / 'timeline.json', timeline)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
