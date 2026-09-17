"""Verify the frozen inputs, run one foreground Lisp process, retain every score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from adaptation_data import ROOT, PROTOCOL, validate_manifest
from table_score import score


def verify_files(folder, files):
    for name, entry in files.items():
        wanted = entry if isinstance(entry, str) else entry['sha256']
        with (folder / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != wanted:
            raise ValueError('Checksum mismatch: ' + str(folder / name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    output, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    fixtures = ROOT / 'tests/fixtures/adaptation-pages'
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    validate_manifest(manifest)
    verify_files(fixtures, manifest['files'])
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    verify_files(checkpoint, lock['files'])
    if output.exists():
        raise ValueError('Output must be new; no automatic retry')
    command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
               '--script', 'scripts/adaptation-experiment.lisp', str(fixtures), str(output)]
    subprocess.run(command, cwd=ROOT, check=True,
                   env={**os.environ, 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': args.device})
    report = dict(protocol=PROTOCOL, model_revision=lock['revision'], device=args.device,
                  platform=platform.platform(), machine=platform.machine(),
                  manifest_sha256=hashlib.sha256((fixtures / 'manifest.json').read_bytes()).hexdigest(),
                  repository_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  cases={}, summary={})
    for case in manifest['cases']:
        name = case['name']
        expected = json.loads((fixtures / f'{name}.json').read_text())
        report['cases'][name] = dict(split=case['split'])
        for phase in ('base', 'adapted'):
            result = json.loads((output / phase / f'{name}.json').read_text())
            report['cases'][name][phase] = score(expected, result)
    for split in ('train', 'heldout', 'regression'):
        subset = [c for c in report['cases'].values() if c['split'] == split]
        report['summary'][split] = {phase: dict(accepted=sum(c[phase]['accepted'] for c in subset), pages=len(subset),
                                                   exact_cells=sum(c[phase]['exact_cell_matches'] for c in subset),
                                                   expected_cells=sum(c[phase]['expected_cells'] for c in subset))
                                    for phase in ('base', 'adapted')}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
