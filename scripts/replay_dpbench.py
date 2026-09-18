"""Hash-bound, table-free parser replay. No inference, repairs, or baseline writes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from dpbench_score import score, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(directory, files):
    for name, wanted in files.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or
                not (directory / relative).resolve().is_relative_to(directory.resolve())):
            raise ValueError('Unsafe evidence path')
        if digest(directory / relative) != wanted:
            raise ValueError('Evidence hash mismatch: ' + name)


def parse_records(output, count):
    records = {}
    for line in output.splitlines():
        if not line.startswith('REPLAY\t'):
            continue
        _, index, written, elements, codes = line.split('\t')
        index = int(index)
        if index in records or written not in ('0', '1') or int(elements) < 0:
            raise ValueError('Invalid native replay record')
        diagnostics = [{'code': c} for c in codes.split(',') if c]
        if (written == '1') == bool(diagnostics):
            raise ValueError('Inconsistent strict export')
        records[index] = dict(markdown_written=written == '1',
                              non_table_elements=int(elements), diagnostics=diagnostics)
    if set(records) != set(range(count)):
        raise ValueError('Incomplete native replay')
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'baseline', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    frozen_path = ROOT / 'tests/fixtures/dpbench/report.json'
    frozen = json.loads(frozen_path.read_text())
    original_run = json.loads((ROOT / 'tests/fixtures/dpbench/run-summary.json').read_text())
    verify_files(ROOT, {'scripts/dpbench_score.py': original_run['implementation_files']['scripts/dpbench_score.py']})
    verify_files(args.inputs, {'manifest.json': frozen['manifest_sha256']})
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    verify_files(args.inputs, {'manifest.json': frozen['manifest_sha256'], **manifest['files']})
    verify_files(args.baseline, {'report.json': digest(frozen_path), 'run.json': frozen['run_sha256'],
                                **frozen['retained_sha256']})
    originals, expected, paths = {}, {}, []
    for case in manifest['cases']:
        name = case['name']
        prefix = args.baseline / 'native/base' / name
        original = json.loads(prefix.with_suffix('.json').read_text())
        reference = json.loads((args.inputs / (name + '.expected.json')).read_text())
        if original['raw'] != prefix.with_suffix('.doctags').read_text() or original['stop_reason'] != 'eos':
            raise ValueError('Replay requires identical recorded raw and EOS')
        if original.get('error') or original['tables'] or reference['tables']:
            raise ValueError('This replay only covers the table-free successful baseline executions')
        originals[name], expected[name] = original, reference
        paths.append(str(prefix.with_suffix('.doctags').resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    command = ['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
               'scripts/replay-dpbench.lisp', str(args.output.resolve()), *paths]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60, check=True)
    (args.output / 'native.log').write_text(completed.stdout + completed.stderr)
    records = parse_records(completed.stdout, len(paths))
    reports = {}
    for i, (name, original) in enumerate(originals.items()):
        result = {**original, **records[i], 'phase': 'parser-replay'}
        assert all(result[k] == original[k] for k in ('tokens', 'raw', 'stop_reason', 'tables'))
        md = args.output / (name + '.md')
        assert md.is_file() == result['markdown_written']
        reports[name] = score(expected[name], result, md.read_text() if md.is_file() else None)
        reports[name]['generated_tokens'] = len(original['tokens'])
        (args.output / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    # Verify again: replay must not mutate the frozen evidence or its inputs.
    verify_files(args.inputs, {'manifest.json': frozen['manifest_sha256'], **manifest['files']})
    verify_files(args.baseline, {'report.json': digest(frozen_path), 'run.json': frozen['run_sha256'],
                                **frozen['retained_sha256']})
    summary = dict(pages=len(reports), strict_exports=sum(r['strict_export'] for r in reports.values()),
                   accepted=sum(r['accepted'] for r in reports.values()))
    for metric, edits, total in (
        ('markdown_cer', 'character_edits', 'reference_characters'),
        ('markdown_wer', 'word_edits', 'reference_words'),
        ('raw_text_cer', 'raw_text_character_edits', 'raw_text_reference_characters'),
        ('raw_text_wer', 'raw_text_word_edits', 'raw_text_reference_words'),
    ):
        summary[metric] = sum(r[edits] for r in reports.values()) / sum(r[total] for r in reports.values())
    report = dict(schema_version=1, policy='Post-baseline parser repair on known regression pages; no new model run or OCR improvement claim.',
                  baseline_report_sha256=digest(frozen_path), manifest_sha256=frozen['manifest_sha256'],
                  original_artifacts_sha256=frozen['retained_sha256'],
                  implementation_sha256={p: digest(ROOT / p) for p in
                      ('src/doctags.lisp', 'src/document.lisp', 'scripts/replay-dpbench.lisp',
                       'scripts/replay_dpbench.py', 'scripts/dpbench_score.py')},
                  retained_sha256={p.name: digest(p) for p in sorted(args.output.iterdir())},
                  summary=summary, cases=reports)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
