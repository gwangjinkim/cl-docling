"""Frozen selected adapter -> native merged export -> fresh matched runtimes.

No training, selection, downloads or publication. All children run sequentially
in the foreground with the existing bounded interruption cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from benchmark_contract import ROOT, compare_runs
from adapted_benchmark_contract import CASES, validate_selected

helpers = runpy.run_path(str(ROOT / 'scripts/run-memory-benchmark.py'))
digest, child = helpers['digest'], helpers['child']


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'adapter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('This benchmark requires Darwin/Metal and Darwin RSS units')
    checkpoint, adapter, output = args.checkpoint.resolve(), args.adapter.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Output must be new')
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    selection = json.loads((ROOT / 'tests/fixtures/selection-results/selection.json').read_text())
    manifest_file = ROOT / 'tests/fixtures/selection-pages/manifest.json'
    manifest = json.loads(manifest_file.read_text())
    if selection['selected_step'] != 16 or digest(manifest_file) != selection['manifest_sha256']:
        raise ValueError('Frozen selection changed')
    for folder, hashes in ((checkpoint, {n: e['sha256'] for n, e in lock['files'].items()}),
                           (adapter, selection['adapter_sha256'])):
        for name, wanted in hashes.items():
            if digest(folder / name) != wanted:
                raise ValueError('Artifact hash mismatch: ' + name)
    for name, path in CASES.items():
        if digest(ROOT / path) != manifest['files'][name + '.png']:
            raise ValueError('Image hash mismatch: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'cases.json', CASES)
    env = {k: v for k, v in os.environ.items() if k not in ('DOCLING_BENCHMARK_ADAPTER', 'DOCLING_BENCHMARK_CASES')}
    env.update(DOCLING_MODEL=str(checkpoint), TB_DEVICE='gpu', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    native = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit', '--script']
    child(native + ['scripts/check-benchmark-native.lisp'], ROOT, env)
    merged = output / 'merged'
    child(native + ['scripts/export-selected-model.lisp', str(merged)], ROOT,
          {**env, 'DOCLING_BENCHMARK_ADAPTER': str(adapter)})
    child([sys.executable, 'scripts/check-selected-export.py', '--checkpoint', str(checkpoint),
           '--adapter', str(adapter), '--merged', str(merged), '--output', str(output / 'export-check.json')], ROOT, env)
    export_lock = dict(revision='local-native-selected-step-16',
                       files={p.name: {'sha256': digest(p)} for p in sorted(merged.iterdir()) if p.is_file()})
    write(output / 'export-lock.json', export_lock)
    child([sys.executable, 'scripts/audit-mlx-vlm.py', '--checkpoint', str(merged), '--cases', str(output / 'cases.json'),
           '--checkpoint-lock', str(output / 'export-lock.json'), '--output', str(output / 'numerical-audit.json')], ROOT, env)
    rows, memory = [], {}
    order = ['native-adapter', 'native-merged', 'python-transformers-merged', 'python-mlx-merged']
    for label in order:
        report_file = output / (label + '.json')
        run_env = {**env, 'DOCLING_MODEL': str(merged), 'DOCLING_BENCHMARK_CASES': str(output / 'cases.json')}
        if label.startswith('native'):
            command = native + ['scripts/benchmark-native.lisp', str(report_file)]
            if label == 'native-adapter':
                run_env.update(DOCLING_MODEL=str(checkpoint), DOCLING_BENCHMARK_ADAPTER=str(adapter))
        else:
            script = 'benchmark-python.py' if label == 'python-transformers-merged' else 'benchmark-mlx-vlm.py'
            command = [sys.executable, 'scripts/' + script, '--checkpoint', str(merged),
                       '--output', str(report_file), '--cases', str(output / 'cases.json')]
            if label == 'python-mlx-merged':
                command += ['--mode', 'corrected', '--reference', str(output / 'native-adapter.json')]
        memory[label] = child(command, ROOT, run_env)
        row = dict(json.loads(report_file.read_text()), runtime=label)
        validate_selected(row['cases'])
        rows.append(row)
    write(output / 'report.json', dict(
        summary=compare_runs(rows, cases=CASES), runtime_order=order, whole_process_memory=memory,
        project_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        engine_revision=subprocess.check_output(['git', '-C', env['DOCLING_ENGINE'], 'rev-parse', 'HEAD'], text=True).strip(),
        base_revision=lock['revision'], adapter_sha256=selection['adapter_sha256'], selected_step=16,
        image_sha256={name: digest(ROOT / path) for name, path in CASES.items()},
        environment_lock_sha256=digest(ROOT / 'references/mlx-vlm/uv.lock'),
        hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        platform=platform.platform(), python=sys.version, exact_selected_outputs=True,
        scope='M6.5 selected adapter versus native/Python merged export; FP32, all tiles; no new quality claim'))
    print('PASS: all 48 measurements and selected outputs match; native export reloads in both Python runtimes.', flush=True)


if __name__ == '__main__':
    main()
