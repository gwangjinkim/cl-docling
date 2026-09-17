"""M6.3: same native harness before/after one engine upload change, then Python."""
import argparse
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from benchmark_contract import ROOT, CASES, compare_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-engine', type=Path, required=True)
    parser.add_argument('--optimized-engine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('Darwin wait4 RSS units and Metal required')
    common = runpy.run_path(str(ROOT / 'scripts/run-mlx-vlm-benchmark.py'))
    child, digest = common['child'], common['digest']
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    for name, entry in lock['files'].items():
        if digest(checkpoint / name) != entry['sha256']:
            raise ValueError('Checkpoint mismatch: ' + name)
    image_hashes = {name: digest(ROOT / file) for name, file in CASES.items()}
    if image_hashes != json.loads((ROOT / 'references/benchmark.lock.json').read_text())['image_sha256']:
        raise ValueError('Image hashes changed')
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': 'gpu',
           'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    engines = {'native-before': args.baseline_engine.resolve(), 'native-after': args.optimized_engine.resolve()}
    revisions = {}
    for label, engine in engines.items():
        revisions[label] = subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
                        'scripts/check-benchmark-native.lisp'], cwd=ROOT,
                       env={**env, 'DOCLING_ENGINE': str(engine)}, check=True)
    output.mkdir(parents=True, exist_ok=False)
    rows, memory = [], {}
    order = ['native-before', 'native-after', 'python-corrected']
    for label in order:
        file = output / f'{label}.json'
        if label in engines:
            command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                       '--script', 'scripts/benchmark-native.lisp', str(file)]
            child_env = {**env, 'DOCLING_ENGINE': str(engines[label])}
        else:
            command = [sys.executable, 'scripts/benchmark-mlx-vlm.py', '--checkpoint', str(checkpoint),
                       '--output', str(file), '--mode', 'corrected']
            child_env = env
        memory[label] = child(command, child_env)
        rows.append(dict(json.loads(file.read_text()), runtime=label))
    report = dict(summary=compare_runs(rows), runtime_order=order, engine_revisions=revisions,
                  whole_process_memory=memory, image_sha256=image_hashes, model_revision=lock['revision'],
                  repository_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  platform=platform.platform(), python=sys.version,
                  hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  environment_lock_sha256=digest(ROOT / 'references/mlx-vlm/uv.lock'),
                  exact_workload_and_outputs=True,
                  scope='M6.3 engine typed-array upload only; same native graph, FP32/Metal/full tiles; controlled Python comparator')
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for runtime, cases in report['summary'].items():
        print(runtime, {name: round(case['generation']['median'], 4) for name, case in cases.items()}, flush=True)


if __name__ == '__main__':
    main()
