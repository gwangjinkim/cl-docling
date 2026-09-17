"""Sequential, foreground-only M6.2 audit and three fresh Metal timing processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from benchmark_contract import ROOT, CASES, compare_runs


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(command, env):
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return dict(whole_process_peak_rss_bytes=usage.ru_maxrss, user_seconds=usage.ru_utime,
                system_seconds=usage.ru_stime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('This protocol qualifies macOS only; wait4 RSS units are bytes')
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    for name, entry in lock['files'].items():
        if digest(checkpoint / name) != entry['sha256']:
            raise ValueError('Checkpoint hash mismatch: ' + name)
    image_hashes = {name: digest(ROOT / path) for name, path in CASES.items()}
    if image_hashes != json.loads((ROOT / 'references/benchmark.lock.json').read_text())['image_sha256']:
        raise ValueError('Frozen images changed')
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': 'gpu',
           'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    subprocess.run(['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
                    'scripts/check-benchmark-native.lisp'], cwd=ROOT, env=env, check=True)
    output.mkdir(parents=True, exist_ok=False)
    child([sys.executable, 'scripts/audit-mlx-vlm.py', '--checkpoint', str(checkpoint),
           '--output', str(output / 'numerics.json')], env)
    runs, memory = [], {}
    order = ['lisp-metal', 'python-stock', 'python-corrected']
    for label in order:
        file = output / f'{label}.json'
        command = (['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                    '--script', 'scripts/benchmark-native.lisp', str(file)] if label == 'lisp-metal' else
                   [sys.executable, 'scripts/benchmark-mlx-vlm.py', '--checkpoint', str(checkpoint),
                    '--output', str(file), '--mode', label.removeprefix('python-')])
        memory[label] = child(command, env)
        runs.append(json.loads(file.read_text()))
    numerics = json.loads((output / 'numerics.json').read_text())
    for run in runs[1:]:
        mode = run['runtime'].removeprefix('python-mlx-vlm-')
        for name, case in run['cases'].items():
            if case['pixel_sha256'] != numerics['results'][mode][name]['pixel_sha256']:
                raise ValueError('Pixels changed between audit and timing')
    report = dict(summary=compare_runs(runs), whole_process_memory=memory, runtime_order=order,
                  image_sha256=image_hashes, model_revision=lock['revision'],
                  model_lock_sha256=digest(ROOT / 'references/smoldocling.lock.json'),
                  environment_lock_sha256=digest(ROOT / 'references/mlx-vlm/uv.lock'),
                  platform=platform.platform(), python=sys.version,
                  hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  repository_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  engine_revision=subprocess.check_output(['git', '-C', env['DOCLING_ENGINE'], 'rev-parse', 'HEAD'], text=True).strip(),
                  exact_workload_and_outputs=True,
                  scope='FP32 base-model Metal; controlled MLX-VLM harness, stock and explicit epsilon correction; no language-only attribution')
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for runtime, pages in report['summary'].items():
        print(runtime, {name: round(stats['generation']['median'], 4) for name, stats in pages.items()}, flush=True)


if __name__ == '__main__':
    main()
