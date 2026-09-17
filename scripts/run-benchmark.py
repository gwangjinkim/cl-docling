"""Foreground-only macOS benchmark orchestration, provenance and wait4 process RSS."""
import argparse
import hashlib
import importlib.metadata
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
            raise ValueError('Model snapshot mismatch: ' + name)
    image_hashes = {name: digest(ROOT / path) for name, path in CASES.items()}
    if image_hashes != json.loads((ROOT / 'references/benchmark.lock.json').read_text())['image_sha256']:
        raise ValueError('Frozen benchmark images changed')
    subprocess.run(['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
                    'scripts/check-benchmark-native.lisp'], cwd=ROOT, check=True)
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint)}
    runs, memory = [], {}
    for label, device in [('lisp-cpu', 'cpu'), ('python-cpu', 'cpu'), ('lisp-metal', 'gpu')]:
        file = output / f'{label}.json'
        command = ([sys.executable, 'scripts/benchmark-python.py', '--checkpoint', str(checkpoint), '--output', str(file)]
                   if label == 'python-cpu' else ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit',
                    '--no-userinit', '--script', 'scripts/benchmark-native.lisp', str(file)])
        process = subprocess.Popen(command, cwd=ROOT, env={**env, 'TB_DEVICE': device})
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
        memory[label] = dict(whole_process_peak_rss_bytes=usage.ru_maxrss, user_seconds=usage.ru_utime,
                             system_seconds=usage.ru_stime)
        runs.append(json.loads(file.read_text()))
    report = dict(summary=compare_runs(runs), whole_process_memory=memory, image_sha256=image_hashes,
                  model_revision=lock['revision'], model_lock_sha256=digest(ROOT / 'references/smoldocling.lock.json'),
                  platform=platform.platform(), python=sys.version, runtime_order=['lisp-cpu', 'python-cpu', 'lisp-metal'],
                  repository_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  engine_revision=subprocess.check_output(['git', '-C', env['DOCLING_ENGINE'], 'rev-parse', 'HEAD'], text=True).strip(),
                  mlx_version=importlib.metadata.version('mlx'),
                  hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  exact_workload_and_outputs=True, scope='Three base-model PNGs; no language-overhead or general-quality claim')
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for runtime, pages in report['summary'].items():
        print(runtime, {name: round(stats['generation']['median'], 4) for name, stats in pages.items()}, flush=True)


if __name__ == '__main__':
    main()
