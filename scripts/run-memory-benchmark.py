"""M6.4: fresh native project revisions on the same engine, then controlled Python."""
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


def child(command, cwd, env):
    process = subprocess.Popen(command, cwd=cwd, env=env)
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
    parser.add_argument('--baseline-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('Darwin RSS units and Metal required')
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    projects = {'native-before': args.baseline_project.resolve(), 'native-after': ROOT}
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    for name, entry in lock['files'].items():
        if digest(checkpoint / name) != entry['sha256']:
            raise ValueError('Checkpoint mismatch: ' + name)
    hashes = {name: digest(ROOT / path) for name, path in CASES.items()}
    if hashes != json.loads((ROOT / 'references/benchmark.lock.json').read_text())['image_sha256']:
        raise ValueError('Frozen images changed')
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint), 'TB_DEVICE': 'gpu',
           'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    revisions = {}
    for label, project in projects.items():
        if any(digest(project / path) != hashes[name] for name, path in CASES.items()):
            raise ValueError('Project images differ')
        if digest(project / 'scripts/benchmark-native.lisp') != digest(ROOT / 'scripts/benchmark-native.lisp'):
            raise ValueError('Native timing harnesses differ')
        revisions[label] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
        subprocess.run(['sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script',
                        'scripts/check-benchmark-native.lisp'], cwd=project, env=env, check=True)
    output.mkdir(parents=True, exist_ok=False)
    order, rows, memory = ['native-before', 'native-after', 'python-corrected'], [], {}
    for label in order:
        file = output / f'{label}.json'
        command = (['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                    '--script', 'scripts/benchmark-native.lisp', str(file)] if label in projects else
                   [sys.executable, 'scripts/benchmark-mlx-vlm.py', '--checkpoint', str(checkpoint),
                    '--output', str(file), '--mode', 'corrected'])
        memory[label] = child(command, projects.get(label, ROOT), env)
        rows.append(dict(json.loads(file.read_text()), runtime=label))
    report = dict(summary=compare_runs(rows), runtime_order=order, project_revisions=revisions,
                  engine_revision=subprocess.check_output(['git', '-C', env['DOCLING_ENGINE'], 'rev-parse', 'HEAD'], text=True).strip(),
                  whole_process_memory=memory, image_sha256=hashes, model_revision=lock['revision'],
                  platform=platform.platform(), python=sys.version,
                  hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  environment_lock_sha256=digest(ROOT / 'references/mlx-vlm/uv.lock'),
                  exact_workload_and_outputs=True, scope='M6.4 vision block lifetimes only; same engine, model, FP32/Metal')
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for row in rows:
        print(row['runtime'], 'maximum MLX peak', max(s['mlx_peak_bytes'] for c in row['cases'].values() for s in c['samples']), flush=True)


if __name__ == '__main__':
    main()
