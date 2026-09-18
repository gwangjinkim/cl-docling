"""Bounded foreground no-update numerical/resource qualification on reviewed books."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]


def validate_native(run, policy, device):
    if (run['device'] != device or run['updates'] != 0 or run['remaining_handles'] != 0 or
            set(run['cases']) != set(policy['cases'])):
        raise ValueError('Incomplete probe, wrong device, update or leaked handle')
    for name, case in run['cases'].items():
        if (any(case[k] != v for k, v in policy['cases'][name].items()) or
                case['gradient_tensors'] != policy['gradient_tensors'] or
                not math.isfinite(case['loss']) or case['loss'] < 0 or
                any(case[k] is not True for k in ('base_and_adapter_unchanged', 'handles_restored', 'parity_passed'))):
            raise ValueError('Changed input, incomplete/invalid gradients or failed numerical check')
        for stage in ('features', 'gradients'):
            record = case[stage]
            if (not math.isfinite(record['seconds']) or record['seconds'] < 0 or
                    not 0 <= record['mlx_active_bytes'] <= record['mlx_peak_bytes'] <= policy['max_stage_mlx_bytes']):
                raise ValueError('Invalid measurement or exceeded native allocation ceiling')
    return True


def run_child(command, env, log, timeout):
    """Reap our direct child on success, error, timeout or cancellation; macOS RSS bytes."""
    start = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic() - start > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.2)
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
    return dict(whole_process_peak_rss_bytes=usage.ru_maxrss,
                wall_seconds=time.monotonic()-start, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'context', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('This resource protocol uses macOS wait4 RSS byte units')
    inputs, context, checkpoint, output = (p.resolve() for p in (args.inputs, args.context, args.checkpoint, args.output))
    policy_path = ROOT / 'references/book-gradients.lock.json'
    policy = json.loads(policy_path.read_text())
    context_path = ROOT / 'tests/fixtures/book-context/report.json'
    context_report = json.loads(context_path.read_text())
    model_path = ROOT / 'references/smoldocling.lock.json'
    model = json.loads(model_path.read_text())
    model_files = {n: v['sha256'] for n, v in model['files'].items()}
    context_files = {'report.json': digest(context_path), **context_report['retained_sha256']}
    engine = ROOT.parent / 'cl-transformer-blocks'
    libraries = {'TB_MLX_LIBRARY': engine / '.build/native/libtb_mlx.dylib',
                 'DOCLING_IMAGE_LIBRARY': ROOT / '.build/native/libdocling_images.dylib'}
    def verify():
        verify_files(inputs, context_report['input_sha256'])
        verify_files(context, context_files)
        verify_files(checkpoint, model_files)
    verify()
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, **{k: str(v) for k, v in libraries.items()}, 'DOCLING_ENGINE': str(engine),
           'DOCLING_MODEL': str(checkpoint), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    memory, runs = {}, {}
    commands = [('python', [sys.executable, 'scripts/book-gradient-reference.py', str(inputs), str(context),
                             str(checkpoint), str(output / 'reference')])]
    commands.extend((device, ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                              '--script', 'scripts/book-gradients.lisp', str(inputs), str(output / 'reference'),
                              str(output / (device + '.json'))]) for device in policy['devices'])
    for label, command in commands:
        print('Starting bounded book gradient probe: ' + label, flush=True)
        memory[label] = run_child(command, {**env, 'TB_DEVICE': label}, output / (label + '.log'),
                                  policy['child_timeout_seconds'])
        # Keep completed child resource evidence even if acceptance fails.
        (output / (label + '-process.json')).write_text(json.dumps(memory[label], indent=2) + '\n')
        if memory[label]['whole_process_peak_rss_bytes'] > policy['max_child_rss_bytes']:
            raise ValueError('Exceeded process RSS acceptance ceiling: ' + label)
        if label != 'python':
            runs[label] = json.loads((output / (label + '.json')).read_text())
            validate_native(runs[label], policy, label)
        print('Completed: ' + label, flush=True)
    verify()
    report = dict(schema_version=1, training_ready=False, gradient_probe_passed=True, updates=0,
                  scope=policy['scope'], policy=policy, policy_sha256=digest(policy_path),
                  context_report_sha256=digest(context_path), model_revision=model['revision'],
                  model_lock_sha256=digest(model_path), input_sha256=context_report['input_sha256'],
                  platform=platform.platform(), hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  engine_revision=subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip(),
                  library_sha256={k: digest(v) for k, v in libraries.items()},
                  implementation_sha256={p: digest(ROOT / p) for p in ('scripts/book_gradients.py',
                      'scripts/book-gradients.lisp', 'scripts/book-gradient-reference.py', 'src/document-training.lisp')},
                  python=json.loads((output / 'reference/report.json').read_text()), native=runs, process_memory=memory,
                  retained_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({d: {n: c['loss'] for n, c in r['cases'].items()} for d, r in runs.items()}, indent=2))


if __name__ == '__main__':
    main()
