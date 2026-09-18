"""Bounded training-only resource/resume qualification; no quality selection."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

from book_gradients import run_child
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'references/book-budget.lock.json'


def validate_native(run, policy, device, phase):
    start = 0 if phase == 'train' else policy['resume_after']
    expected = policy['schedule'][start:]
    if (phase not in ('train', 'resume') or run['phase'] != phase or run['device'] != device or
            run['initial_step'] != start or run['final_step'] != len(policy['schedule']) or
            run['remaining_handles'] != 0 or [s['name'] for s in run['steps']] != expected or
            [s['step'] for s in run['steps']] != list(range(start+1, len(policy['schedule'])+1))):
        raise ValueError('Incomplete or changed training-only schedule/device/cleanup')
    handles = run['steps'][0]['handles']
    for step in run['steps']:
        if (any(step[k] != v for k, v in policy['cases'][step['name']].items()) or
                not math.isfinite(step['loss']) or step['loss'] < 0 or
                step['base_unchanged'] is not True or step['adapter_changed'] is not True or
                step['handles'] != handles or handles <= 0):
            raise ValueError('Input, update, base preservation or handle failure')
        if (not math.isfinite(step['seconds']) or step['seconds'] < 0 or
                not 0 <= step['mlx_active_bytes'] <= step['mlx_peak_bytes'] <= policy['max_stage_mlx_bytes'] or
                not 0 <= step['mlx_cache_bytes'] <= policy['max_cache_bytes'] or
                not 0 <= step['post_cleanup_active_bytes'] <= step['mlx_peak_bytes']):
            raise ValueError('Exceeded or invalid resource measurement')
    return True


def validate_resume(train, resumed, policy):
    left = [s['loss'] for s in train['steps'][policy['resume_after']:]]
    right = [s['loss'] for s in resumed['steps']]
    if left != right or not left:
        raise ValueError('Fresh-process same-device losses differ')
    return True


def tensor_payloads(path):
    """Read exact finite FP32 checkpoint payloads without an ML runtime."""
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header, data = json.loads(raw[8:8+size]), raw[8+size:]
    result = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        start, end = info['data_offsets']
        shape = info['shape']
        if (info['dtype'] != 'F32' or not 0 <= start <= end <= len(data) or
                end-start != 4*math.prod(shape)):
            raise ValueError('Invalid FP32 checkpoint payload')
        payload = data[start:end]
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
            raise ValueError('Nonfinite checkpoint payload')
        result[name] = (shape, payload)
    return result


def compare_checkpoints(left, right, policy, expected_step):
    counts = {}
    for name, count in [('adapter_model.safetensors', policy['adapter_tensors']),
                        ('optimizer.safetensors', policy['optimizer_tensors'])]:
        a, b = (tensor_payloads(root / 'decoder' / name) for root in (left, right))
        if len(a) != count or a != b:
            raise ValueError('Checkpoint tensor coverage or exact equality failed: ' + name)
        counts[name] = len(a)
    a, b = (json.loads((root / 'decoder/training_state.json').read_text()) for root in (left, right))
    if a != b or a['step'] != expected_step:
        raise ValueError('Optimizer state/step count changed on resume')
    return dict(exact=True, step=expected_step, tensors=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('inputs', 'checkpoint', 'output'):
        parser.add_argument('--' + field, required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise ValueError('macOS RSS byte units and qualified CPU/Metal required')
    inputs, checkpoint, output = (p.resolve() for p in (args.inputs, args.checkpoint, args.output))
    policy = json.loads(POLICY.read_text())
    context_path = ROOT / 'tests/fixtures/book-context/report.json'
    context = json.loads(context_path.read_text())
    model_lock = ROOT / 'references/smoldocling.lock.json'
    model = json.loads(model_lock.read_text())
    model_files = {n: info['sha256'] for n, info in model['files'].items()}
    engine = ROOT.parent / 'cl-transformer-blocks'
    libraries = {'TB_MLX_LIBRARY': engine / '.build/native/libtb_mlx.dylib',
                 'DOCLING_IMAGE_LIBRARY': ROOT / '.build/native/libdocling_images.dylib'}
    verify_files(inputs, context['input_sha256'])
    verify_files(checkpoint, model_files)
    output.mkdir(parents=True, exist_ok=False)
    implementations = ['scripts/book_budget.py', 'scripts/book-budget.lisp', 'scripts/book_gradients.py',
                       'scripts/experiment-common.lisp', 'scripts/load-mlx.lisp']
    implementations += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'src').glob('*.lisp'))]
    provenance = dict(policy=policy, policy_sha256=digest(POLICY), input_sha256=context['input_sha256'],
                      context_report_sha256=digest(context_path), model_revision=model['revision'],
                      model_lock_sha256=digest(model_lock),
                      source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      engine_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=engine, text=True).strip(),
                      library_sha256={k: digest(v) for k, v in libraries.items()},
                      implementation_sha256={n: digest(ROOT / n) for n in implementations},
                      platform=platform.platform(), hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    env = {**os.environ, **{k: str(v) for k, v in libraries.items()},
           'DOCLING_ENGINE': str(engine), 'DOCLING_MODEL': str(checkpoint)}
    native, processes, comparisons = {}, {}, {}
    try:
        for device in policy['devices']:
            native[device], processes[device] = {}, {}
            root = output / device
            for phase in ('train', 'resume'):
                print(f'Starting bounded book budget probe: {device}/{phase}', flush=True)
                command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
                           '--script', 'scripts/book-budget.lisp', str(inputs), str(root), phase]
                memory = run_child(command, {**env, 'TB_DEVICE': device}, output / f'{device}-{phase}.log',
                                   policy['child_timeout_seconds'])
                processes[device][phase] = memory
                (output / f'{device}-{phase}-process.json').write_text(json.dumps(memory, indent=2) + '\n')
                if memory['whole_process_peak_rss_bytes'] > policy['max_child_rss_bytes']:
                    raise ValueError('Process RSS acceptance ceiling exceeded')
                run = json.loads((root / f'{phase}.json').read_text())
                validate_native(run, policy, device, phase)
                native[device][phase] = run
            validate_resume(native[device]['train'], native[device]['resume'], policy)
            comparisons[device] = dict(
                restored=compare_checkpoints(root / 'step-three', root / 'restored', policy, policy['resume_after']),
                final=compare_checkpoints(root / 'train-final', root / 'resume-final', policy, len(policy['schedule'])))
            print(f'Completed {device}: exact checkpoint and resumed losses.', flush=True)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(type=type(error).__name__, message=str(error)), indent=2) + '\n')
        raise
    finally:
        verify_files(inputs, context['input_sha256'])
        verify_files(checkpoint, model_files)
    report = dict(schema_version=1, training_ready=False, resource_probe_passed=True,
                  **provenance, native=native, process_memory=processes, resume=comparisons,
                  retained_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: bounded resource probe; not a selected adapter or quality result.')


if __name__ == '__main__':
    main()
