"""Verify the original retained resource/checkpoint evidence without model execution."""
import argparse
import json
from pathlib import Path
from book_budget import ROOT, POLICY, compare_checkpoints, validate_native, validate_resume
from replay_dpbench import digest, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    fixture = ROOT / 'tests/fixtures/book-budget/report.json'
    report = json.loads(fixture.read_text())
    verify_files(args.run, {'report.json': digest(fixture), **report['retained_sha256']})
    if report['policy_sha256'] != digest(POLICY):
        raise ValueError('Frozen policy changed')
    policy = report['policy']
    for device in policy['devices']:
        root = args.run / device
        for phase in ('train', 'resume'):
            validate_native(report['native'][device][phase], policy, device, phase)
        validate_resume(report['native'][device]['train'], report['native'][device]['resume'], policy)
        compare_checkpoints(root / 'step-three', root / 'restored', policy, policy['resume_after'])
        compare_checkpoints(root / 'train-final', root / 'resume-final', policy, len(policy['schedule']))
    print('PASS: every original artifact hash; exact restored/final tensors and resumed losses on both devices.')


if __name__ == '__main__':
    main()
