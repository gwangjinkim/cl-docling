"""Bounded MLX kernel probe: identical FP32 Q/K/V, zero mask versus no mask.

Not a page benchmark. Both are bidirectional; no precision/shape change.
One warm-up then five evaluations of newly constructed attention graphs per mode.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(17)
    q, k, v = [mx.array(rng.normal(size=(1, 12, 1024, 64)).astype(np.float32)) for _ in range(3)]
    zero = mx.zeros((1, 1, 1, 1024), dtype=mx.float32)
    mx.eval(q, k, v, zero)
    results, outputs = {}, {}
    for mode, mask in [('zero-mask', zero), ('no-mask', None)]:
        samples = []
        for repeat in range(6):
            mx.synchronize()
            mx.reset_peak_memory()
            start = perf_counter()
            result = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
            mx.eval(result)
            elapsed = perf_counter() - start
            samples.append(dict(seconds=elapsed, mlx_peak_bytes=mx.get_peak_memory()))
        outputs[mode] = np.array(result)
        results[mode] = dict(warmup=samples[0], samples=samples[1:])
    delta = np.abs(outputs['zero-mask'] - outputs['no-mask'])
    equivalent = bool(np.allclose(outputs['zero-mask'], outputs['no-mask'], atol=2e-5, rtol=3e-4))
    report = dict(mlx=importlib.metadata.version('mlx'), device='gpu', dtype='float32',
                  shape=[1, 12, 1024, 64], seed=17, results=results,
                  maximum_absolute_difference=float(delta.max()), atol=2e-5, rtol=3e-4,
                  equivalent=equivalent, scope='Attention kernel probe, not page inference')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)
    if not equivalent:
        raise ValueError('Kernel equivalence failed')


if __name__ == '__main__':
    main()
