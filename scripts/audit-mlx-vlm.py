"""Independent Torch features/last-prefill-logits vs stock and corrected MLX-VLM.

All three frozen pages, FP32; predeclared real-model tolerance 1e-3 + 3e-4*abs(ref).
This is a correctness audit, not a timing run. Retain stock mismatches too.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import runpy
import mlx.core as mx
import numpy as np
import torch
from transformers import Idefics3ForConditionalGeneration
from benchmark_contract import ROOT, CASES
from mlx_vlm_contract import ATOL, RTOL, SOURCE_REVISION


def difference(actual, reference):
    delta = np.abs(actual - reference)
    return dict(shape=list(actual.shape), finite=bool(np.isfinite(actual).all()),
                maximum_absolute_error=float(delta.max()),
                outside_tolerance=int(np.count_nonzero(delta > ATOL + RTOL * np.abs(reference))),
                passes=bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Explicit benchmark name-to-image JSON mapping')
    parser.add_argument('--checkpoint-lock', type=Path, help='Verified local export lock instead of the base snapshot')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    lock = json.loads((args.checkpoint_lock or ROOT / 'references/smoldocling.lock.json').read_text())
    cases_to_run = json.loads(args.cases.read_text()) if args.cases else CASES
    for name, entry in lock['files'].items():
        with (args.checkpoint / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256'], name
    harness = runpy.run_path(str(ROOT / 'scripts/benchmark-mlx-vlm.py'))
    model, processor, _ = harness['load'](args.checkpoint, False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    oracle = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval()
    assert oracle.config.vision_config.hidden_act == 'gelu_pytorch_tanh'
    assert oracle.config.vision_config.layer_norm_eps == 1e-6
    references = {}
    with torch.inference_mode():
        for name, file in cases_to_run.items():
            batch = harness['prepare'](processor, ROOT / file)
            pixels = torch.from_numpy(batch.pixel_values)
            features = torch.cat([oracle.model.get_image_features(pixel_values=pixels[:, i:i+1],
                                 return_dict=True).pooler_output for i in range(pixels.shape[1])], dim=0)
            logits = oracle(input_ids=torch.from_numpy(batch.input_ids), image_hidden_states=features,
                            use_cache=False).logits[:, -1]
            references[name] = (features.reshape(-1, features.shape[-1]).numpy().copy(), logits.numpy().copy())
            del features, logits, pixels
    del oracle
    gc.collect()
    results = {}
    for mode in ('stock', 'corrected'):
        if mode == 'corrected':
            model.vision_model.post_layernorm.eps = 1e-6
        cases = {}
        for name, file in cases_to_run.items():
            batch = harness['prepare'](processor, ROOT / file)
            features = harness['visual_features'](model, batch.pixel_values)
            logits = harness['prefill'](model, batch, features, None)[:, -1]
            mx.eval(features, logits)
            cases[name] = dict(features=difference(np.array(features), references[name][0]),
                               last_prefill_logits=difference(np.array(logits), references[name][1]),
                               pixel_sha256=hashlib.sha256(batch.pixel_values.tobytes()).hexdigest())
            del features, logits
            print(mode, name, cases[name], flush=True)
        results[mode] = cases
    import mlx_vlm.models.idefics3.vision as vision
    import mlx_vlm.models.mlp as mlp
    report = dict(atol=ATOL, rtol=RTOL, model_revision=lock['revision'],
                  upstream_source_revision=SOURCE_REVISION, results=results,
                  source_sha256={module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                                 for module in (vision, mlp)},
                  versions={name: importlib.metadata.version(name) for name in
                            ['mlx-vlm', 'mlx', 'torch', 'transformers', 'pillow', 'numpy']})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    if not all(case[key]['passes'] for case in results['corrected'].values()
               for key in ('features', 'last_prefill_logits')):
        raise ValueError('Corrected numerical gate failed; report retained, benchmark not qualified')


if __name__ == '__main__':
    main()
