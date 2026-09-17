"""Controlled FP32/Metal harness; NOT the stock MLX-VLM generate/processor path.

Unmodified upstream model in --mode stock; one explicit, parameter-free epsilon
correction in --mode corrected. No quantization or cross-request cache. Upstream
operators may internally compile; this harness adds no whole-model compilation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from mlx_vlm.utils import load_model
from mlx_vlm.models.cache import KVCache
from benchmark_contract import ROOT, CASES
from mlx_vlm_contract import VERSION, SOURCE_REVISION, validate_config, compare_output


def load(checkpoint, corrected):
    if importlib.metadata.version('mlx-vlm') != VERSION:
        raise ValueError('Expected pinned MLX-VLM version')
    validate_config(json.loads((checkpoint / 'config.json').read_text()))
    mx.set_default_device(mx.gpu)
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = json.loads((checkpoint / 'chat_template.json').read_text())['chat_template']
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    model = load_model(checkpoint, lazy=True)
    model.update(tree_map(lambda x: x.astype(mx.float32), model.parameters()))
    model.eval()
    # Source-audited epsilon difference. MLX GELU 'precise' already means tanh.
    if model.vision_model.post_layernorm.eps != 1e-5:
        raise ValueError('Upstream LayerNorm changed; re-audit before correcting')
    if corrected:
        model.vision_model.post_layernorm.eps = 1e-6
    mx.eval(model.parameters())
    assert all(x.dtype == mx.float32 for _, x in tree_flatten(model.parameters()))
    return model, processor, perf_counter() - start


def prepare(processor, file):
    with Image.open(file) as image:
        prompt = processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': 'Convert this page to docling.'}]}],
            add_generation_prompt=True)
        batch = processor(text=prompt, images=[[image.convert('RGB')]], return_tensors='np')
    if not batch.pixel_attention_mask.all() or not batch.attention_mask.all():
        raise ValueError('This controlled harness only supports unpadded tiles/prompts')
    return batch


def visual_features(model, pixels):
    # Sequential vision and explicit tile evaluation match the native harness.
    features = []
    for i in range(pixels.shape[1]):
        tile = mx.array(pixels[:, i]).transpose(0, 2, 3, 1)
        hidden = model.vision_model(tile, patch_attention_mask=None, output_hidden_states=False)[0]
        value = model.connector(hidden)
        mx.eval(value)
        features.append(value)
    value = mx.concatenate(features, axis=1).reshape(-1, model.config.text_config.hidden_size)
    mx.eval(value)
    return value


def prefill(model, batch, features, cache):
    ids = mx.array(batch.input_ids)
    # Upstream cached-feature entry point retains upstream multimodal scatter.
    return model(ids, mx.array(batch.pixel_values), cache=cache,
                 pixel_attention_mask=mx.array(batch.pixel_attention_mask),
                 cached_image_features=features).logits


def page(model, processor, file):
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    batch = prepare(processor, file)
    t1 = perf_counter()
    features = visual_features(model, batch.pixel_values)
    t2 = perf_counter()
    cache = [KVCache() for _ in model.layers]
    logits = prefill(model, batch, features, cache)
    tokens = [int(mx.argmax(logits[:, -1], axis=-1).item())]
    del logits
    t3 = perf_counter()
    while len(tokens) < 512 and tokens[-1] != 49279:
        logits = model(mx.array([[tokens[-1]]]), None, cache=cache).logits
        tokens.append(int(mx.argmax(logits[:, -1], axis=-1).item()))
        del logits
    del cache
    t4 = perf_counter()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False)
    t5 = perf_counter()
    del features
    mx.synchronize()
    total = perf_counter() - start
    return (dict(tokens=tokens, raw=raw, stop_reason='eos' if tokens[-1] == 49279 else 'length',
                 prompt_ids=batch.input_ids[0].tolist(), tiles=batch.pixel_values.shape[1],
                 pixel_sha256=hashlib.sha256(batch.pixel_values.tobytes()).hexdigest()),
            dict(preprocess=t1-start, vision=t2-t1, prefill=t3-t2, decode=t4-t3, detokenize=t5-t4,
                 generation=total, parse_markdown=None, mlx_peak_bytes=mx.get_peak_memory(),
                 mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Explicit benchmark name-to-image JSON mapping')
    parser.add_argument('--reference', type=Path, help='Exact expected benchmark outputs')
    parser.add_argument('--mode', choices=['stock', 'corrected'], required=True)
    parser.add_argument('--audit', action='store_true', help='one untimed-claim output gate per page; not a benchmark')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    model, processor, load_seconds = load(args.checkpoint, args.mode == 'corrected')
    expected = json.loads((args.reference or ROOT / 'tests/fixtures/benchmark/python-cpu.json').read_text())
    cases = {}
    for name, file in (json.loads(args.cases.read_text()) if args.cases else CASES).items():
        result, warmup = page(model, processor, ROOT / file)
        mismatches = compare_output(result, expected['cases'][name])
        samples = []
        if not args.audit:
            if mismatches:
                raise ValueError(f'Output mismatch before timing: {name} {mismatches}')
            for repeat in range(3):
                again, sample = page(model, processor, ROOT / file)
                if again != result:
                    raise ValueError('Non-repeatable MLX-VLM output')
                samples.append(sample)
                print(f'{name} sample {repeat+1}: {sample["generation"]:.4f} seconds', flush=True)
        cases[name] = {**result, 'warmup': warmup, 'samples': samples, 'reference_mismatches': mismatches}
        print(f'{args.mode} {name}: {len(result["tokens"])} tokens; mismatches={mismatches}', flush=True)
    report = dict(runtime='python-mlx-vlm-' + args.mode, device='gpu', dtype='float32',
                  max_new_tokens=512, task='Convert this page to docling.', warmups=1, repeats=0 if args.audit else 3,
                  audit_only=args.audit, load_seconds=load_seconds, cases=cases,
                  upstream_source_revision=SOURCE_REVISION,
                  versions={name: importlib.metadata.version(name) for name in
                            ['mlx-vlm', 'mlx', 'transformers', 'pillow', 'numpy', 'tokenizers']},
                  corrections=[] if args.mode == 'stock' else ['vision post LayerNorm epsilon 1e-6'],
                  harness='Controlled HF PIL processor, sequential tiles, cached greedy, explicit EOS 49279; not stock CLI')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
