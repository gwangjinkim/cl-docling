"""Independent Torch W + (alpha/r) B@A check of every native-exported weight."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'adapter', 'merged', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    torch.set_num_threads(4)
    base = load_file(args.checkpoint / 'model.safetensors')
    exported = load_file(args.merged / 'model.safetensors')
    factors = load_file(args.adapter / 'adapter_model.safetensors')
    config = json.loads((args.adapter / 'adapter_config.json').read_text())
    expected_config = json.loads((args.checkpoint / 'config.json').read_text())
    for part in (expected_config, expected_config['vision_config'], expected_config['text_config']):
        part['dtype'] = 'float32'
        if part.get('torch_dtype'):
            part['torch_dtype'] = 'float32'
    assert json.loads((args.merged / 'config.json').read_text()) == expected_config
    assert base.keys() == exported.keys()
    changed, unchanged, used = {}, 0, set()
    for name, tensor in base.items():
        a = 'base_model.model.' + name.removesuffix('.weight') + '.lora_A.weight'
        b = a.replace('.lora_A.', '.lora_B.')
        expected = tensor.float()
        actual = exported[name]
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all(), name
        if a in factors:
            expected = expected + (config['lora_alpha'] / config['r']) * (factors[b] @ factors[a])
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-4)
            changed[name] = float((actual - expected).abs().max())
            used.update((a, b))
        else:
            assert torch.equal(actual, expected), name
            unchanged += 1
    assert used == set(factors) and len(changed) == 60 and len(used) == 120
    assets = []
    for file in args.checkpoint.iterdir():
        if file.is_file() and file.name not in ('config.json', 'model.safetensors') and file.suffix in ('.json', '.jinja', '.txt', '.model'):
            assert (args.merged / file.name).read_bytes() == file.read_bytes(), file.name
            assets.append(file.name)
    report = dict(weight_count=len(base), exact_unchanged_weights=unchanged, adapter_factors=len(used),
                  merged_weights=len(changed), maximum_merge_error=max(changed.values()), per_weight_max_error=changed,
                  atol=2e-6, rtol=3e-4, byte_identical_assets=sorted(assets), config_only_fp32_declarations=True,
                  torch_version=torch.__version__)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(f'PASS: {unchanged} exact frozen weights, {len(changed)} independently merged weights, {len(used)} factors.', flush=True)


if __name__ == '__main__':
    main()
