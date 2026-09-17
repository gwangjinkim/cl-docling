"""Independent Idefics3 vision fixtures; Python is reference tooling only."""
import argparse
import copy
import hashlib
import json
import platform
import sys
import importlib.metadata
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Idefics3Config
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3Connector,
    Idefics3VisionTransformer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--padded", action="store_true", help="Exercise nontrivial real patch masks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(417)
    if args.checkpoint:
        lock = json.loads((Path(__file__).resolve().parents[1] / "references/smoldocling.lock.json").read_text())
        for name, metadata in lock["files"].items():
            with (args.checkpoint / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != metadata["sha256"]:
                raise SystemExit(f"Pinned model asset mismatch: {name}")
        config = Idefics3Config.from_pretrained(args.checkpoint, local_files_only=True)
        source = args.checkpoint / "model.safetensors"
    else:
        config = Idefics3Config(
            vision_config=dict(hidden_size=12, intermediate_size=20, num_hidden_layers=2,
                               num_attention_heads=3, num_channels=3, image_size=8, patch_size=2),
            text_config=dict(model_type="llama", hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1),
            scale_factor=2,
        )
        source = None
    config.vision_config._attn_implementation = "eager"
    vision = Idefics3VisionTransformer(config.vision_config).float().eval()
    connector = Idefics3Connector(config).float().eval()
    if source:
        with safe_open(source, framework="pt", device="cpu") as archive:
            for module, prefix in [(vision, "model.vision_model."), (connector, "model.connector.")]:
                module.load_state_dict({key[len(prefix):]: archive.get_tensor(key).float()
                                        for key in archive.keys() if key.startswith(prefix)}, strict=True)
    weights = {"model.vision_model." + k: v.contiguous() for k, v in vision.state_dict().items()}
    weights.update({"model.connector." + k: v.contiguous() for k, v in connector.state_dict().items()})
    save_file(weights, args.output / "vision.safetensors")
    config.save_pretrained(args.output)
    size, patch = config.vision_config.image_size, config.vision_config.patch_size
    side = size // patch
    batch = 1 if source else 2
    pixels = torch.linspace(-1, 1, batch * 3 * size * size).reshape(batch, 3, size, size)
    masks = torch.ones(batch, side, side, dtype=torch.bool)
    if not source:
        masks[1, 3:, :] = False
        masks[1, :, 2:] = False
    elif args.padded:
        masks[0, side - 1:, :] = False
        masks[0, :, side // 2 + 1:] = False
    outputs = {}
    def capture(name):
        def hook(module, inputs, output):
            del module, inputs
            outputs[name] = output.detach().contiguous()
        return hook
    def capture_positions(module, inputs):
        del module
        outputs["positions"] = inputs[0].detach().contiguous()
    handles = [vision.embeddings.position_embedding.register_forward_pre_hook(capture_positions),
               vision.embeddings.register_forward_hook(capture("embeddings")),
               vision.encoder.layers[0].register_forward_hook(capture("first-layer"))]
    with torch.no_grad():
        outputs["patch-projection"] = vision.embeddings.patch_embedding(pixels).flatten(2).transpose(1, 2).contiguous()
        hidden = vision(pixel_values=pixels, patch_attention_mask=masks).last_hidden_state
        outputs["vision"] = hidden.contiguous()
        outputs["shuffle"] = connector.pixel_shuffle(hidden, config.scale_factor).contiguous()
        outputs["connector"] = connector(hidden).contiguous()
    for handle in handles:
        handle.remove()
    save_file({"pixels": pixels, "mask": masks.to(torch.float32), **outputs}, args.output / "reference.safetensors")
    manifest = {"schema_version": 1, "seed": 417, "torch": torch.__version__,
                "transformers": transformers.__version__, "real": bool(source),
                "revision": "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8" if source else None,
                "execution_dtype": "float32", "attention": "eager", "weights": len(weights),
                "padded": args.padded or not bool(source),
                "atol": 0.001 if source else 0.00002, "rtol": 0.0003,
                "sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                           for name in ["config.json", "vision.safetensors", "reference.safetensors"]}}
    manifest["reference_environment"] = {"python": sys.version, "platform": platform.platform(),
                                         "machine": platform.machine(), "mlx": importlib.metadata.version("mlx")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not source:
        for kind in ["unknown-field", "unknown-root", "activation", "dropout", "missing-weight", "shape", "extra-weight"]:
            directory = args.output / "invalid" / kind
            directory.mkdir(parents=True, exist_ok=True)
            invalid_config = json.loads((args.output / "config.json").read_text())
            invalid_weights = copy.copy(weights)
            if kind == "unknown-field":
                invalid_config["vision_config"]["rotary_vision"] = True
            elif kind == "unknown-root":
                invalid_config["connector_type"] = "resampler"
            elif kind == "activation":
                invalid_config["vision_config"]["hidden_act"] = "relu"
            elif kind == "dropout":
                invalid_config["vision_config"]["attention_dropout"] = 0.1
            elif kind == "missing-weight":
                del invalid_weights["model.vision_model.post_layernorm.bias"]
            elif kind == "shape":
                invalid_weights["model.vision_model.post_layernorm.bias"] = torch.zeros(1)
            else:
                invalid_weights["model.vision_model.surprise.weight"] = torch.zeros(1)
            (directory / "config.json").write_text(json.dumps(invalid_config))
            save_file(invalid_weights, directory / "vision.safetensors")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
