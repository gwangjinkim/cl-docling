"""Independent Transformers oracle for the native tensor-to-token milestone."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from transformers import Idefics3Config, Idefics3ForConditionalGeneration, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tied", action="store_true", help="Tiny canonical tied-head checkpoint")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(418)
    if args.checkpoint:
        lock = json.loads((Path(__file__).resolve().parents[1] / "references/smoldocling.lock.json").read_text())
        for name, entry in lock["files"].items():
            with (args.checkpoint / name).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], name
        model = Idefics3ForConditionalGeneration.from_pretrained(
            args.checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
        config = model.config
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        image_tokens = (config.vision_config.image_size // config.vision_config.patch_size // config.scale_factor) ** 2
        # Explicit pre-expanded prompt: this does not implement the image processor.
        prompt = ("<|im_start|>User: Convert this page to docling.\n<fake_token_around_image>"
                  "<global-img>" + "<image>" * image_tokens +
                  "<fake_token_around_image><end_of_utterance>\nAssistant:")
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        batch = 1
    else:
        config = Idefics3Config(
            vision_config=dict(hidden_size=12, intermediate_size=20, num_hidden_layers=2,
                               num_attention_heads=3, num_channels=3, image_size=8, patch_size=2),
            text_config=dict(model_type="llama", hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             vocab_size=32, max_position_embeddings=128, rms_norm_eps=1e-5),
            scale_factor=2, image_token_id=31, pad_token_id=0, tie_word_embeddings=args.tied,
        )
        config._attn_implementation = "eager"
        model = Idefics3ForConditionalGeneration(config).float().eval()
        model.save_pretrained(args.output)
        ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3], [4, 31, 31, 31, 31, 5, 0]])
        batch = 2
        tokenizer = None
    size = config.vision_config.image_size
    pixels = torch.linspace(-1, 1, batch * 3 * size * size).reshape(batch, 3, size, size)
    side = size // config.vision_config.patch_size
    patch_mask = torch.ones(batch, side, side, dtype=torch.bool)
    if batch == 2:
        patch_mask[1, 3:, :] = False
        patch_mask[1, :, 2:] = False
    pixel_mask = patch_mask.repeat_interleave(config.vision_config.patch_size, 1).repeat_interleave(config.vision_config.patch_size, 2)
    mask = torch.ones_like(ids)
    if batch == 2:
        mask[1, -1] = 0
    outputs = {"pixels": pixels, "input_ids": ids.float(), "attention_mask": mask.float(),
               "patch_mask": patch_mask.float(), "pixel_mask": pixel_mask.float()}
    with torch.no_grad():
        features = model.model.get_image_features(pixel_values=pixels[:, None], pixel_attention_mask=pixel_mask[:, None],
                                                 return_dict=True).pooler_output
        merged = model.model.inputs_merger(ids, model.get_input_embeddings()(ids), features)
        outputs["features"] = features.contiguous()
        outputs["merged"] = merged.contiguous()
        outputs["logits"] = model(input_ids=ids, image_hidden_states=features, attention_mask=mask,
                                  use_cache=False).logits.contiguous()
        # Greedy single-row sequence, image insertion only on prefill, native-like cache.
        prompt_ids = ids[:1]
        result = model(input_ids=prompt_ids, image_hidden_states=features[:1], use_cache=True)
        # Independent explicit regression for an image-token ID occurring in decode.
        forced = model(input_ids=torch.tensor([[config.image_token_id]]),
                       past_key_values=result.past_key_values, use_cache=True)
        outputs["forced-image-token"] = forced.logits.contiguous()
        result = model(input_ids=prompt_ids, image_hidden_states=features[:1], use_cache=True)
        tokens = []
        margins = []
        for step in range(8):
            logits = result.logits[:, -1:, :]
            outputs[f"step-{step}"] = logits.contiguous()
            top = logits[0, 0].topk(2)
            margins.append(float(top.values[0] - top.values[1]))
            token = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            # Full recomputation includes the ORIGINAL image slots only. A generated
            # image token is an ordinary text embedding, not another image request.
            if step == 0:
                full_emb = merged[:1]
            else:
                full_emb = torch.cat((merged[:1], model.get_input_embeddings()(torch.tensor([tokens[:-1]]))), dim=1)
            full = model(inputs_embeds=full_emb, use_cache=False).logits[:, -1:, :]
            outputs[f"full-{step}"] = full.contiguous()
            if step < 7:
                result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
    save_file(outputs, args.output / "reference.safetensors")
    source = args.checkpoint or args.output
    with safe_open(source / "model.safetensors", framework="pt") as archive:
        weight_count = len(archive.keys())
    manifest = {"schema_version": 1, "seed": 418, "real": bool(args.checkpoint), "tied": args.tied,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "atol": 0.001 if args.checkpoint else 0.00002, "rtol": 0.0003,
                "tokens": tokens, "top_two_margins": margins,
                "raw_text": tokenizer.decode(tokens, skip_special_tokens=False) if tokenizer else None,
                "weights": weight_count, "dtype": "float32", "attention": "eager",
                "revision": lock["revision"] if args.checkpoint else None,
                "assets_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in (lock["files"] if args.checkpoint else ["generation_config.json"])
                                  if name not in {"config.json", "model.safetensors"}},
                "reference_sha256": hashlib.sha256((args.output / "reference.safetensors").read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.checkpoint:
        for case in ["unknown-text", "resampler", "qk-norm", "noise", "tie-policy", "missing-text", "wrong-shape", "extra-weight"]:
            destination = args.output / "invalid" / case
            destination.mkdir(parents=True, exist_ok=True)
            invalid_config = json.loads((args.output / "config.json").read_text())
            weights = load_file(args.output / "model.safetensors")
            text = invalid_config["text_config"]
            if case == "unknown-text":
                text["unknown_numerical_feature"] = True
            elif case == "resampler":
                text["use_resampler"] = True
            elif case == "qk-norm":
                text["qk_layer_norms"] = True
            elif case == "noise":
                text["neftune_noise_alpha"] = 1.0
            elif case == "tie-policy":
                invalid_config["tie_word_embeddings"] = "false"
            elif case == "missing-text":
                del weights["model.text_model.norm.weight"]
            elif case == "wrong-shape":
                weights["model.text_model.norm.weight"] = torch.zeros(1)
            else:
                weights["model.text_model.unknown.weight"] = torch.zeros(1)
            (destination / "config.json").write_text(json.dumps(invalid_config))
            save_file(weights, destination / "model.safetensors")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
