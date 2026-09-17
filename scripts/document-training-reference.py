"""Tiny Idefics3 + PEFT oracle for frozen-vision q/v LoRA; no downloads."""
import argparse
import json
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import Idefics3Config, Idefics3ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(418)
    config = Idefics3Config(
        vision_config=dict(hidden_size=12, intermediate_size=20, num_hidden_layers=2,
                           num_attention_heads=3, num_channels=3, image_size=8, patch_size=2),
        text_config=dict(model_type="llama", hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=32, max_position_embeddings=128, rms_norm_eps=1e-5),
        scale_factor=2, image_token_id=31, pad_token_id=0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    base = Idefics3ForConditionalGeneration(config).float().eval()
    base.save_pretrained(root / "base")
    pixels = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    ids = torch.tensor([[1, 31, 31, 31, 31, 3, 6, 2, 0], [4, 31, 31, 31, 31, 5, 9, 8, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 6, 2, -100],
                           [-100, -100, -100, -100, -100, -100, 9, 8, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        features = base.model.get_image_features(pixel_values=pixels[:, None], return_dict=True).pooler_output
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=r"model.text_model.layers.\d+.self_attn.(q_proj|v_proj)",
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    params = {k.replace(".default.", "."): p for k, p in model.named_parameters() if p.requires_grad}
    # Public native initialization contract: uint32 LCG, A uniform/sqrt(input width), B zero.
    seed = 17
    with torch.no_grad():
        for name, p in sorted(params.items()):
            if ".lora_A." in name:
                values = []
                for _ in range(p.numel()):
                    seed = (1664525 * seed + 1013904223) % 2**32
                    fraction = torch.tensor(seed / 2**32, dtype=torch.float32)
                    values.append((2 * fraction - 1) / torch.sqrt(torch.tensor(p.shape[1], dtype=torch.float32)))
                p.copy_(torch.stack(values).reshape_as(p))
            else:
                p.zero_()
    data = {"pixels": pixels, "features": features.contiguous(), "ids": ids.float(),
            "labels": labels.float(), "mask": mask.float()}
    optimizer = torch.optim.SGD(params.values(), lr=0.03)
    losses = []
    for step in range(3):
        optimizer.zero_grad()
        out = model(input_ids=ids, image_hidden_states=features, attention_mask=mask,
                    labels=labels, use_cache=False)
        losses.append(out.loss.item())
        out.loss.backward()
        save_file({k: p.grad.contiguous() for k, p in params.items()}, root / f"gradients-{step}.safetensors")
        if step == 0:
            with torch.no_grad():
                changed_loss = model(input_ids=ids, image_hidden_states=-features, attention_mask=mask,
                                     labels=labels, use_cache=False).loss.item()
            assert abs(changed_loss - losses[0]) > 1e-5
        optimizer.step()
    save_file({k: p.detach().contiguous() for k, p in params.items()}, root / "updated.safetensors")
    save_file(data, root / "inputs.safetensors")
    (root / "reference.json").write_text(json.dumps({"losses": losses, "changed_loss": changed_loss,
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "seed": 418, "adapter_seed": 17, "dtype": "float32", "attention": "eager"}, indent=2) + "\n")
    print(json.dumps({"losses": losses, "changed_loss": changed_loss}))


if __name__ == "__main__":
    main()
