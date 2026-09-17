"""Independent full-PNG Transformers/PEFT oracle and fresh native-adapter verification."""
import argparse
import hashlib
import json
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

ROOT = Path(__file__).resolve().parents[1]
TASK = "Convert this page to docling."
ANSWER = "<doctag><otsl><ched>Item<ched>Count<ched>Price<nl><fcel>Books<fcel>3<fcel>12<nl><fcel>Pens<fcel>5<fcel>2<nl><fcel>Folders<fcel>2<fcel>8<nl></otsl></doctag>"


def parameters(model):
    return {k.replace(".default.", "."): p for k, p in model.named_parameters() if p.requires_grad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-adapter", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    lock = json.loads((ROOT / "references/smoldocling.lock.json").read_text())
    for name, info in lock["files"].items():
        with (args.checkpoint / name).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == info["sha256"], name
    image_path = ROOT / "tests/fixtures/table-pages/grid.png"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    template = json.loads((args.checkpoint / "chat_template.json").read_text())["chat_template"]
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TASK}]}
    image = Image.open(image_path).convert("RGB")
    prompt = processor(text=processor.apply_chat_template([user], add_generation_prompt=True),
                       images=[[image]], return_tensors="pt")
    transcript = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}],
                                              add_generation_prompt=False)
    assert transcript.endswith("<end_of_utterance>\n")
    batch = processor(text=transcript[:-1], images=[[image]], return_tensors="pt")
    start = prompt.input_ids.shape[1]
    assert torch.equal(batch.input_ids[:, :start], prompt.input_ids)
    eos = tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    assert batch.input_ids[0, -1].item() == eos
    ids = torch.nn.functional.pad(batch.input_ids, (0, 7), value=eos)
    mask = torch.nn.functional.pad(batch.attention_mask, (0, 7))
    labels = ids.clone()
    labels[:, :start] = -100
    labels[mask == 0] = -100
    assert batch.pixel_values.shape[1] == 13
    selected = torch.where(labels[0, 1:] != -100)[0]
    base = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                         dtype=torch.float32, attn_implementation="eager").eval()
    # Sequential independent Python encoder calls bound peak vision memory.
    with torch.no_grad():
        features = torch.cat([base.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                  pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                  for i in range(13)], dim=0).detach()
    if args.verify_adapter:
        model = PeftModel.from_pretrained(base, args.verify_adapter, is_trainable=True).eval()
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                target_modules=r"model.text_model.layers.\d+.self_attn.(q_proj|v_proj)",
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")).eval()
        seed = 17
        with torch.no_grad():
            for name, p in sorted(parameters(model).items()):
                if ".lora_A." in name:
                    values = []
                    for _ in range(p.numel()):
                        seed = (1664525 * seed + 1013904223) % 2**32
                        fraction = torch.tensor(seed / 2**32, dtype=torch.float32)
                        values.append((2 * fraction - 1) / torch.sqrt(torch.tensor(p.shape[1], dtype=torch.float32)))
                    p.copy_(torch.stack(values).reshape_as(p))
                else:
                    p.zero_()

    def forward():
        return model(input_ids=ids, image_hidden_states=features, attention_mask=mask, labels=labels, use_cache=False)

    if args.verify_adapter:
        expected = load_file(args.output / "updated.safetensors")
        actual = load_file(args.verify_adapter / "adapter_model.safetensors")
        assert expected.keys() == actual.keys() == parameters(model).keys()
        for k, p in parameters(model).items():
            assert torch.equal(p, actual[k]), k
            torch.testing.assert_close(actual[k], expected[k], atol=2e-6, rtol=3e-4)
        with torch.no_grad():
            output = forward()
            reference = load_file(args.output / "final.safetensors")
            torch.testing.assert_close(output.logits[:, selected], reference["logits"], atol=1e-3, rtol=3e-4)
            torch.testing.assert_close(output.loss.reshape(1), reference["loss"], atol=1e-3, rtol=0)
        report = {"loss": output.loss.item(), "max_logit_error": (output.logits[:, selected] - reference["logits"]).abs().max().item(),
                  "parameters": len(actual)}
        (args.verify_adapter / "python-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return

    save_file({"ids": ids.float(), "labels": labels.float(), "mask": mask.float(),
               "features": features.reshape(1, -1, features.shape[-1]).contiguous()}, args.output / "inputs.safetensors")
    optimizer = torch.optim.SGD(parameters(model).values(), lr=0.001)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        output = forward()
        losses.append(output.loss.item())
        output.loss.backward()
        save_file({k: p.grad.contiguous() for k, p in parameters(model).items()}, args.output / f"gradients-{step}.safetensors")
        optimizer.step()
        print(f"Python page step {step}: loss={losses[-1]}", flush=True)
        del output
    save_file({k: p.detach().contiguous() for k, p in parameters(model).items()}, args.output / "updated.safetensors")
    with torch.no_grad():
        output = forward()
        save_file({"logits": output.logits[:, selected].contiguous(), "loss": output.loss.reshape(1)}, args.output / "final.safetensors")
    report = {"task": TASK, "answer": ANSWER, "image": str(image_path.relative_to(ROOT)), "tiles": 13,
              "start": start, "length": ids.shape[1], "selected": selected.tolist(), "losses": losses,
              "final_loss": output.loss.item(), "rate": 0.001, "padding": 7,
              "revision": lock["revision"], "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
