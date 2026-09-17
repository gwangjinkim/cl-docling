"""Independent FP32 Transformers image generation oracle; synthetic public input."""
import argparse
import hashlib
import json
import inspect
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    task = "Convert this page to docling."
    if not args.tiny:
        lock = json.loads((Path(__file__).resolve().parents[1] / "references/smoldocling.lock.json").read_text())
        for name, entry in lock["files"].items():
            with (args.checkpoint / name).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], name
        image = Image.new("RGB", (320, 448), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 30), "COMMON LISP", fill="black", font=ImageFont.load_default(size=28))
        draw.multiline_text((24, 95), "Native document models.\n\nRead a page.\nKeep every tile.\nBuild together.",
                            fill="black", font=ImageFont.load_default(size=20), spacing=12)
        image.save(args.output / "page.png")
        template = json.loads((args.checkpoint / "chat_template.json").read_text())["chat_template"]
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint),
                                      tokenizer, image_seq_len=64, chat_template=template)
        chat = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": task}]}], add_generation_prompt=True)
        batch = processor(text=chat, images=[[image]], return_tensors="pt")
        pixels, ids = batch["pixel_values"], batch["input_ids"]
        image_info = processor.image_processor([[image]], return_row_col_info=True, return_tensors="pt")
        expanded = chat.replace("<image>", processor.replace_image_token(image_info, 0))
        (args.output / "prompt.txt").write_text(expanded)
    else:
        pixels = torch.linspace(-0.98, 0.99, 3*3*8*8).reshape(1, 3, 3, 8, 8)
        ids = torch.tensor([[1] + [31]*12 + [2, 3]])
        tokenizer = None
    model = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                dtype=torch.float32, attn_implementation="eager").eval()
    count = pixels.shape[1]
    data = {"pixels": pixels.contiguous(), "input_ids": ids.float()}
    with torch.no_grad():
        features = model.model.get_image_features(pixel_values=pixels, return_dict=True).pooler_output
        merged = model.model.inputs_merger(ids, model.get_input_embeddings()(ids), features)
        data["features"] = features.reshape(1, -1, features.shape[-1]).contiguous()
        data["merged"] = merged.contiguous()
        result = model(input_ids=ids, pixel_values=pixels, use_cache=True)
        tokens, margins = [], []
        budget = 16 if args.tiny else 256
        eos = None if args.tiny else model.generation_config.eos_token_id
        for i in range(budget):
            logits = result.logits[:, -1:, :]
            data[f"step-{i}"] = logits.contiguous()
            top = logits[0, 0].topk(2).values
            margins.append(float(top[0]-top[1]))
            token = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            if tokens[-1] == eos:
                break
            if i < budget-1:
                result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        # Independent complete recomputation for the last cached step.
        full = torch.cat((merged, model.get_input_embeddings()(torch.tensor([tokens[:-1]]))), dim=1)
        data["full-last"] = model(inputs_embeds=full, use_cache=False).logits[:, -1:, :].contiguous()
    save_file(data, args.output / "reference.safetensors")
    manifest = dict(tiles=count, tokens=tokens, margins=margins, task=task,
                    real=not args.tiny, budget=budget, eos=eos, stopped_on_eos=tokens[-1] == eos,
                    atol=0.00002 if args.tiny else 0.001, rtol=0.0003,
                    raw=tokenizer.decode(tokens, skip_special_tokens=False) if tokenizer else None,
                    prompt_length=ids.shape[1], dtype="float32", attention="eager",
                    versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                  pillow=PIL.__version__, numpy=np.__version__),
                    revision=None if args.tiny else lock["revision"],
                    processor_source_sha256=hashlib.sha256(Path(inspect.getfile(Idefics3ImageProcessorPil)).read_bytes()).hexdigest(),
                    image_sha256=None if args.tiny else hashlib.sha256((args.output / "page.png").read_bytes()).hexdigest(),
                    reference_sha256=hashlib.sha256((args.output / "reference.safetensors").read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
