"""Independent pinned Transformers CPU/FP32 greedy reference for both frozen pages."""
import argparse
import hashlib
import json
from pathlib import Path

import PIL
from PIL import Image
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "references/smoldocling.lock.json").read_text())
    for name, entry in lock["files"].items():
        with (args.checkpoint / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], name
    fixtures = root / "tests/fixtures/table-pages"
    fixture_manifest = json.loads((fixtures / "manifest.json").read_text())
    for name, digest in fixture_manifest["files"].items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    template = json.loads((args.checkpoint / "chat_template.json").read_text())["chat_template"]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    all_equal = True
    for name in ("grid", "merged"):
        print(f"Generating independent Python reference: {name}", flush=True)
        image = Image.open(fixtures / f"{name}.png").convert("RGB")
        chat = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}], add_generation_prompt=True)
        batch = processor(text=chat, images=[[image]], return_tensors="pt")
        tokens = []
        with torch.no_grad():
            result = model(**batch, use_cache=True)
            for step in range(512):
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                if tokens[-1] == model.generation_config.eos_token_id:
                    break
                if step < 511:
                    result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        native = json.loads((args.native / f"{name}.json").read_text())
        same = tokens == native["tokens"] and raw == native["raw"]
        all_equal &= same
        report = {"case": name, "tokens": tokens, "raw": raw, "matches_native": same,
                  "stop_reason": "eos" if tokens[-1] == model.generation_config.eos_token_id else "length",
                  "tiles": batch["pixel_values"].shape[1], "prompt_tokens": batch["input_ids"].shape[1],
                  "device": "cpu", "dtype": "float32", "attention": "eager", "max_new_tokens": 512,
                  "revision": lock["revision"], "image_sha256": fixture_manifest["files"][f"{name}.png"],
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__}}
        (args.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {len(tokens)} tokens; exact native IDs/raw: {same}", flush=True)
    if not all_equal:
        raise SystemExit("Native/Python mismatch: see retained references")


if __name__ == "__main__":
    main()
