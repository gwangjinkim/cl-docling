"""Fresh-process, ordinary Transformers reload of a Lisp-exported checkpoint."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    ref = load_file(args.fixture / "reference.safetensors")
    with (args.export / "model.safetensors").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] % 8 == 0, "Unaligned tensor payload"
    # Every stored parameter is EXACT after the explicitly promised FP32 promotion.
    with safe_open(args.source / "model.safetensors", framework="pt") as source:
        with safe_open(args.export / "model.safetensors", framework="pt") as exported:
            assert set(source.keys()) == set(exported.keys())
            for name in source.keys():
                torch.testing.assert_close(source.get_tensor(name).float(), exported.get_tensor(name), atol=0, rtol=0)
            count = len(source.keys())
    assets = []
    for name, digest in manifest["assets_sha256"].items():
        original = (args.source / name).read_bytes()
        assert hashlib.sha256(original).hexdigest() == digest, name
        assert (args.export / name).read_bytes() == original, name
        assets.append(name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.export, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        logits = model(input_ids=ref["input_ids"].long(), pixel_values=ref["pixels"][:, None],
                       pixel_attention_mask=ref["pixel_mask"][:, None].bool(),
                       attention_mask=ref["attention_mask"].long(), use_cache=False).logits
        torch.testing.assert_close(logits, ref["logits"], atol=manifest["atol"], rtol=manifest["rtol"])
        result = model(input_ids=ref["input_ids"][:1].long(), pixel_values=ref["pixels"][:1, None], use_cache=True)
        tokens = []
        for step in range(8):
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            if step < 7:
                result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        assert tokens == manifest["tokens"]
    if (args.export / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.export, local_files_only=True)
        assert tokenizer.decode(tokens, skip_special_tokens=False) == manifest["raw_text"]
    report = {"weights_exact_after_fp32_promotion": count, "assets_byte_exact": sorted(assets),
              "payload_aligned_to_8_bytes": True,
              "max_logit_error": float((logits - ref["logits"]).abs().max()),
              "tokens": tokens, "ordinary_transformers_reload": True}
    print(json.dumps(report, indent=2))
    (args.export / "reload-evidence.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
