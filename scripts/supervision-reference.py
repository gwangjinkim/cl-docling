"""Pinned Transformers chat/tokenizer/processor and NumPy answer-only label oracle."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import transformers
from transformers import AutoTokenizer
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor


def sexp(value):
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(value, (int, np.integer)):
        return str(value)
    return '(' + ' '.join(map(sexp, value)) + ')'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert transformers.__version__ == "5.16.1"
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "references/smoldocling.lock.json").read_text())
    for name, info in lock["files"].items():
        with (args.checkpoint / name).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == info["sha256"], name
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    template = json.loads((args.checkpoint / "chat_template.json").read_text())["chat_template"]
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    eos = tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    cases = [
        ("grid", "tests/fixtures/table-pages/grid.png", "Convert this page to docling.",
         "<doctag><otsl><ched>Item<ched>Count<ched>Price<nl><fcel>Books<fcel>3<fcel>12<nl><fcel>Pens<fcel>5<fcel>2<nl><fcel>Folders<fcel>2<fcel>8<nl></otsl></doctag>", 7),
        ("merged-target", "tests/fixtures/table-pages/merged.png", "Convert this page to docling.",
         "<doctag><otsl><ched>Office supplies<lcel><ched>Count<nl><fcel>Paper<fcel>Books<fcel>3<nl><ucel><fcel>Folders<fcel>2<nl><fcel>Writing<fcel>Pens<fcel>5<nl></otsl></doctag>", 0),
        ("unicode", "examples/data/native-page.png", "Read the page.\nKeep punctuation: 日本語!",
         "<doctag><text>Überblick — 日本語!\nSecond line.</text></doctag>", 3),
    ]
    entries = []
    summary = []
    for name, image_path, task, answer, padding in cases:
        image = Image.open(root / image_path).convert("RGB")
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task}]}
        prompt = processor.apply_chat_template([user], add_generation_prompt=True)
        full = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                                              add_generation_prompt=False)
        # The final newline is template formatting after EOS, not a training target.
        assert full.endswith("<end_of_utterance>\n")
        full = full[:-1]
        prompt_batch = processor(text=prompt, images=[[image]], return_tensors="np")
        full_batch = processor(text=full, images=[[image]], return_tensors="np")
        ids = full_batch["input_ids"][0]
        start = prompt_batch["input_ids"].shape[1]
        assert np.array_equal(ids[:start], prompt_batch["input_ids"][0]), "Tokenization crosses answer boundary"
        assert ids[-1] == eos and image_id not in ids[start:]
        assert eos not in ids[start:-1]
        size = len(ids)
        ids = np.pad(ids, (0, padding), constant_values=eos)
        mask = (np.arange(len(ids)) < size).astype(np.int32)
        labels = np.where((np.arange(len(ids)) >= start) & (mask == 1), ids, -100)
        selected = np.flatnonzero(labels[1:] != -100)
        targets = labels[1:][selected]
        fields = {"name": name, "image": image_path, "task": task, "answer": answer,
                  "prompt-length": start, "length": size, "pad-to": len(ids), "vocab-size": len(tokenizer),
                  "image-token-id": image_id, "eos-token-id": eos, "tiles": full_batch["pixel_values"].shape[1],
                  "ids": ids.tolist(), "labels": labels.tolist(), "mask": mask.tolist(),
                  "selected": selected.tolist(), "targets": targets.tolist()}
        entries.append('(' + ' '.join(':' + key + ' ' + sexp(value) for key, value in fields.items()) + ')')
        summary.append({"name": name, "prompt_tokens": start, "supervised_tokens": len(targets),
                        "padding": padding, "image_sha256": hashlib.sha256((root / image_path).read_bytes()).hexdigest()})
    data = '(\n' + '\n'.join(entries) + '\n)\n'
    (args.output / "cases.sexp").write_text(data, encoding="utf-8")
    manifest = {"revision": lock["revision"], "transformers": transformers.__version__, "numpy": np.__version__,
                "policy": "Exact single-user image-first prompt; assistant space + answer + EOS; final template newline excluded; labels unshifted",
                "source_sha256": hashlib.sha256(Path(inspect.getfile(Idefics3Processor)).read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(data.encode()).hexdigest(), "cases": summary}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
