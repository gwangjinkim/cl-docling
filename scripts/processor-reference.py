#!/usr/bin/env python3
"""Independent PIL Idefics3 oracle. Synthetic data only; never run by Lisp.

Use explicit classes: Transformers 5.16.1 AutoProcessor incorrectly gates the PIL
class on torchvision in an environment without torchvision.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image
import transformers
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-size", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tile, longest, seq = (512, 2048, 64) if args.full_size else (16, 64, 2)
    cases = []
    for name, mode, h, w in [("portrait", "RGB", 37, 19), ("landscape", "RGB", 19, 37),
                              ("gray", "L", 31, 31), ("alpha", "RGBA", 7, 23),
                              ("gray-alpha", "LA", 23, 7), ("thin", "RGB", 1, 73),
                              ("downsample", "RGB", 131, 157), ("unsplit", "RGB", 19, 37)]:
        channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
        y, x, c = np.indices((h, w, channels))
        pixels = ((x * 37 + y * 71 + c * 97 + x * y * 3) % 256).astype(np.uint8)
        if channels == 1:
            pixels = pixels[:, :, 0]
        im = Image.fromarray(pixels, mode=mode)
        im.save(args.output / f"{name}.png")
        im.convert("RGB").save(args.output / f"{name}-rgb.png")
        proc = Idefics3ImageProcessorPil(size={"longest_edge": longest},
            max_image_size={"longest_edge": tile}, do_image_splitting=name != "unsplit",
            image_mean=[0.5] * 3, image_std=[0.5] * 3)
        result = proc([[im]], return_row_col_info=True, return_tensors="np")
        values = np.ascontiguousarray(result["pixel_values"], dtype="<f4")
        values.tofile(args.output / f"{name}.f32")
        rows, cols = result["rows"][0][0], result["cols"][0][0]
        prompt = Idefics3Processor.replace_image_token(
            SimpleNamespace(image_seq_len=seq, fake_image_token="<fake_token_around_image>",
                            image_token="<image>", global_image_tag="<global-img>"), result, 0)
        (args.output / f"{name}.prompt").write_text(prompt, encoding="utf-8")
        assert np.all(result["pixel_attention_mask"] == 1)
        cases.append(dict(name=name, shape=list(values.shape), rows=rows, cols=cols,
                          sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    # Independent padded batch: unequal tile counts, zero pixels AFTER normalization.
    proc = Idefics3ImageProcessorPil(size={"longest_edge": longest},
        max_image_size={"longest_edge": tile}, image_mean=[0.5]*3, image_std=[0.5]*3)
    batch = proc([[Image.open(args.output / "portrait.png")],
                  [Image.open(args.output / "thin.png")]], return_tensors="np")
    np.asarray(batch["pixel_values"], dtype="<f4").tofile(args.output / "batch.f32")
    np.asarray(batch["pixel_attention_mask"], dtype="u1").tofile(args.output / "batch.mask")
    manifest = dict(schema=1, tile=tile, longest=longest, seq=seq, cases=cases,
                    batch_shape=list(batch["pixel_values"].shape),
                    versions=dict(pillow=PIL.__version__, numpy=np.__version__,
                                  transformers=transformers.__version__),
                    processor_source_sha256=hashlib.sha256(
                        Path(inspect.getfile(Idefics3ImageProcessorPil)).read_bytes()).hexdigest())
    # Inputs deliberately outside the supported decoder contract.
    Image.new("P", (3, 5)).save(args.output / "unsupported-palette.png")
    Image.fromarray(np.arange(15, dtype=np.uint16).reshape(5, 3)).save(args.output / "unsupported-16bit.png")
    (args.output / "truncated.png").write_bytes((args.output / "portrait.png").read_bytes()[:80])
    damaged = bytearray((args.output / "portrait.png").read_bytes())
    damaged[-5] ^= 1
    (args.output / "bad-crc.png").write_bytes(damaged)
    Image.new("RGB", (3, 5), "red").save(args.output / "animated.png", save_all=True,
        append_images=[Image.new("RGB", (3, 5), "blue")], duration=100, loop=0)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
