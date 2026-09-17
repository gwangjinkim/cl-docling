"""Fresh-process native resume versus uninterrupted training, with a Torch state oracle."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def mutate_checkpoints(root):
    import torch
    from safetensors.torch import load_file, save_file
    source = root / "step-two/decoder"
    for case in ["version", "step", "names", "base", "missing", "shape", "dtype", "nonfinite"]:
        folder = root / "invalid" / case / "decoder"
        shutil.copytree(source, folder)
        manifest = json.loads((folder / "training_state.json").read_text())
        if case in {"version", "step", "names"}:
            key, value = {"version": ("format_version", 99), "step": ("step", 0), "names": ("trainable_parameters", [])}[case]
            manifest[key] = value
            (folder / "training_state.json").write_text(json.dumps(manifest))
        elif case == "base":
            config = json.loads((folder / "adapter_config.json").read_text())
            config["base_model_name_or_path"] = "wrong/base"
            (folder / "adapter_config.json").write_text(json.dumps(config))
        else:
            file = folder / "optimizer.safetensors"
            weights = load_file(file)
            name = next(iter(weights))
            if case == "missing": del weights[name]
            elif case == "shape": weights[name] = torch.zeros(1)
            elif case == "dtype": weights[name] = weights[name].half()
            else:
                weights[name].flatten()[0] = float("nan")
            save_file(weights, file)


def verify(args):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import Idefics3ForConditionalGeneration
    torch.set_num_threads(4)
    root = args.output.resolve()
    # Exact payload equality, not merely a similar loss after restarting.
    for left, right in [("step-two", "restored-before-step"), ("train-final", "resume-final")]:
        for file in ["adapter_model.safetensors", "optimizer.safetensors"]:
            a, b = (load_file(root / part / "decoder" / file) for part in (left, right))
            assert a.keys() == b.keys()
            for k in a: assert torch.equal(a[k], b[k]), (left, right, file, k)
        a, b = (json.loads((root / part / "decoder/training_state.json").read_text()) for part in (left, right))
        assert a == b
    reports = {phase: json.loads((root / f"{phase}.json").read_text()) for phase in ["train", "resume", "reset"]}
    assert reports["train"]["loss"] == reports["resume"]["loss"]
    expected = load_file(root / "train-peft/adapter_model.safetensors")
    reset = load_file(root / "reset-peft/adapter_model.safetensors")
    reset_difference = max((expected[k] - reset[k]).abs().max().item() for k in expected)
    assert reset_difference > 1e-8, "Control with discarded optimizer state must diverge"
    # Independent PyTorch continuation from the native moments, not a public format converter.
    base = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                            dtype=torch.float32, attn_implementation="eager").eval()
    model = PeftModel.from_pretrained(base, root / "step-two-peft", is_trainable=True).eval()
    params = {k.replace(".default.", "."): p for k, p in model.named_parameters() if p.requires_grad}
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params.values(), lr=0.0001, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.02)
    else:
        optimizer = torch.optim.SGD(params.values(), lr=0.0001, momentum=0.8, weight_decay=0.02)
    moments = load_file(root / "step-two/decoder/optimizer.safetensors")
    for name, p in params.items():
        canonical = name.replace(".model.text_model.", ".model.")
        if args.optimizer == "adamw":
            optimizer.state[p] = {"step": torch.tensor(2.), "exp_avg": moments[f"optimizer.first_moment.{canonical}"],
                                  "exp_avg_sq": moments[f"optimizer.second_moment.{canonical}"]}
        else:
            optimizer.state[p] = {"momentum_buffer": moments[f"optimizer.momentum.{canonical}"]}
    data = load_file(args.fixture / "inputs.safetensors")
    if args.real:
        features = data["features"].reshape(13, 64, -1)
    else:
        with torch.no_grad():
            features = base.model.get_image_features(pixel_values=data["pixels"][:, None], return_dict=True).pooler_output
    out = model(input_ids=data["ids"].long(), labels=data["labels"].long(), attention_mask=data["mask"].long(),
                image_hidden_states=features, use_cache=False)
    assert abs(out.loss.item() - reports["train"]["loss"]) < (1e-3 if args.real else 2e-5)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(params.values(), 0.1)
    optimizer.step()
    for k, p in params.items():
        torch.testing.assert_close(p, expected[k], atol=2e-6, rtol=3e-4)
    result = {"device": args.device, "real": args.real, "optimizer": args.optimizer,
              "step": 3, "factors": len(expected), "loss": out.loss.item(), "reset_max_difference": reset_difference,
              "torch_factor_max_error": max((p - expected[k]).abs().max().item() for k, p in params.items()),
              "native": reports}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--real", action="store_true", help="Use the existing M5.4 independent page reference")
    args = parser.parse_args()
    args.checkpoint, args.fixture, args.output = (p.resolve() for p in (args.checkpoint, args.fixture, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "DOCLING_MODEL": str(args.checkpoint), "DOCLING_RESUME_FIXTURE": str(args.fixture),
           "DOCLING_RESUME_OUTPUT": str(args.output), "TB_DEVICE": args.device, "DOCLING_RESUME_OPTIMIZER": args.optimizer}
    if args.real: env["DOCLING_RESUME_REAL"] = "1"
    else: env.pop("DOCLING_RESUME_REAL", None)
    for phase in ["train", "resume", "reset"]:
        subprocess.run(["sbcl", "--dynamic-space-size", "4096", "--noinform", "--no-sysinit", "--no-userinit", "--script",
                        "scripts/test-training-resume.lisp"], cwd=ROOT, env={**env, "DOCLING_RESUME_PHASE": phase}, check=True)
        if phase == "train" and not args.real: mutate_checkpoints(args.output)
    verify(args)
    print("PASS: fresh-process native document optimizer resume", flush=True)


if __name__ == "__main__":
    main()
