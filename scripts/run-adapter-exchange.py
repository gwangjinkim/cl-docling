"""Fresh-process native Lisp -> Python/PEFT -> Lisp training exchange, no downloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, env=None):
    subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)


def python_phase(args):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import Idefics3ForConditionalGeneration

    torch.set_num_threads(4)
    root = args.output.resolve()
    data = load_file(root / "fixture/inputs.safetensors")
    ids, labels, mask = (data[k].long() for k in ("ids", "labels", "mask"))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else root / "fixture/base"
    real = args.checkpoint is not None
    atol = 1e-3 if real else 2e-5
    rate = 0.001 if real else 0.03

    def base(path=checkpoint):
        return Idefics3ForConditionalGeneration.from_pretrained(path, local_files_only=True,
                          dtype=torch.float32, attn_implementation="eager").eval()

    def params(model):
        return {k.replace(".default.", "."): p for k, p in model.named_parameters() if p.requires_grad}

    def output(model, features):
        return model(input_ids=ids, image_hidden_states=features, attention_mask=mask, labels=labels, use_cache=False)

    def greedy(model, features):
        n = int(torch.where(labels[0] != -100)[0][0])
        result = model(input_ids=ids[:1, :n], image_hidden_states=features[:1], use_cache=True)
        tokens = []
        for _ in range(4):
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(token.item())
            result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        return torch.tensor(tokens)

    artifact = "native" if args.phase == "python" else "returned"
    model = PeftModel.from_pretrained(base(), root / f"{artifact}-adapter", is_trainable=True).eval()
    transported = load_file(root / f"{artifact}-adapter/adapter_model.safetensors")
    expected_names = set(transported)
    assert set(params(model)) == expected_names
    assert all("model.text_model.layers." in k and (".q_proj." in k or ".v_proj." in k) for k in expected_names)
    for k, p in params(model).items():
        assert torch.equal(p, transported[k]), k
    observations = load_file(root / f"{artifact}-observations/model.safetensors")
    with torch.no_grad():
        features = model.base_model.model.model.get_image_features(pixel_values=data["pixels"][:, None], return_dict=True).pooler_output
        result = output(model, features)
        torch.testing.assert_close(result.logits, observations["logits"], atol=atol, rtol=3e-4)
        torch.testing.assert_close(result.loss.reshape(1), observations["loss"], atol=atol, rtol=0)
        assert torch.equal(greedy(model, features), observations["tokens"].long())
    report = {"phase": args.phase, "device_native": args.device, "real": real,
              "parameters": len(expected_names), "loss": result.loss.item(),
              "logits_max_error": (result.logits - observations["logits"]).abs().max().item(),
              "tokens": observations["tokens"].long().tolist()}
    merged = base(root / f"{artifact}-merged")
    for asset in checkpoint.iterdir():
        if asset.name not in {"config.json", "model.safetensors"} and asset.is_file() and asset.suffix in {".json", ".jinja", ".txt", ".model"}:
            exported = root / f"{artifact}-merged" / asset.name
            assert exported.read_bytes() == asset.read_bytes(), asset.name
    with torch.no_grad():
        actual = merged(input_ids=ids, pixel_values=data["pixels"][:, None], attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual, result.logits, atol=atol, rtol=3e-4)
    del merged, actual, result
    if args.phase == "python":
        frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.SGD(params(model).values(), lr=rate)
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            result = output(model, features)
            losses.append(result.loss.item())
            result.loss.backward()
            optimizer.step()
        for k, p in model.named_parameters():
            if k in frozen:
                assert torch.equal(p, frozen[k]), k
        del frozen
        model.save_pretrained(root / "python-adapter")
        with torch.no_grad():
            result = output(model, features)
            save_file({"logits": result.logits.contiguous(), "loss": result.loss.reshape(1).contiguous()}, root / "python-observations.safetensors")
        # One further independent step is the oracle for training after returning to Lisp.
        optimizer.zero_grad()
        output(model, features).loss.backward()
        optimizer.step()
        save_file({k: p.detach().contiguous() for k, p in params(model).items()}, root / "expected-returned.safetensors")
        report["python_training_losses"] = losses
        invalid = root / "invalid"
        config = json.loads((root / "native-adapter/adapter_config.json").read_text())
        weights = load_file(root / "native-adapter/adapter_model.safetensors")
        for case in ["base", "revision", "vision-targets", "dora", "missing", "extra", "shape", "dtype", "nonfinite"]:
            folder = invalid / case
            folder.mkdir(parents=True)
            c = dict(config)
            w = dict(weights)
            key = next(iter(w))
            if case == "base": c["base_model_name_or_path"] = "wrong/base"
            elif case == "revision": c["revision"] = "wrong-revision"
            elif case == "vision-targets": c["target_modules"] = ["q_proj", "v_proj"]
            elif case == "dora": c["use_dora"] = True
            elif case == "missing": del w[key]
            elif case == "extra": w["unexpected.weight"] = torch.zeros(1)
            elif case == "shape": w[key] = torch.zeros(1)
            elif case == "dtype": w[key] = w[key].half()
            else:
                w[key] = w[key].clone()
                w[key].flatten()[0] = float("nan")
            (folder / "adapter_config.json").write_text(json.dumps(c))
            save_file(w, folder / "adapter_model.safetensors")
    else:
        expected = load_file(root / "expected-returned.safetensors")
        actual = load_file(root / "returned-adapter/adapter_model.safetensors")
        assert actual.keys() == expected.keys()
        for k in expected:
            torch.testing.assert_close(actual[k], expected[k], atol=2e-6, rtol=3e-4)
    (root / f"report-{args.phase}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--real-fixture", type=Path, help="Existing M2 real-model reference tensors")
    parser.add_argument("--phase", choices=["run", "python", "verify"], default="run")
    args = parser.parse_args()
    if args.phase != "run":
        python_phase(args)
        return
    if args.checkpoint and not args.real_fixture:
        parser.error("--checkpoint requires --real-fixture")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.checkpoint:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        fixture = root / "fixture"
        fixture.mkdir()
        with safe_open(args.real_fixture / "reference.safetensors", framework="pt") as f:
            pixels, prompt = f.get_tensor("pixels"), f.get_tensor("input_ids").long()
        ids = torch.cat((prompt, torch.tensor([[3, 4]])), dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, -2:] = ids[:, -2:]
        save_file({"pixels": pixels, "ids": ids.float(), "labels": labels.float(), "mask": torch.ones_like(ids).float()}, fixture / "inputs.safetensors")
    else:
        run(sys.executable, ROOT / "scripts/document-training-reference.py", "--output", root / "fixture")
    common = ["--output", root, "--device", args.device]
    if args.checkpoint:
        common += ["--checkpoint", args.checkpoint.resolve()]
    env = {**os.environ, "TB_DEVICE": args.device, "DOCLING_EXCHANGE": str(root),
           "DOCLING_MODEL": str(args.checkpoint.resolve() if args.checkpoint else root / "fixture/base"),
           "DOCLING_TRAINING_FIXTURE": str(root / "fixture")}
    if args.checkpoint:
        env["DOCLING_EXCHANGE_REAL"] = "1"
    for phase in ["export", "reload"]:
        run("sbcl", "--dynamic-space-size", "4096", "--noinform", "--no-sysinit", "--no-userinit", "--script", ROOT / "scripts/test-adapter-exchange.lisp",
            env={**env, "DOCLING_EXCHANGE_PHASE": phase})
        run(sys.executable, __file__, *common, "--phase", "python" if phase == "export" else "verify")
    print(f"PASS: fresh-process Lisp -> Python -> Lisp adapter exchange ({args.device}, real={bool(args.checkpoint)})", flush=True)


if __name__ == "__main__":
    main()
