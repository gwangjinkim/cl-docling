"""Fresh ordinary Transformers reload and exact final-token parity for selected base."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

from book_final_evaluation import (MODEL_LOCK, POLICY, load_output, validate_policy,
                                   verify_inputs)
from replay_dpbench import digest, verify_files

ROOT = Path(__file__).resolve().parents[1]


def validate_native_report(report, policy):
    if (report.get('selected_kind') != 'base' or report.get('base_selected_identity') is not True
            or report.get('native_inference_runs') != 1
            or set(report.get('final', {})) != set(policy['final_cases'])
            or any(report['final'][name].get('same_native_output') is not True
                   for name in policy['final_cases'])):
        raise ValueError('Incomplete or non-identity native final report')
    return True


def generate(checkpoint, image_path, task, maximum, processor, model):
    from PIL import Image
    import torch
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    chat = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': task}]}], add_generation_prompt=True)
    batch = processor(text=chat, images=[[image]], return_tensors='pt')
    tokens = []
    with torch.no_grad():
        features = torch.cat([model.model.get_image_features(
            pixel_values=batch.pixel_values[:, index:index + 1],
            pixel_attention_mask=batch.pixel_attention_mask[:, index:index + 1],
            return_dict=True).pooler_output for index in range(batch.pixel_values.shape[1])], dim=0)
        result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                       image_hidden_states=features, use_cache=True)
        for step in range(maximum):
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            if tokens[-1] == model.generation_config.eos_token_id:
                break
            if step + 1 < maximum:
                result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=False)
    return dict(tokens=tokens, raw=raw,
                stop_reason='eos' if tokens and tokens[-1] == model.generation_config.eos_token_id else 'length',
                tiles=int(batch.pixel_values.shape[1]), prompt_tokens=int(batch.input_ids.shape[1]))


def run(inputs, checkpoint, native_run, selection_root, selected_root, output):
    import torch
    import transformers
    from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
    from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
    policy, manifest = verify_inputs(inputs, selection_root, selected_root)
    validate_policy(policy)
    native_report = json.loads((native_run / 'scores/report.json').read_text())
    validate_native_report(native_report, policy)
    verify_files(native_run, native_report['retained_sha256'])
    model_lock = json.loads(MODEL_LOCK.read_text())
    model_files = {name: info['sha256'] for name, info in model_lock['files'].items()}
    verify_files(checkpoint, model_files)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = json.loads((checkpoint / 'chat_template.json').read_text())['chat_template']
    processor = Idefics3Processor(
        Idefics3ImageProcessorPil.from_pretrained(checkpoint, local_files_only=True), tokenizer,
        image_seq_len=64, chat_template=template)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32,
        attn_implementation='eager').eval()
    cases, all_equal = {}, True
    for case in manifest['cases']:
        name = case['name']
        print('Fresh Python base generation: ' + name, flush=True)
        generated = generate(checkpoint, inputs / case['image'], policy['generation']['task'],
                             policy['generation']['max_new_tokens'], processor, model)
        native, error = load_output(native_run / 'native/base', name,
                                    policy['generation']['max_new_tokens'])
        if error or native is None:
            raise ValueError('Native parity output unavailable: ' + name)
        equal = generated['tokens'] == native['tokens'] and generated['raw'] == native['raw']
        all_equal &= equal
        cases[name] = {**generated, 'matches_native': equal,
                       'native_json_sha256': digest(native_run / f'native/base/{name}.json'),
                       'native_raw_sha256': digest(native_run / f'native/base/{name}.doctags')}
        (output / f'{name}.json').write_text(json.dumps(cases[name], indent=2) + '\n')
        print(f"{name}: {len(generated['tokens'])} tokens; exact native match: {equal}", flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = dict(schema_version=1, selected_id='base', selected_kind='base', adapter_loaded=False,
                  exact_all_native_ids_and_raw=all_equal, cases=cases, device='cpu', dtype='float32',
                  attention='eager', model_revision=model_lock['revision'],
                  native_report_sha256=digest(native_run / 'scores/report.json'),
                  protocol_sha256=digest(POLICY), final_manifest_sha256=digest(inputs / 'manifest.json'),
                  platform=platform.platform(), wall_seconds=time.monotonic() - started,
                  peak_rss_bytes=usage.ru_maxrss,
                  versions={'torch': torch.__version__, 'transformers': transformers.__version__},
                  implementation_sha256=digest(ROOT / 'scripts/book_final_python.py'))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    verify_files(checkpoint, model_files)
    verify_inputs(inputs, selection_root, selected_root)
    if not all_equal:
        raise SystemExit('Fresh Python/native final mismatch; retained every result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('inputs', 'checkpoint', 'native-run', 'selection-root', 'selected-root', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs.resolve(), args.checkpoint.resolve(), args.native_run.resolve(),
        args.selection_root.resolve(), args.selected_root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
