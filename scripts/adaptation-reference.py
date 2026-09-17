"""Fresh ordinary Transformers/PEFT reload; all held-out greedy outputs retained."""
import argparse
import json
from pathlib import Path
import peft
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from adaptation_data import ROOT, validate_manifest
import hashlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-experiment', action='store_true', help='Verify the M5.7 selected model on new final-test pages')
    args = parser.parse_args()
    fixtures = ROOT / 'tests/fixtures' / ('selection-pages' if args.selection_experiment else 'adaptation-pages')
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    selected_base = False
    if args.selection_experiment:
        from selection_data import verify_manifest
        verify_manifest(manifest)
        selection = json.loads((args.native.parent / 'selection.json').read_text())
        selected_base = selection['selected_step'] == 0
        for name, wanted in selection['adapter_sha256'].items():
            assert hashlib.sha256((args.native / 'adapter' / name).read_bytes()).hexdigest() == wanted
    else:
        validate_manifest(manifest)
    protocol = manifest['protocol']
    lock = json.loads((ROOT / 'references/smoldocling.lock.json').read_text())
    for folder, files in ((args.checkpoint, lock['files']), (fixtures, manifest['files'])):
        for name, entry in files.items():
            wanted = entry if isinstance(entry, str) else entry['sha256']
            with (folder / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == wanted, name
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    template = json.loads((args.checkpoint / 'chat_template.json').read_text())['chat_template']
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    base = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                  dtype=torch.float32, attn_implementation='eager').eval()
    model = base if selected_base else PeftModel.from_pretrained(base, args.native / 'adapter', local_files_only=True).eval()
    factor_count = 0
    if not selected_base:
        saved = load_file(args.native / 'adapter/adapter_model.safetensors')
        loaded = {name.replace('.default.', '.'): p for name, p in model.named_parameters() if '.lora_' in name}
        assert saved.keys() == loaded.keys() and len(saved) == 120
        assert all(torch.equal(tensor, loaded[name]) for name, tensor in saved.items())
        factor_count = len(saved)
    all_equal = True
    for case in manifest['cases']:
        if case['split'] != ('final' if args.selection_experiment else 'heldout'):
            continue
        name = case['name']
        print('Python adapter generation: ' + name, flush=True)
        with Image.open(fixtures / f'{name}.png') as image:
            chat = processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': protocol['task']}]}], add_generation_prompt=True)
            batch = processor(text=chat, images=[[image.convert('RGB')]], return_tensors='pt')
        tokens = []
        with torch.no_grad():
            features = torch.cat([base.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                                  pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                                  for i in range(batch.pixel_values.shape[1])], dim=0)
            result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           image_hidden_states=features, use_cache=True)
            for step in range(protocol['max_new_tokens']):
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                if tokens[-1] == base.generation_config.eos_token_id:
                    break
                if step + 1 < protocol['max_new_tokens']:
                    result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        native = json.loads((args.native / 'adapted' / f'{name}.json').read_text())
        equal = tokens == native['tokens'] and raw == native['raw']
        all_equal &= equal
        report = dict(case=name, tokens=tokens, raw=raw, matches_native=equal, exact_adapter_factors=factor_count,
                      device='cpu', dtype='float32', attention='eager', model_revision=lock['revision'],
                      stop_reason='eos' if tokens[-1] == base.generation_config.eos_token_id else 'length',
                      tiles=batch.pixel_values.shape[1], prompt_tokens=batch.input_ids.shape[1],
                      versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__))
        (args.output / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{name}: {len(tokens)} tokens; exact native match: {equal}', flush=True)
    if not all_equal:
        raise SystemExit('Native/Python mismatch; retained all held-out references')


if __name__ == '__main__':
    main()
