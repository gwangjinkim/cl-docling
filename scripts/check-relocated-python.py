"""Independent ordinary Transformers/PEFT consumer of the relocated adapter."""
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
from adapter_portability import adapter_inventory, base_inventory, read_json, write_json
from adapted_benchmark_contract import CASES, ROOT, validate_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'adapter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = read_json(args.adapter / 'relocation.json')
    assert base_inventory(args.base) == receipt['base_files']
    assert adapter_inventory(args.adapter) == receipt['staged_adapter_files']
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    processor = Idefics3Processor(
        Idefics3ImageProcessorPil.from_pretrained(args.base, local_files_only=True), tokenizer,
        image_seq_len=64, chat_template=read_json(args.base / 'chat_template.json')['chat_template'])
    base = Idefics3ForConditionalGeneration.from_pretrained(args.base, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval()
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).eval()
    saved = load_file(args.adapter / 'adapter_model.safetensors')
    loaded = {name.replace('.default.', '.'): p for name, p in model.named_parameters() if '.lora_' in name}
    assert saved.keys() == loaded.keys() and len(saved) == 120
    assert all(torch.equal(tensor, loaded[name]) for name, tensor in saved.items())
    cases = {}
    for name, path in CASES.items():
        with Image.open(ROOT / path) as image:
            chat = processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': 'Convert this page to docling.'}]}],
                add_generation_prompt=True)
            batch = processor(text=chat, images=[[image.convert('RGB')]], return_tensors='pt')
        tokens = []
        with torch.inference_mode():
            features = torch.cat([base.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                for i in range(batch.pixel_values.shape[1])], dim=0)
            result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           image_hidden_states=features, use_cache=True)
            for step in range(512):
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                if tokens[-1] == base.generation_config.eos_token_id:
                    break
                if step + 1 < 512:
                    result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        cases[name] = dict(tokens=tokens, raw=tokenizer.decode(tokens, skip_special_tokens=False),
                          stop_reason='eos' if tokens[-1] == base.generation_config.eos_token_id else 'length',
                          tiles=batch.pixel_values.shape[1])
        print(f'Python relocated {name}: {len(tokens)} tokens', flush=True)
    validate_selected(cases)
    write_json(args.output, dict(cases=cases, exact_adapter_factors=len(saved), device='cpu', dtype='float32',
        all_selected_outputs_equal=True,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)))


if __name__ == '__main__':
    main()
