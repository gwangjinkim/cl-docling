"""Fresh ordinary Transformers CPU reference for the frozen native PDF rasters."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from adapter_portability import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read_json(ROOT / 'tests/fixtures/public-pdf/protocol.json')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
        tokenizer, image_seq_len=64, chat_template=read_json(args.checkpoint / 'chat_template.json')['chat_template'])
    model = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval()
    equal = True
    for case in protocol['cases']:
        name = case['name']
        with Image.open(args.native / 'raster' / (name + '.png')) as image:
            chat = processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': protocol['generation']['task']}]}], add_generation_prompt=True)
            batch = processor(text=chat, images=[[image.convert('RGB')]], return_tensors='pt')
        tokens = []
        with torch.inference_mode():
            features = torch.cat([model.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                for i in range(batch.pixel_values.shape[1])], dim=0)
            result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           image_hidden_states=features, use_cache=True)
            for step in range(protocol['generation']['max_new_tokens']):
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                if tokens[-1] == model.generation_config.eos_token_id:
                    break
                if step + 1 < protocol['generation']['max_new_tokens']:
                    result = model(input_ids=token, past_key_values=result.past_key_values, use_cache=True)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        native = read_json(args.native / 'base' / (name + '.json'))
        match = tokens == native['tokens'] and raw == native['raw']
        equal &= match
        write_json(args.output / (name + '.json'), dict(tokens=tokens, raw=raw, matches_native=match,
            stop_reason='eos' if tokens[-1] == model.generation_config.eos_token_id else 'length',
            tiles=batch.pixel_values.shape[1], prompt_tokens=batch.input_ids.shape[1], device='cpu',
            torch_version=torch.__version__, transformers_version=transformers.__version__))
        print(f'{name}: {len(tokens)} tokens; native match {match}', flush=True)
    if not equal:
        raise SystemExit('Native/Python mismatch; all references retained')


if __name__ == '__main__':
    main()
