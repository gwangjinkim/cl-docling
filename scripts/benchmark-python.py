"""Synchronous CPU FP32/eager Transformers baseline for the frozen workloads."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import PIL
from PIL import Image
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from benchmark_contract import ROOT, CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Explicit benchmark name-to-image JSON mapping')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    load_start = perf_counter()
    template = json.loads((args.checkpoint / 'chat_template.json').read_text())['chat_template']
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    model = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval()
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
    assert model.generation_config.eos_token_id == 49279
    load_seconds = perf_counter() - load_start

    @torch.inference_mode()
    def page(file):
        start = perf_counter()
        with Image.open(file) as image:
            prompt = processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': 'Convert this page to docling.'}]}], add_generation_prompt=True)
            batch = processor(text=prompt, images=[[image.convert('RGB')]], return_tensors='pt')
        t1 = perf_counter()
        features = torch.cat([model.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                              pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                              for i in range(batch.pixel_values.shape[1])], dim=0)
        t2 = perf_counter()
        result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                       image_hidden_states=features, use_cache=True)
        tokens = [int(result.logits[:, -1].argmax(-1).item())]
        cache = result.past_key_values
        del result
        t3 = perf_counter()
        while len(tokens) < 512 and tokens[-1] != 49279:
            result = model(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=cache, use_cache=True)
            tokens.append(int(result.logits[:, -1].argmax(-1).item()))
            cache = result.past_key_values
            del result
        del cache
        t4 = perf_counter()
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        t5 = perf_counter()
        del features
        total = perf_counter() - start
        return (dict(tokens=tokens, raw=raw, stop_reason='eos' if tokens[-1] == 49279 else 'length',
                     prompt_ids=batch.input_ids[0].tolist(), tiles=batch.pixel_values.shape[1]),
                dict(preprocess=t1-start, vision=t2-t1, prefill=t3-t2, decode=t4-t3, detokenize=t5-t4,
                     generation=total, parse_markdown=None, mlx_peak_bytes=None))

    cases = {}
    for name, path in (json.loads(args.cases.read_text()) if args.cases else CASES).items():
        print('Python CPU warm-up: ' + name, flush=True)
        result, warmup = page(ROOT / path)
        samples = []
        for repeat in range(3):
            again, sample = page(ROOT / path)
            if again != result:
                raise ValueError('Non-repeatable Python output')
            samples.append(sample)
            print(f'{name} sample {repeat+1}: {sample["generation"]:.4f} seconds; {len(result["tokens"])} tokens', flush=True)
        cases[name] = {**result, 'warmup': warmup, 'samples': samples}
    report = dict(runtime='python-transformers', device='cpu', dtype='float32', attention='eager',
                  max_new_tokens=512, task='Convert this page to docling.', warmups=1, repeats=3,
                  load_seconds=load_seconds, cases=cases, torch_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(),
                  versions=dict(torch=torch.__version__, transformers=transformers.__version__, pillow=PIL.__version__))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
