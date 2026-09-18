"""Independent FP32 Transformers/PEFT features and initial gradients, never an update."""
import json
from pathlib import Path
import sys

import peft
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import save_file
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

from book_context import TASK


def main():
    inputs, context, checkpoint, output = map(Path, sys.argv[1:])
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / 'references/book-gradients.lock.json').read_text())
    if transformers.__version__ != '5.16.1' or torch.__version__.split('+')[0] != '2.14.0':
        raise ValueError('Use pinned Transformers 5.16.1 / Torch 2.14.0')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = json.loads((checkpoint / 'chat_template.json').read_text())['chat_template']
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    base = Idefics3ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
                         dtype=torch.float32, attn_implementation='eager').eval()
    model = get_peft_model(base, LoraConfig(r=policy['rank'], lora_alpha=policy['alpha'],
                target_modules=r'model.text_model.layers.\d+.self_attn.(q_proj|v_proj)',
                lora_dropout=0.0, bias='none', task_type='CAUSAL_LM')).eval()
    params = {k.replace('.default.', '.'): p for k, p in model.named_parameters() if p.requires_grad}
    if len(params) != policy['gradient_tensors']:
        raise ValueError('Incomplete q/v factor set')
    # Same published LCG initialization contract as the earlier independent page oracle.
    seed = policy['seed']
    with torch.no_grad():
        for name, p in sorted(params.items()):
            if '.lora_A.' in name:
                values = []
                for _ in range(p.numel()):
                    seed = (1664525 * seed + 1013904223) % 2**32
                    fraction = torch.tensor(seed / 2**32, dtype=torch.float32)
                    values.append((2 * fraction - 1) / torch.sqrt(torch.tensor(p.shape[1], dtype=torch.float32)))
                p.copy_(torch.stack(values).reshape_as(p))
            else:
                p.zero_()
    initial = {k: p.detach().clone() for k, p in params.items()}
    save_file(initial, output / 'initial.safetensors')
    prior = {c['name']: c for c in json.loads((context / 'oracle.json').read_text())}
    report = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                  device='cpu', dtype='float32', threads=4, updates=0, cases={})
    for name, expected in policy['cases'].items():
        with Image.open(inputs / (name + '.png')) as source:
            image = source.convert('RGB')
        answer = (inputs / (name + '.doctags')).read_text(encoding='utf-8')
        user = {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': TASK}]}
        transcript = processor.apply_chat_template([user, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                                                   add_generation_prompt=False)
        if not transcript.endswith('<end_of_utterance>\n'):
            raise ValueError('Changed chat policy')
        batch = processor(text=transcript[:-1], images=[[image]], return_tensors='pt')
        image.close()
        ids, mask = batch.input_ids, batch.attention_mask
        labels = ids.clone()
        labels[:, :prior[name]['answer_start']] = -100
        if (ids.tolist()[0] != prior[name]['ids'] or labels.tolist()[0] != prior[name]['labels'] or
                mask.tolist()[0] != prior[name]['attention'] or ids.shape[1] != expected['sequence_tokens'] or
                batch.pixel_values.shape[1] != expected['tiles']):
            raise ValueError('Changed full inputs')
        # No native features enter the oracle; process every PNG tile independently.
        with torch.no_grad():
            features = torch.cat([base.model.get_image_features(pixel_values=batch.pixel_values[:, i:i+1],
                       pixel_attention_mask=batch.pixel_attention_mask[:, i:i+1], return_dict=True).pooler_output
                       for i in range(expected['tiles'])], dim=0).detach()
        model.zero_grad(set_to_none=True)
        result = model(input_ids=ids, image_hidden_states=features, attention_mask=mask, labels=labels, use_cache=False)
        result.loss.backward()
        gradients = {k: p.grad.detach().contiguous() for k, p in params.items()}
        if not all(torch.isfinite(x).all().item() for x in [features, result.loss, *gradients.values()]):
            raise ValueError('Non-finite oracle result')
        if not all(torch.equal(p, initial[k]) for k, p in params.items()):
            raise ValueError('Unexpected adapter mutation')
        save_file(gradients, output / (name + '-gradients.safetensors'))
        save_file({'features': features.reshape(1, -1, features.shape[-1]).contiguous(),
                   'ids': ids.float(), 'labels': labels.float(), 'mask': mask.float()},
                  output / (name + '-inputs.safetensors'))
        report['cases'][name] = dict(loss=result.loss.item(), gradient_tensors=len(gradients), **expected)
        print(name + ': ' + str(report['cases'][name]), flush=True)
        del result, gradients, features, batch
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
