"""Fresh ordinary Transformers/PEFT CPU check of all frozen coverage outputs."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from peft import PeftModel
import torch
import transformers
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','adapter','native','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    folder=ROOT/'tests/fixtures/coverage'; manifest=json.loads((folder/'manifest.json').read_text())
    lock=json.loads((ROOT/'references/smoldocling.lock.json').read_text())
    for path,files in ((folder,manifest['files']),(args.adapter,manifest['adapter_sha256']),
                       (args.checkpoint,{n:v['sha256'] for n,v in lock['files'].items()})):
        for name,wanted in files.items():
            assert hashlib.sha256((path/name).read_bytes()).hexdigest()==wanted,name
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    processor=Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(args.checkpoint,local_files_only=True),tokenizer,
        image_seq_len=64,chat_template=json.loads((args.checkpoint/'chat_template.json').read_text())['chat_template'])
    base=Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    all_equal=True
    for phase in ('base','adapted'):
        model=base if phase=='base' else PeftModel.from_pretrained(base,args.adapter,local_files_only=True).eval()
        (args.output/phase).mkdir()
        for case in json.loads((folder/'final.json').read_text())['cases']:
            name=case['name']; print(phase,name,flush=True)
            with Image.open(folder/(name+'.png')) as image:
                chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'}, {'type':'text','text':manifest['protocol']['task']}]}],add_generation_prompt=True)
                batch=processor(text=chat,images=[[image.convert('RGB')]],return_tensors='pt')
            tokens=[]
            with torch.inference_mode():
                features=torch.cat([base.model.get_image_features(pixel_values=batch.pixel_values[:,i:i+1],pixel_attention_mask=batch.pixel_attention_mask[:,i:i+1],return_dict=True).pooler_output for i in range(batch.pixel_values.shape[1])],dim=0)
                result=model(input_ids=batch.input_ids,attention_mask=batch.attention_mask,image_hidden_states=features,use_cache=True)
                for step in range(manifest['protocol']['max_new_tokens']):
                    token=result.logits[:,-1].argmax(-1,keepdim=True); tokens.append(int(token.item()))
                    if tokens[-1]==49279: break
                    if step+1<manifest['protocol']['max_new_tokens']:
                        result=model(input_ids=token,past_key_values=result.past_key_values,use_cache=True)
            raw=tokenizer.decode(tokens,skip_special_tokens=False)
            reason='eos' if tokens[-1]==49279 else 'length'
            native=json.loads((args.native/phase/(name+'.json')).read_text())
            match=tokens==native['tokens'] and raw==native['raw'] and reason==native['stop_reason']; all_equal &= match
            report=dict(tokens=tokens,raw=raw,stop_reason=reason,matches_native=match,tiles=batch.pixel_values.shape[1],prompt_tokens=batch.input_ids.shape[1],device='cpu',dtype='float32',torch=torch.__version__,transformers=transformers.__version__)
            (args.output/phase/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
            print(f'{len(tokens)} tokens; exact native match: {match}',flush=True)
    if not all_equal: raise SystemExit('Native/Python differences retained; no tolerance changed')


if __name__=='__main__': main()
