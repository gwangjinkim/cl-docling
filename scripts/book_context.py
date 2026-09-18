"""Full native input accounting with an independent Transformers oracle; no forward/update."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

from replay_dpbench import verify_files, digest

ROOT = Path(__file__).resolve().parents[1]
TASK = 'Convert this page to docling.'


def complete_cases(names, records):
    result = {r['name']: r for r in records}
    if len(result) != len(records) or set(result) != set(names):
        raise ValueError('Missing, duplicate or extra input case')
    return result


def summarize_case(native, oracle):
    keys = ('ids', 'labels', 'attention', 'answer_start', 'tiles', 'pixel_shape',
            'image_token_id', 'eos_token_id', 'image_seq_len', 'context_limit')
    if any(native[k] != oracle[k] for k in keys):
        raise ValueError('Native/Python full-input mismatch')
    ids, labels, mask = (native[k] for k in ('ids', 'labels', 'attention'))
    n, start = len(ids), native['answer_start']
    tiles, limit, slots = (native[k] for k in ('tiles', 'context_limit', 'image_seq_len'))
    if (any(type(x) is not int for x in [start, tiles, limit, slots, *ids, *labels, *mask]) or
            not 1 <= start < n <= limit <= 8192 or tiles < 1 or slots < 1 or
            len(labels) != n or len(mask) != n or mask != [1] * n or
            any(not 0 <= x < 2**31 for x in ids) or
            labels != [-100] * start + ids[start:] or
            ids[-1] != native['eos_token_id'] or native['eos_token_id'] in ids[start:-1] or
            native['image_token_id'] in ids[start:] or
            ids[:start].count(native['image_token_id']) != tiles * slots):
        raise ValueError('Invalid full unpadded answer-only supervision')
    shape = native['pixel_shape']
    if (len(shape) != 5 or shape[:3] != [1, tiles, 3] or
            any(type(x) is not int or x < 1 for x in shape)):
        raise ValueError('Invalid all-tile FP32 pixel shape')
    hashes = {key + '_int32_le_sha256': hashlib.sha256(struct.pack('<' + 'i' * n, *native[key])).hexdigest()
              for key in ('ids', 'labels', 'attention')}
    return dict(sequence_tokens=n, prompt_tokens=start, supervised_tokens=n-start,
                image_tokens=tiles * slots, other_prompt_tokens=start-tiles * slots,
                tiles=tiles, pixel_shape=shape, context_limit=limit, context_remaining=limit-n,
                pixel_payload_bytes=4 * math.prod(shape), ids_labels_payload_bytes=8*n,
                attention_payload_bits=n, native_python_exact=True, **hashes)


def reference(checkpoint, inputs, cases):
    # Only the optional verification driver uses Python; the Lisp path never calls it.
    import numpy as np
    from PIL import Image
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
    if transformers.__version__ != '5.16.1':
        raise ValueError('Use the pinned Transformers 5.16.1 oracle')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = json.loads((checkpoint / 'chat_template.json').read_text())['chat_template']
    processor = Idefics3Processor(Idefics3ImageProcessorPil.from_pretrained(checkpoint, local_files_only=True),
                                 tokenizer, image_seq_len=64, chat_template=template)
    config = json.loads((checkpoint / 'config.json').read_text())
    records = []
    for case in cases:
        with Image.open(inputs / case['image']) as source:
            image = source.convert('RGB')
        answer = (inputs / case['target']).read_text(encoding='utf-8')
        user = {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': TASK}]}
        prompt = processor.apply_chat_template([user], add_generation_prompt=True)
        full = processor.apply_chat_template([user, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                                             add_generation_prompt=False)
        if not full.endswith('<end_of_utterance>\n'):
            raise ValueError('Unqualified assistant template')
        prefix = processor(text=prompt, images=[[image]], return_tensors='np')['input_ids'][0]
        batch = processor(text=full[:-1], images=[[image]], return_tensors='np')
        ids = batch['input_ids'][0]
        start = len(prefix)
        if not np.array_equal(prefix, ids[:start]):
            raise ValueError('Ambiguous oracle answer boundary')
        records.append(dict(name=Path(case['image']).stem, ids=ids.tolist(),
                            labels=[-100]*start + ids[start:].tolist(), attention=batch['attention_mask'][0].tolist(),
                            answer_start=start, tiles=batch['pixel_values'].shape[1],
                            pixel_shape=list(batch['pixel_values'].shape),
                            image_token_id=tokenizer.convert_tokens_to_ids('<image>'),
                            eos_token_id=tokenizer.convert_tokens_to_ids('<end_of_utterance>'), image_seq_len=64,
                            context_limit=min(8192, config['text_config']['max_position_embeddings'])))
        image.close()
    return records, dict(transformers=transformers.__version__, numpy=np.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    inputs, checkpoint, output = (p.resolve() for p in (args.inputs, args.checkpoint, args.output))
    frozen_path = ROOT / 'tests/fixtures/book-pilot/report.json'
    frozen = json.loads(frozen_path.read_text())
    model_lock_path = ROOT / 'references/smoldocling.lock.json'
    lock = json.loads(model_lock_path.read_text())
    files = {'report.json': digest(frozen_path)}
    cases = [r for r in frozen['inventory'] if r['decision'] == 'approved']
    for case in cases:
        files[case['image']] = case['image_sha256']
        files[case['target']] = case['target_sha256']
    model_files = {name: info['sha256'] for name, info in lock['files'].items()}
    verify_files(inputs, files)
    verify_files(checkpoint, model_files)
    output.mkdir(parents=True, exist_ok=False)
    oracle, versions = reference(checkpoint, inputs, cases)
    (output / 'oracle.json').write_text(json.dumps(oracle) + '\n')
    command = ['sbcl', '--dynamic-space-size', '4096', '--noinform', '--no-sysinit', '--no-userinit',
               '--script', 'scripts/book-context.lisp', str(inputs), str(output / 'native.json')]
    env = {**os.environ, 'DOCLING_MODEL': str(checkpoint)}
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=240)
    (output / 'native.log').write_text(completed.stdout + completed.stderr)
    completed.check_returncode()
    native = json.loads((output / 'native.json').read_text())
    if native['remaining_handles'] != 0:
        raise ValueError('Native model handles remain')
    names = [Path(c['image']).stem for c in cases]
    native_cases = complete_cases(names, native['cases'])
    oracle_cases = complete_cases(names, oracle)
    results = {name: summarize_case(native_cases[name], oracle_cases[name]) for name in names}
    verify_files(inputs, files)
    verify_files(checkpoint, model_files)
    report = dict(schema_version=1, training_ready=False, context_passed=True,
                  scope='Full input preparation only. No model forward, gradient, update or total training-memory qualification.',
                  task=TASK, device='cpu', dtype='float32', remaining_handles=native['remaining_handles'],
                  book_report_sha256=digest(frozen_path), model_revision=lock['revision'],
                  model_lock_sha256=digest(model_lock_path), input_sha256=files,
                  versions={**versions, 'sbcl': native['sbcl']},
                  implementation_sha256={p: digest(ROOT / p) for p in ('scripts/book_context.py',
                      'scripts/book-context.lisp', 'src/training-inputs.lisp', 'src/supervision.lisp')},
                  retained_sha256={p.name: digest(p) for p in sorted(output.iterdir())}, cases=results)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
