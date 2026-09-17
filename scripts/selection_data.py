"""Frozen M5.7 sources and validation-only selection; no model dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from adaptation_data import ROOT, PROTOCOL as PREVIOUS, cases as old_cases, merged, occupancy, answer

STEPS = (0, 16, 48, 96)
PROTOCOL = {**PREVIOUS, 'epochs': 24, 'learning_rate': 0.0005,
            'candidate_epochs': [4, 12, 24],
            'selection': 'max(accepted pages, exact cell matches, negative step); validation only'}


def cases():
    data = old_cases()
    for case in data:
        if case['split'] == 'heldout':
            case['split'] = 'validation'
    for name, words, box, font in [
        ('library', ['Library stock', 'Count', 'Books', 'Novels', '7', 'Atlases', '2', 'Media', 'Films', '4'],
         [30, 40, 180, 70], 24),
        ('pets', ['Pet stock', 'Count', 'Food', 'Pellets', '6', 'Flakes', '3', 'Toys', 'Ropes', '8'],
         [48, 52, 164, 64], 22),
    ]:
        data.append(dict(name=name, source=name, split='final', expected=merged(words), box=box, font=font))
    for name, rows in [
        ('fruit', [['Item', 'Count', 'Price'], ['Apples', '6', '2'], ['Pears', '3', '4'], ['Plums', '8', '1']]),
        ('tea', [['Item', 'Count', 'Price'], ['Green tea', '5', '7'], ['Black tea', '9', '4'], ['Mint tea', '3', '6']]),
    ]:
        data.append(dict(name=name, source=name, split='final', box=[30, 40, 180, 70], font=24,
                         expected=dict(rows=4, columns=3, cells=[
                             [r, c, 1, 1, text] for r, row in enumerate(rows) for c, text in enumerate(row)])))
    return data


def validate(data):
    for field in ('name', 'source'):
        if len({case[field] for case in data}) != len(data):
            raise ValueError('Duplicate ' + field)
    if len({json.dumps(c['expected'], sort_keys=True) for c in data}) != len(data):
        raise ValueError('Duplicate source document')
    for case in data:
        occupancy(case['expected'])
        if case['split'] not in ('train', 'validation', 'final', 'regression'):
            raise ValueError('Unknown split')


def choose_candidate(reports):
    names = {c['name'] for c in cases() if c['split'] == 'validation'}
    if set(reports) != set(STEPS) or any(set(scores) != names for scores in reports.values()):
        raise ValueError('Selection requires every frozen candidate and only validation pages')
    return max(STEPS, key=lambda step: (
        sum(s['accepted'] for s in reports[step].values()),
        sum(s['exact_cell_matches'] for s in reports[step].values()), -step))


def entries(data):
    result = []
    for case in data:
        entry = {key: case[key] for key in ('name', 'source', 'split')}
        if case['split'] == 'train':
            entry['answer'] = answer(case['expected'])
        result.append(entry)
    return result


def verify_manifest(manifest):
    data = cases()
    validate(data)
    expected_files = {f"{c['name']}.{ext}" for c in data for ext in ('png', 'json')}
    expected_files |= {'development.json', 'final.json'}
    if (manifest['cases'] != entries(data) or manifest['protocol'] != PROTOCOL
            or set(manifest['files']) != expected_files):
        raise ValueError('Frozen corpus/protocol changed')


def main():
    from PIL import Image, ImageDraw, ImageFont, __version__
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if __version__ != '12.3.0':
        raise ValueError('Use pinned Pillow 12.3.0')
    data = cases()
    validate(data)
    args.output.mkdir(parents=True, exist_ok=False)
    for case in data:
        name = case['name']
        if case['split'] != 'final':
            for ext in ('png', 'json'):
                shutil.copyfile(ROOT / 'tests/fixtures/adaptation-pages' / f'{name}.{ext}', args.output / f'{name}.{ext}')
            continue
        image = Image.new('RGB', (600, 360), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=case['font'])
        left, top, width, height = case['box']
        for r, c, rs, cs, text in case['expected']['cells']:
            box = (left+width*c, top+height*r, left+width*(c+cs), top+height*(r+rs))
            draw.rectangle(box, outline='black', width=2)
            draw.text((box[0]+12, (box[1]+box[3])/2), text, font=font, fill='black', anchor='lm')
        image.save(args.output / f'{name}.png')
        (args.output / f'{name}.json').write_text(json.dumps(case['expected'], indent=2) + '\n')
    for name, splits in [('development', ('train', 'validation')), ('final', ('final', 'regression'))]:
        payload = dict(protocol=PROTOCOL, cases=entries([c for c in data if c['split'] in splits]))
        (args.output / f'{name}.json').write_text(json.dumps(payload, indent=2) + '\n')
    manifest = dict(protocol=PROTOCOL, cases=entries(data), license='MIT; project-authored synthetic pages',
                    pillow=__version__, limitation='Shared template family, not broad document quality',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
