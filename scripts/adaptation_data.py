"""Predetermined synthetic adaptation corpus; no model imports or model labels."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(epochs=4, rank=2, alpha=4, seed=17, optimizer='adamw',
                learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                weight_decay=0.01, max_grad_norm=1.0, max_new_tokens=512,
                task='Convert this page to docling.')


def merged(words):
    geometry = [(0, 0, 1, 2), (0, 2, 1, 1), (1, 0, 2, 1), (1, 1, 1, 1),
                (1, 2, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1), (3, 0, 1, 1),
                (3, 1, 1, 1), (3, 2, 1, 1)]
    return dict(rows=4, columns=3, cells=[[*g, text] for g, text in zip(geometry, words, strict=True)])


def cases():
    specs = [
        ('garden', 'train', ['Garden stock', 'Count', 'Tools', 'Spades', '4', 'Rakes', '2', 'Seeds', 'Beans', '7']),
        ('kitchen', 'train', ['Kitchen stock', 'Count', 'Dishes', 'Plates', '6', 'Bowls', '3', 'Cutlery', 'Forks', '9']),
        ('workshop', 'train', ['Workshop stock', 'Count', 'Tools', 'Drills', '2', 'Saws', '5', 'Parts', 'Nails', '8']),
        ('art', 'train', ['Art stock', 'Count', 'Colors', 'Paints', '3', 'Inks', '4', 'Tools', 'Brushes', '6']),
        ('sports', 'heldout', ['Sports stock', 'Count', 'Games', 'Balls', '8', 'Bats', '3', 'Clothes', 'Shirts', '5']),
        ('music', 'heldout', ['Music stock', 'Count', 'Strings', 'Violins', '2', 'Guitars', '4', 'Wind', 'Flutes', '6']),
    ]
    result = [dict(name=n, source=n, split=s, expected=merged(w),
                   box=[30, 40, 180, 70], font=24) for n, s, w in specs]
    # Held-out layout shift, not a transformed copy of a training source.
    result[-1].update(box=[48, 52, 164, 64], font=22)
    result.append(dict(name='bakery', source='bakery', split='heldout',
                       box=[30, 40, 180, 70], font=24,
                       expected=dict(rows=4, columns=3, cells=[
                           [r, c, 1, 1, text] for r, row in enumerate([
                               ['Item', 'Count', 'Price'], ['Bread', '4', '3'],
                               ['Rolls', '8', '1'], ['Cakes', '2', '9']])
                           for c, text in enumerate(row)])))
    for name in ('grid', 'merged'):
        result.append(dict(name=name, source='old-' + name, split='regression',
                           box=[30, 40, 180, 70], font=24,
                           expected=json.loads((ROOT / 'tests/fixtures/table-pages' / f'{name}.json').read_text())))
    return result


def occupancy(expected):
    occupied = {}
    for cell in expected['cells']:
        r, c, rs, cs, text = cell
        if rs < 1 or cs < 1 or not text or '<' in text or '>' in text:
            raise ValueError('Invalid cell')
        for y in range(r, r + rs):
            for x in range(c, c + cs):
                if not (0 <= y < expected['rows'] and 0 <= x < expected['columns']) or (y, x) in occupied:
                    raise ValueError('Out of range or overlapping cells')
                occupied[y, x] = cell
    if len(occupied) != expected['rows'] * expected['columns']:
        raise ValueError('Missing cell coverage')
    return occupied


def answer(expected):
    grid = occupancy(expected)
    parts = ['<doctag><otsl>']
    for y in range(expected['rows']):
        for x in range(expected['columns']):
            r, c, _, _, text = grid[y, x]
            parts.append(('<ched>' if y == 0 else '<fcel>') + text if (y, x) == (r, c)
                         else '<lcel>' if y == r else '<ucel>' if x == c else '<xcel>')
        parts.append('<nl>')
    return ''.join(parts) + '</otsl></doctag>'


def validate(data):
    for field in ('name', 'source'):
        if len({c[field] for c in data}) != len(data):
            raise ValueError('Repeated ' + field)
    targets = [json.dumps(c['expected'], sort_keys=True) for c in data]
    if len(set(targets)) != len(targets):
        raise ValueError('Repeated document')
    for case in data:
        occupancy(case['expected'])
        if case['split'] not in ('train', 'heldout', 'regression'):
            raise ValueError('Unknown split')


def validate_manifest(manifest):
    data = cases()
    validate(data)
    expected = []
    for case in data:
        entry = {key: case[key] for key in ('name', 'source', 'split')}
        if case['split'] == 'train':
            entry['answer'] = answer(case['expected'])
        expected.append(entry)
    files = {f"{case['name']}.{extension}" for case in data for extension in ('png', 'json')}
    if manifest['protocol'] != PROTOCOL or manifest['cases'] != expected or set(manifest['files']) != files:
        raise ValueError('Frozen protocol, split, supervision, or file inventory changed')


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
    entries = []
    for case in data:
        name = case['name']
        im = Image.new('RGB', (600, 360), 'white')
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=case['font'])
        left, top, width, height = case['box']
        for r, c, rs, cs, text in case['expected']['cells']:
            box = (left + width*c, top + height*r, left + width*(c+cs), top + height*(r+rs))
            draw.rectangle(box, outline='black', width=2)
            draw.text((box[0]+12, (box[1]+box[3])/2), text, font=font, fill='black', anchor='lm')
        im.save(args.output / f'{name}.png')
        (args.output / f'{name}.json').write_text(json.dumps(case['expected'], indent=2) + '\n')
        entry = {k: case[k] for k in ('name', 'source', 'split')}
        # The runner never receives held-out or regression answers.
        if case['split'] == 'train':
            entry['answer'] = answer(case['expected'])
        entries.append(entry)
    manifest = dict(protocol=PROTOCOL, cases=entries, license='MIT; project-authored synthetic pages',
                    pillow=__version__, font='Pillow bundled Aileron',
                    limitation='Shared synthetic table template, not unseen-layout or real-world qualification',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
