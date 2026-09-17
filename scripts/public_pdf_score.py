"""Frozen table-only case study; not a whole-page or semantic-math metric."""
from collections import Counter
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from table_score import score


def whitespace(text):
    return ' '.join(text.split())


def word_inventory(expected, actual):
    # Ignore reading order, but preserve case/punctuation and duplicate counts.
    wanted, found = Counter(expected.split()), Counter(actual.split())
    matched = sum((wanted & found).values())
    return dict(matched=matched, expected=sum(wanted.values()),
                recall=matched / sum(wanted.values()) if wanted else 1)


def score_page(rows, result):
    expected = dict(rows=len(rows), columns=len(rows[0]),
                    cells=[[r, c, 1, 1, whitespace(text)]
                           for r, row in enumerate(rows) for c, text in enumerate(row)])
    normalized = copy.deepcopy(result)
    for table in normalized['tables']:
        for cell in table['cells']:
            cell[4] = whitespace(cell[4])
    report = score(expected, normalized)
    del report['accepted']  # Earlier synthetic scorer disallows the real page margins.
    report['table_accepted'] = (report['exact_table'] and result['stop_reason'] == 'eos'
                               and not result['diagnostics'] and result['markdown_written'])
    report['raw_word_inventory'] = word_inventory(' '.join(c[4] for c in expected['cells']),
                                                 re.sub(r'<[^>]*>', ' ', result['raw']))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ('native', 'python', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixtures = root / 'tests/fixtures/public-pdf'
    protocol = json.loads((fixtures / 'protocol.json').read_text())
    for name, wanted in protocol['files'].items():
        if hashlib.sha256((fixtures / name).read_bytes()).hexdigest() != wanted:
            raise ValueError('Frozen reference changed: ' + name)
    rows = json.loads((fixtures / 'expected.json').read_text())
    report = {}
    for name in rows:
        native = json.loads((args.native / 'base' / (name + '.json')).read_text())
        reference = json.loads((args.python / (name + '.json')).read_text())
        raster = args.native / 'raster' / (name + '.png')
        if hashlib.sha256(raster.read_bytes()).hexdigest() != protocol['files'][name + '.png']:
            raise ValueError('Native raster differs from frozen direct Poppler raster')
        result = score_page(rows[name], native)
        result['python_equal'] = all(native[k] == reference[k] for k in ('tokens', 'raw', 'stop_reason'))
        result['output_tokens'] = len(native['tokens'])
        result['text_layer_word_inventory'] = word_inventory(
            ' '.join(text for row in rows[name] for text in row),
            (fixtures / (name + '-text-layer.txt')).read_text())
        report[name] = result
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({n: {k: v for k, v in r.items() if k in
          ('exact_cell_matches', 'expected_cells', 'table_accepted', 'python_equal',
           'raw_word_inventory', 'text_layer_word_inventory')} for n, r in report.items()}, indent=2))


if __name__ == '__main__':
    main()
