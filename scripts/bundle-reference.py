"""Transport fixture from unchanged recorded generation JSON, not new inference."""
import argparse
import hashlib
import json
from pathlib import Path


def fixture_data(root):
    entries, hashes = [], {}
    for page in (27, 28):
        base = root / 'tests/fixtures/public-pdf/results/native'
        record = json.loads((base / f'page-{page}.json').read_text())
        raw = (base / f'page-{page}.doctags').read_text()
        assert raw == record['raw'] and record['stop_reason'] == 'eos'
        ids = record['tokens']
        assert all(type(i) is int and i >= 0 for i in ids)
        entries.append(f'(:page-number {page} :stop-reason :eos :token-ids #({" ".join(map(str, ids))}))')
        for suffix in ('json', 'doctags'):
            file = base / f'page-{page}.{suffix}'
            hashes[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return '(\n' + '\n'.join(entries) + '\n)\n', hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text, hashes = fixture_data(Path(__file__).resolve().parents[1])
    (args.output / 'generation.sexp').write_text(text, encoding='utf-8')
    (args.output / 'manifest.json').write_text(json.dumps({
        'policy': 'Exact token/stop metadata from recorded M6.9 JSON; no model run.',
        'source_sha256': hashes,
        'generation_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
