"""Offline geometry experiment on the existing development-only 20-page audit.

Input frame is an explicit hypothesis backed by sample inspection, not a general
DocAtlas-format guarantee. Never use this report alone to approve training data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from dataset_audit import ROOT, select_rows, verify_file, native_probe
from dataset_geometry import convert_pixel_locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--convention', required=True, choices=['top-left-pixels'])
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq
    import PIL
    from PIL import Image
    if pyarrow.__version__ != '21.0.0' or PIL.__version__ != '12.3.0':
        raise ValueError('Expected pyarrow 21.0.0 and pillow 12.3.0')
    lock_path = ROOT/'references/dataset-audit.lock.json'
    lock = json.loads(lock_path.read_text())
    verify_file(args.parquet, lock['sha256'], lock['bytes'])
    old_path = ROOT/'tests/fixtures/dataset-audit/report.json'
    old = json.loads(old_path.read_text())
    selected = select_rows(pq.read_table(args.parquet, columns=['id', 'language', 'doctags']).to_pylist(), 20)
    if [(i, r['id']) for i, r in selected] != [(r['shard_row'], r['id']) for r in old['cases']]:
        raise ValueError('Audit sample identity differs')
    by_index = {index: (ordinal, row) for ordinal, (index, row) in enumerate(selected)}
    args.output.mkdir(parents=True, exist_ok=False)
    cases, paths, offset = [], [], 0
    for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=32, columns=['image']):
        for local_index, value in enumerate(batch.column(0)):
            index = offset + local_index
            if index not in by_index:
                continue
            ordinal, row = by_index[index]
            data = value.as_py()['bytes']
            if not isinstance(data, bytes) or len(data) > 20_000_000:
                raise ValueError('Missing or oversized image bytes')
            image = Image.open(io.BytesIO(data))
            if image.format != 'PNG':
                raise ValueError('Audit scope is original PNG bytes, no transcoding')
            width, height = image.size
            image.verify()
            raw = row['doctags']
            original_hash = hashlib.sha256(raw.encode('utf-8')).hexdigest()
            if original_hash != old['cases'][ordinal]['target_sha256']:
                raise ValueError('Original target hash differs')
            normalized = convert_pixel_locations(raw, width, height, convention=args.convention)
            strip = lambda text: re.sub(r'<loc_[0-9]+>', '', text)
            if strip(raw) != strip(normalized):
                raise ValueError('Noncoordinate content changed')
            prefix = args.output/f'{ordinal:03d}'
            prefix.with_suffix('.png').write_bytes(data)
            prefix.with_suffix('.original.doctags').write_text(raw, encoding='utf-8')
            path = prefix.with_suffix('.normalized.doctags')
            path.write_text(normalized, encoding='utf-8')
            paths.append(path)
            cases.append(dict(ordinal=ordinal, shard_row=index, id=row['id'],
                              image_size=[width, height], image_sha256=hashlib.sha256(data).hexdigest(),
                              original_target_sha256=original_hash,
                              normalized_target_sha256=hashlib.sha256(normalized.encode('utf-8')).hexdigest(),
                              noncoordinate_sha256=hashlib.sha256(strip(raw).encode('utf-8')).hexdigest(),
                              box_count=len(re.findall(r'(?:<loc_[0-9]+>){4}', raw))))
        offset += len(batch)
        if len(cases) == 20:
            break
    if len(cases) != 20:
        raise ValueError('Incomplete image audit')
    for case, result in zip(cases, native_probe(paths), strict=True):
        case.update(result)
    report = dict(schema_version=1, dataset=lock['dataset'], revision=lock['revision'],
                  shard_sha256=lock['sha256'], prior_audit_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                  convention=args.convention,
                  coordinate_evidence='All 20 page bounds checked; original portrait row 6 and landscape row 48 visually spot-checked. Sample-level evidence, not a global serialization guarantee.',
                  training_approved=False,
                  privacy_review='A sampled source appears to contain a filled health questionnaire. No training/publication approval; manual source review required. No sensitive text in this report.',
                  tools={'pyarrow': pyarrow.__version__, 'pillow': PIL.__version__},
                  implementation_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (Path(__file__), ROOT/'scripts/dataset_geometry.py', ROOT/'scripts/audit-doctags.lisp')},
                  summary=dict(pages=20, pixel_boxes=sum(c['box_count'] for c in cases),
                               native_renderable=sum(c['markdown_renderable'] for c in cases),
                               remaining_location_diagnostics=sum('location' in c['diagnostic_codes'] for c in cases)),
                  cases=cases)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
