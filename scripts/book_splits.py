"""Fixed book-source reservation and validation-only preparation; never open final content."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from book_pilot import source_id, make_target, reviewed_image, digest
from corpus_preflight import content_fingerprint
from dataset_audit import ROOT, native_probe, verify_file


def assign_splits(rows, lock):
    sources = lock['sources']
    if (len(rows) != 11 or len(sources) != 4 or
            [s['split'] for s in sources] != ['train', 'train', 'validation', 'final'] or
            [s['rows'] for s in sources] != [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10]] or
            len({s['source'] for s in sources}) != 4):
        raise ValueError('Changed fixed source/row partition')
    result, identities = [], set()
    for source in sources:
        url = 'https://archive.org/details/' + source['source']
        identity = source_id(url)
        if len(source['pages']) != len(source['rows']):
            raise ValueError('Missing selected page')
        for index, page in zip(source['rows'], source['pages'], strict=True):
            row = rows[index]
            if (type(page) is not int or page < 1 or row['url'] != url or row['page_number'] != page or
                    not isinstance(row['id'], str) or not row['id']):
                raise ValueError('Changed source/page identity')
            identifier = identity + f'_p{page:05d}'
            if identifier in identities:
                raise ValueError('Duplicate selected page')
            identities.add(identifier)
            result.append(dict(row=index, id=identifier, upstream_id=row['id'], source=source['source'],
                               source_id=identity, page=page, split=source['split']))
    return result


def validation_pages(records):
    result = [r for r in records if r['split'] == 'validation']
    if ([r['row'] for r in result] != [5, 6, 7] or
            any(r['source'] != 'inventorsmechani00peck' for r in result)):
        raise ValueError('Only the fixed validation pages may be opened')
    return result


def check_training_isolation(train, validation):
    """Exact known-source/image/target/content isolation only, not global/perceptual dedup."""
    known = {key: set() for key in ('source_id', 'image_sha256', 'target_sha256', 'content_sha256')}
    for case in train['inventory'] + train['preflight']['cases']:
        for key in known:
            if key in case:
                known[key].add(case[key])
    for case in validation:
        for key, values in known.items():
            if key in case and case[key] in values:
                raise ValueError('Validation/training overlap: ' + key)
    for key in ('id', 'image_sha256', 'target_sha256', 'content_sha256'):
        values = [c[key] for c in validation if key in c]
        if len(values) != len(set(values)):
            raise ValueError('Duplicate validation page: ' + key)
    return True


def reviewed_target(record, review):
    if (record['row'], record['page']) != (review['row'], review['page']):
        raise ValueError('Review page differs from the frozen selection')
    if review['decision'] == 'reviewed-validation':
        return make_target(review['elements'])
    if review['decision'] == 'blocked-illustration' and not review['elements']:
        return None
    raise ValueError('Unknown or incomplete validation review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parquet', 'pdfs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq
    if pyarrow.__version__ != '21.0.0':
        raise ValueError('Use pyarrow 21.0.0')
    lock_path = ROOT / 'references/book-splits.lock.json'
    review_path = ROOT / 'references/book-validation-review.json'
    train_path = ROOT / 'tests/fixtures/book-pilot/report.json'
    lock, review, train = (json.loads(p.read_text()) for p in (lock_path, review_path, train_path))
    files = {}
    for index, split, sha in [(2, 'validation', review['pdf_sha256']), (3, 'final', review['reserved_final_pdf_sha256'])]:
        name = lock['sources'][index]['source'] + '.pdf'
        info = lock[split + '_metadata']
        path = args.pdfs / name
        verify_file(path, sha, info['pdf_bytes'])
        if hashlib.sha1(path.read_bytes()).hexdigest() != info['pdf_sha1']:
            raise ValueError('Original PDF differs from Archive file metadata')
        files[name] = dict(sha256=sha, bytes=info['pdf_bytes'], split=split)
    verify_file(args.parquet, lock['parquet_sha256'], lock['parquet_bytes'])
    # Column projection deliberately excludes HF response text, especially the final rows.
    rows = pq.read_table(args.parquet, columns=['url', 'page_number', 'id']).slice(0, 11).to_pylist()
    inventory = assign_splits(rows, lock)
    selected = validation_pages(inventory)
    if [c['row'] for c in review['cases']] != [5, 6, 7]:
        raise ValueError('Every validation review must be retained once')
    version = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True, timeout=10, check=True)
    if (version.stdout + version.stderr).splitlines()[0] != 'pdftoppm version 26.02.0':
        raise ValueError('Requalify changed renderer explicitly')
    args.output.mkdir(parents=True, exist_ok=False)
    records, paths, accepted = [], [], []
    for case, checked in zip(selected, review['cases'], strict=True):
        name = f'validation-{case["row"]:03d}'
        image = args.output / (name + '.png')
        # No final PDF rendering/text extraction/weak-label access exists in this path.
        subprocess.run(['pdftoppm', '-f', str(case['page']), '-l', str(case['page']), '-singlefile', '-r', '150',
                        '-png', str((args.pdfs / (case['source'] + '.pdf')).resolve()),
                        str((args.output / name).resolve())], capture_output=True, timeout=30, check=True)
        record = dict(case, image=image.name, image_sha256=reviewed_image(image, checked['image_sha256']),
                      decision=checked['decision'], training_approved=False)
        target = reviewed_target(case, checked)
        if target is not None:
            path = args.output / (name + '.doctags')
            path.write_text(target, encoding='utf-8')
            record.update(target=path.name, target_sha256=digest(path), content_sha256=content_fingerprint(target),
                          target_characters=len(target))
            paths.append(path)
            accepted.append(record)
        records.append(record)
    for record, result in zip(accepted, native_probe(paths), strict=True):
        record.update(result)
        if not record['markdown_renderable'] or record['diagnostic_codes']:
            raise ValueError('Reviewed validation target rejected by native parser')
    check_training_isolation(train, records)
    for name, info in files.items():
        verify_file(args.pdfs / name, info['sha256'], info['bytes'])
    verify_file(args.parquet, lock['parquet_sha256'], lock['parquet_bytes'])
    report = dict(schema_version=1, dataset=lock['dataset'], revision=lock['revision'], training_ready=False,
                  validation_complete=False, final_status='reserved-unreviewed',
                  scope='Two reviewed text-only validation references, one retained unsupported illustration; final PDF hashed only, no final pixels/text/labels/model output inspected. No inference or training.',
                  lock_sha256=digest(lock_path), review_sha256=digest(review_path), train_report_sha256=digest(train_path),
                  implementation_sha256={p: digest(ROOT / p) for p in ('scripts/book_splits.py', 'scripts/book_pilot.py',
                       'scripts/corpus_preflight.py', 'scripts/dataset_audit.py', 'scripts/audit-doctags.lisp')},
                  tools=dict(pyarrow=pyarrow.__version__, poppler='26.02.0'), pdfs=files, inventory=inventory,
                  validation=records, known_training_isolation_passed=True,
                  summary=dict(selected_rows=11, sources=4, training_seed_rows=5, validation_rows=3,
                               reviewed_validation_targets=len(accepted), blocked_validation_rows=len(records)-len(accepted),
                               reserved_final_rows=3))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
