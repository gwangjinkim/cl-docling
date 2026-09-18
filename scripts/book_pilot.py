"""Prepare five reviewed historical-book pages; no inference or training.

All downloads are explicit prior steps. Original HF responses remain weak labels;
only separately reviewed targets can enter the three-page preflight.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from corpus_preflight import content_fingerprint, evaluate_candidates
from dataset_audit import ROOT, native_probe, verify_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_image(path, wanted):
    actual = digest(path)
    if actual != wanted:
        raise ValueError('Rendered image differs from the visually reviewed bytes')
    return actual


def source_id(url):
    if not isinstance(url, str) or not re.fullmatch(r'https://archive.org/details/[a-zA-Z0-9_\-]+', url):
        raise ValueError('Expected canonical Internet Archive book URL')
    return 'doc_' + hashlib.sha1(url.encode()).hexdigest()


def make_target(elements):
    allowed = {'text', 'page_header', 'page_footer', 'section_header_level_1'}
    if not isinstance(elements, list) or not 1 <= len(elements) <= 100:
        raise ValueError('Expected 1..100 reviewed text elements')
    pieces = ['<doctag>']
    for tag, text in elements:
        if (tag not in allowed or not isinstance(text, str) or not text.strip() or
                len(text.encode('utf-8')) > 2_000_000 or '<' in text or '>' in text or
                re.search(r'&(?:#\w+|\w+);', text) or any(ord(c) < 32 for c in text)):
            raise ValueError('Unreviewed/ambiguous target structure')
        pieces.append(f'<{tag}>{text}</{tag}>')
    pieces.append('</doctag>')
    target = ''.join(pieces)
    if len(target.encode('utf-8')) > 2_000_000:
        raise ValueError('Oversized target')
    return target


def validate_review(rows, review, lock):
    cases = review['cases']
    if lock['rows'] != [0, 1, 2, 3, 4] or [c['row'] for c in cases] != lock['rows']:
        raise ValueError('Every fixed review row must be retained exactly once')
    selected = []
    for case in cases:
        row = rows[case['row']]
        source = lock['sources'][case['source']]
        if row['url'] != source['url'] or row['page_number'] != case['page']:
            raise ValueError('Changed source/page identity')
        identity = source_id(row['url']) + f'_p{row["page_number"]:05d}'
        response = json.loads(row['response'])
        if case['decision'] == 'approved':
            if not isinstance(response.get('natural_text'), str) or not response['natural_text'].strip():
                raise ValueError('Expected nonblank weak label for this reviewed page')
            target = make_target(case['elements'])
        elif case['decision'] == 'excluded-blank':
            if response.get('natural_text') is not None or case['elements']:
                raise ValueError('Changed blank-page evidence')
            target = None
        else:
            raise ValueError('Unknown review decision')
        selected.append(dict(row=case['row'], id=identity, upstream_id=row['id'], source=case['source'],
                             source_id=source_id(row['url']), url=row['url'], page=case['page'],
                             reviewed_image_sha256=case['image_sha256'],
                             decision=case['decision'], target=target,
                             weak_response_sha256=hashlib.sha256(row['response'].encode()).hexdigest()))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parquet', 'pdfs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq
    if pyarrow.__version__ != '21.0.0':
        raise ValueError('Use pyarrow 21.0.0')
    lock_path = ROOT / 'references/book-pilot.lock.json'
    review_path = ROOT / 'references/book-pilot-review.json'
    lock, review = (json.loads(p.read_text()) for p in (lock_path, review_path))
    verify_file(args.parquet, lock['sha256'], lock['bytes'])
    for name, source in lock['sources'].items():
        if not re.fullmatch(r'[a-zA-Z0-9_\-]+', name):
            raise ValueError('Unsafe source basename')
        verify_file(args.pdfs / (name + '.pdf'), source['pdf_sha256'], source['pdf_bytes'])
    version = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True, timeout=10, check=True)
    if (version.stdout + version.stderr).splitlines()[0] != 'pdftoppm version ' + lock['render']['poppler']:
        raise ValueError('Different Poppler version; qualify render changes explicitly')
    rows = pq.read_table(args.parquet).to_pylist()
    selected = validate_review(rows, review, lock)
    args.output.mkdir(parents=True, exist_ok=False)
    candidates, paths, inventory = [], [], []
    ledger = dict(schema_version=1, dataset=lock['dataset'], revision=lock['revision'],
                  training_source_allowlist=[], quarantined_source_ids=[], source_reviews={}, approved_targets={})
    for source in lock['sources'].values():
        identity = source_id(source['url'])
        if source.get('training_approved') is not True or source.get('privacy_cleared') is not True:
            raise ValueError('Source decision not approved for this local pilot')
        ledger['training_source_allowlist'].append(identity)
        ledger['source_reviews'][identity] = dict(original_source_url=source['url'], license_url=lock['license_url'],
                                                reviewer=review['reviewer'], decision_note=source['rights_review'],
                                                training_approved=True, privacy_cleared=True)
    for case in selected:
        name = f'page-{case["row"]:03d}'
        page = str(case['page'])
        command = ['pdftoppm', '-f', page, '-l', page, '-singlefile', '-r', str(lock['render']['dpi']), '-png',
                   str((args.pdfs / (case['source'] + '.pdf')).resolve()), str((args.output / name).resolve())]
        subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        image_sha = reviewed_image(args.output / (name + '.png'), case['reviewed_image_sha256'])
        target = case['target']
        entry = {k: v for k, v in case.items() if k != 'target'}
        entry.update(image=name + '.png', image_sha256=image_sha)
        if target is not None:
            path = args.output / (name + '.doctags')
            path.write_text(target, encoding='utf-8')
            paths.append(path)
            target_sha = digest(path)
            ledger['approved_targets'][case['id']] = target_sha
            candidates.append(dict(dataset=lock['dataset'], revision=lock['revision'], id=case['id'], split='train',
                                   image_sha256=image_sha, target_sha256=target_sha,
                                   content_sha256=content_fingerprint(target)))
            entry.update(target=path.name, target_sha256=target_sha, target_characters=len(target))
        inventory.append(entry)
    for candidate, result in zip(candidates, native_probe(paths), strict=True):
        candidate.update(result)
    # Every inspected source is development-only; future pages of these books cannot be final tests.
    preflight = evaluate_candidates(candidates, ledger, set(ledger['training_source_allowlist']))
    if not preflight['summary']['split_preflight_passed']:
        raise ValueError('Reviewed targets failed native or duplicate checks')
    for name, source in lock['sources'].items():
        verify_file(args.pdfs / (name + '.pdf'), source['pdf_sha256'], source['pdf_bytes'])
    verify_file(args.parquet, lock['sha256'], lock['bytes'])
    report = dict(schema_version=1, dataset=lock['dataset'], revision=lock['revision'], training_ready=False,
                  scope='Three assistant-reviewed training candidates from two source books, not complete train/validation/final splits or resource qualification. No model run or quality gain.',
                  lock_sha256=digest(lock_path), review_sha256=digest(review_path),
                  implementation_sha256={p: digest(ROOT / p) for p in ('scripts/book_pilot.py',
                      'scripts/corpus_preflight.py', 'scripts/dataset_audit.py', 'scripts/audit-doctags.lisp')},
                  tools=dict(pyarrow=pyarrow.__version__, poppler=lock['render']['poppler']),
                  inventory=inventory, preflight=preflight,
                  summary=dict(selected_pages=len(selected), sources=len(lock['sources']),
                               reviewed_training_candidates=len(candidates), excluded_blank=len(selected) - len(candidates)))
    (args.output / 'review-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
