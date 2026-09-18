"""Bounded source/label/split preflight, not a training loader or legal review.

The CLI checks the existing 20 development pages as proposed training candidates.
It never assigns final pages, trains, changes labels, or grants review approval.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from dataset_audit import ROOT, IDENTITY, native_probe

SOURCE = re.compile(r'doc_[0-9a-f]{40}')
SHA = re.compile(r'[0-9a-f]{64}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_fingerprint(raw):
    if not isinstance(raw, str) or not raw.strip() or len(raw.encode('utf-8')) > 2_000_000:
        raise ValueError('Missing or oversized target')
    # Conservative duplicate flag only: never used as a replacement training label.
    text = ' '.join(re.sub(r'<loc_[0-9]+>', '', raw).split())
    return digest(text.encode('utf-8'))


def reviewed_source(review):
    if not isinstance(review, dict):
        return False
    for name in ('original_source_url', 'license_url', 'reviewer', 'decision_note'):
        value = review.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            return False
        if name.endswith('_url'):
            url = urlparse(value)
            if url.scheme not in ('http', 'https') or not url.hostname:
                return False
    return review.get('training_approved') is True and review.get('privacy_cleared') is True


def evaluate_candidates(cases, ledger, development_sources):
    """Reject all conflicts; preserve every candidate and never silently repartition."""
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError('Expected 1..100 explicit candidates')
    if (ledger.get('schema_version') != 1 or not isinstance(ledger.get('dataset'), str) or
            not ledger['dataset'].strip() or not isinstance(ledger.get('revision'), str) or
            not re.fullmatch(r'[0-9a-f]{40}', ledger['revision'])):
        raise ValueError('Unsupported review ledger')
    source_sets = []
    for value in (ledger.get('training_source_allowlist'), ledger.get('quarantined_source_ids'),
                  development_sources):
        if not isinstance(value, (list, set)) or any(not isinstance(s, str) or not SOURCE.fullmatch(s) for s in value):
            raise ValueError('Invalid source list')
        source_sets.append(set(value))
    allowed, quarantined, development = source_sets
    reviews, targets = ledger.get('source_reviews', {}), ledger.get('approved_targets', {})
    if not isinstance(reviews, dict) or not isinstance(targets, dict):
        raise ValueError('Invalid review records')
    records = []
    for case in cases:
        if (case.get('dataset'), case.get('revision')) != (ledger.get('dataset'), ledger.get('revision')):
            raise ValueError('Dataset/revision differs from review scope')
        identity = IDENTITY.fullmatch(case.get('id', ''))
        if identity is None or case.get('split') not in ('train', 'validation', 'final'):
            raise ValueError('Invalid identity or explicit split')
        for key in ('image_sha256', 'target_sha256', 'content_sha256'):
            if not isinstance(case.get(key), str) or not SHA.fullmatch(case[key]):
                raise ValueError('Invalid content identity')
        codes = case.get('diagnostic_codes')
        if (type(case.get('markdown_renderable')) is not bool or not isinstance(codes, list) or
                any(not isinstance(c, str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,80}', c) for c in codes) or
                case['markdown_renderable'] == bool(codes)):
            raise ValueError('Missing or inconsistent native validation')
        # Output allowlist: no raw labels, paths or unreviewed free-form metadata.
        records.append({**{k: case[k] for k in ('dataset', 'revision', 'id', 'split',
                         'image_sha256', 'target_sha256', 'content_sha256', 'markdown_renderable')},
                        'source_id': identity[1], 'diagnostic_codes': list(codes), 'reasons': []})

    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    indexes = {}
    duplicates = set()
    for i, case in enumerate(records):
        for key in ('source_id', 'id', 'image_sha256', 'target_sha256', 'content_sha256'):
            identity = key, case[key]
            if identity in indexes:
                other = indexes[identity]
                parent[find(i)] = find(other)
                if key != 'source_id':
                    duplicates.update((i, other))
            else:
                indexes[identity] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    components = []
    for members in groups.values():
        splits = sorted({records[i]['split'] for i in members})
        sources = sorted({records[i]['source_id'] for i in members})
        seen_development = bool(set(sources) & development)
        seen_quarantine = bool(set(sources) & quarantined)
        components.append(dict(members=[records[i]['id'] for i in members], sources=sources,
                               splits=splits, development_seen=seen_development,
                               quarantine_seen=seen_quarantine))
        for i in members:
            case = records[i]
            source = case['source_id']
            reasons = case['reasons']
            if source in quarantined:
                reasons.append('quarantined-source')
            elif seen_quarantine:
                reasons.append('quarantined-component')
            if source not in allowed:
                reasons.append('source-not-approved')
            if not reviewed_source(reviews.get(source)):
                reasons.append('source-review-incomplete')
            if targets.get(case['id']) != case['target_sha256']:
                reasons.append('target-not-reviewed')
            if not case['markdown_renderable']:
                reasons.append('native-target-rejected')
            if len(splits) > 1:
                reasons.append('cross-split-component')
            if seen_development and case['split'] == 'final':
                reasons.append('development-source-in-final')
            if i in duplicates:
                reasons.append('duplicate-page')
            case['eligible'] = not reasons
    eligible = sum(c['eligible'] for c in records)
    return dict(schema_version=1, training_approved=False,
                scope='Source/label/split preflight only; full-context and resource gates still required. Review truth and unseen duplicates are not established by this tool.',
                fingerprint_policy='Exact encoded-image/target bytes; target content removes location tokens and collapses whitespace only. No perceptual matching or label modification.',
                summary=dict(pages=len(records), sources=len({c['source_id'] for c in records}),
                             eligible=eligible, blocked=len(records) - eligible,
                             split_preflight_passed=eligible == len(records),
                             native_renderable=sum(c['markdown_renderable'] for c in records),
                             blocked_reasons=dict(sorted(Counter(r for c in records for r in c['reasons']).items()))),
                components=components, cases=records)


def checked_bytes(root, name, wanted, maximum):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > maximum:
        raise ValueError('Evidence path/size outside scope')
    data = path.read_bytes()
    if digest(data) != wanted:
        raise ValueError('Evidence hash differs: ' + name)
    return data


def load_candidates(directory, geometry, audit):
    """Bind bytes to the trusted reference inventory, not a local parser report."""
    files, records, consumed = [], [], {}
    for i, (page, old) in enumerate(zip(geometry['cases'], audit['cases'], strict=True)):
        if (page['ordinal'], page['id'], page['original_target_sha256']) != (i, old['id'], old['target_sha256']):
            raise ValueError('Changed development-page identity')
        names = {f'{i:03d}.png': (page['image_sha256'], 20_000_000),
                 f'{i:03d}.original.doctags': (page['original_target_sha256'], 2_000_000),
                 f'{i:03d}.normalized.doctags': (page['normalized_target_sha256'], 2_000_000)}
        for name, (wanted, maximum) in names.items():
            checked_bytes(directory, name, wanted, maximum)
            consumed[name] = wanted
        path = directory / f'{i:03d}.normalized.doctags'
        files.append(path)
        records.append(dict(dataset=geometry['dataset'], revision=geometry['revision'], id=page['id'],
                            split='train', image_sha256=page['image_sha256'],
                            target_sha256=page['normalized_target_sha256'],
                            content_sha256=content_fingerprint(path.read_text(encoding='utf-8'))))
    return records, files, consumed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh metadata-only output directory')
    args = parser.parse_args()
    geometry_path = ROOT / 'tests/fixtures/dataset-geometry/report.json'
    audit_path = ROOT / 'tests/fixtures/dataset-audit/report.json'
    ledger_path = ROOT / 'references/dataset-source-review.json'
    geometry_bytes, audit_bytes, ledger_bytes = (p.read_bytes() for p in (geometry_path, audit_path, ledger_path))
    geometry, audit, ledger = (json.loads(b) for b in (geometry_bytes, audit_bytes, ledger_bytes))
    if geometry['prior_audit_sha256'] != digest(audit_bytes):
        raise ValueError('Development audit identity changed')
    if len(geometry['cases']) != 20 or len(audit['cases']) != 20:
        raise ValueError('Expected unchanged twenty-page development sample')
    records, files, consumed = load_candidates(args.audit_geometry, geometry, audit)
    for case, native in zip(records, native_probe(files), strict=True):
        case.update(native)
    report = evaluate_candidates(records, ledger, {c['source_id'] for c in audit['cases']})
    for name, wanted in consumed.items():
        checked_bytes(args.audit_geometry, name, wanted, 20_000_000)
    report.update(candidate_policy='All 20 previously audited development pages proposed as train candidates solely to exercise rejection. No corpus selected; no final examples assigned.',
                  evidence_sha256={'geometry_report': digest(geometry_bytes), 'audit_report': digest(audit_bytes),
                                   'review_ledger': digest(ledger_bytes)},
                  implementation_sha256={p: digest((ROOT / p).read_bytes()) for p in
                      ('scripts/corpus_preflight.py', 'scripts/dataset_audit.py', 'scripts/audit-doctags.lisp',
                       'src/doctags.lisp', 'src/document.lisp')})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    if not report['summary']['split_preflight_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
