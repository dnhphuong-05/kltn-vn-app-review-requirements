"""GĐ4a → gold: final labels for the calibration reviews (Q9: they go to Train, never to Hold-out).

Final label per review:
  - review listed in disagreements_to_finalize.csv → agreed_label / agreed_ambiguous (adjudicated,
    includes A=B labels that contradicted guideline v1.2);
  - otherwise A = B and consistent with v1.2 → consensus label; ambiguous_flag = 1 only when BOTH
    annotators marked it (one-sided marks are counted and reported).
Two outputs:
  - train-facing gold (no rating / dates: Q17 + ĐC "text-only input");
  - provenance (Phụ lục B columns + who/how/when), where rating and review_date stay as metadata.
"""
import csv
import hashlib
import json
import os

from .agreement import merge
from .calibration import LABELS

PROVENANCE_COLUMNS = [   # Phụ lục B + traceability of the adjudication
    'review_id', 'app', 'content', 'raw_text_hash', 'rating', 'review_date', 'app_version',
    'sampling_source', 'annotation_set', 'split', 'label_a', 'label_b', 'agreement_flag',
    'gold_label', 'ambiguous_flag', 'guideline_version', 'annotated_with_guideline',
    'labeled_by', 'resolution', 'label_date',
]
TRAIN_GOLD_COLUMNS = ['review_id', 'app', 'text', 'gold_label', 'ambiguous_flag', 'sampling_source',
                      'split', 'guideline_version']

SET_INFO = {   # calibration set → (sampling_source, guideline version used when annotating)
    'batch1': ('random', 'v1'),
    'batch2': ('random', 'v1.1'),
    'feature_ux': ('keyword_enriched', 'v1.1'),
}


def load_finalized(path: str) -> dict:
    """review_id → (label, ambiguous). Raises on blank / invalid entries."""
    out, bad = {}, []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            lab = (r.get('agreed_label') or '').strip().upper()
            amb = (r.get('agreed_ambiguous') or '').strip()
            if lab not in LABELS or amb not in ('0', '1'):
                bad.append((r['review_id'], lab, amb))
            out[r['review_id']] = (lab, int(amb) if amb in ('0', '1') else None)
    if bad:
        raise ValueError(f'disagreements_to_finalize.csv có dòng thiếu/sai: {bad}')
    return out


def build_calibration_gold(sets: dict, finalized: dict, interim_by_id: dict, guideline_version: str,
                           label_date: str) -> tuple:
    """Returns (provenance_rows, stats). sets: name → (path_a, path_b)."""
    rows, stats = [], {'from_finalize': 0, 'consensus': 0, 'ambiguous_one_sided': []}
    for name, (pa, pb) in sets.items():
        sampling_source, annotated_with = SET_INFO[name]
        for r in merge(pa, pb)['df'].to_dict('records'):
            rid = r['review_id']
            if rid in finalized:
                gold, amb = finalized[rid]
                resolution = 'adjudicated'
                stats['from_finalize'] += 1
            elif r['label_a'] == r['label_b']:
                gold, resolution = r['label_a'], 'consensus'
                amb = int(r['ambiguous_a'] and r['ambiguous_b'])
                if r['ambiguous_a'] != r['ambiguous_b']:
                    stats['ambiguous_one_sided'].append(rid)
                stats['consensus'] += 1
            else:
                raise ValueError(f'{rid}: A≠B nhưng không có trong file chốt nhãn')
            src = interim_by_id[rid]
            rows.append({
                'review_id': rid, 'app': src['source_app'], 'content': src['text'],
                'raw_text_hash': hashlib.sha256(src['text'].encode('utf-8')).hexdigest(),
                'rating': src['rating'], 'review_date': src['review_date'], 'app_version': src['app_version'] or '',
                'sampling_source': sampling_source, 'annotation_set': name, 'split': 'train',
                'label_a': r['label_a'], 'label_b': r['label_b'], 'agreement_flag': int(r['label_a'] == r['label_b']),
                'gold_label': gold, 'ambiguous_flag': amb, 'guideline_version': guideline_version,
                'annotated_with_guideline': annotated_with, 'labeled_by': 'A+B', 'resolution': resolution,
                'label_date': label_date,
            })
    ids = [r['review_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('review_id trùng giữa các bộ calibration')
    unused = set(finalized) - set(ids)
    if unused:
        raise ValueError(f'File chốt nhãn có review không thuộc calibration: {sorted(unused)}')
    return rows, stats


def build_double_label_gold(name: str, path_a: str, path_b: str, finalize_path: str, interim_by_id: dict,
                            sampling_source: str, split: str, annotated_with: str, guideline_version: str,
                            label_date: str) -> tuple:
    """Gold rows for any double-labelled set (pilot, Hold-out, Validation, …): consensus where A = B,
    agreed_label from `finalize_path` where they differ. Every disagreement must be finalised."""
    finalized = load_finalized(finalize_path) if os.path.exists(finalize_path) else {}
    rows, stats = [], {'from_finalize': 0, 'consensus': 0, 'ambiguous_one_sided': []}
    for r in merge(path_a, path_b)['df'].to_dict('records'):
        rid = r['review_id']
        if r['label_a'] != r['label_b']:
            if rid not in finalized:
                raise ValueError(f'{rid}: A≠B nhưng chưa có agreed_label')
            gold, amb = finalized[rid]
            resolution = 'adjudicated'
            stats['from_finalize'] += 1
        else:
            gold, resolution = r['label_a'], 'consensus'
            amb = int(r['ambiguous_a'] and r['ambiguous_b'])
            if r['ambiguous_a'] != r['ambiguous_b']:
                stats['ambiguous_one_sided'].append(rid)
            stats['consensus'] += 1
        src = interim_by_id[rid]
        rows.append({
            'review_id': rid, 'app': src['source_app'], 'content': src['text'],
            'raw_text_hash': hashlib.sha256(src['text'].encode('utf-8')).hexdigest(),
            'rating': src['rating'], 'review_date': src['review_date'], 'app_version': src['app_version'] or '',
            'sampling_source': sampling_source, 'annotation_set': name, 'split': split,
            'label_a': r['label_a'], 'label_b': r['label_b'], 'agreement_flag': int(r['label_a'] == r['label_b']),
            'gold_label': gold, 'ambiguous_flag': amb, 'guideline_version': guideline_version,
            'annotated_with_guideline': annotated_with, 'labeled_by': 'A+B', 'resolution': resolution,
            'label_date': label_date,
        })
    return rows, stats


def write_provenance(rows, path: str) -> str:
    """Creates provenance.csv with the full column set. A file holding only a header (the old stub)
    is replaced; a file that already holds data is appended to, refusing duplicate review_ids."""
    existing = []
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig', newline='') as f:
            existing = list(csv.DictReader(f))
    for r in existing:   # column renamed calibration_set → annotation_set when pilot/Hold-out were added
        if 'annotation_set' not in r and 'calibration_set' in r:
            r['annotation_set'] = r.pop('calibration_set')
    if existing:
        have = {r['review_id'] for r in existing}
        dup = [r['review_id'] for r in rows if r['review_id'] in have]
        if dup:
            raise ValueError(f'provenance.csv đã có {len(dup)} review_id này, vd {dup[:3]}')
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PROVENANCE_COLUMNS)
        w.writeheader()
        for r in existing + rows:
            w.writerow({k: r.get(k, '') for k in PROVENANCE_COLUMNS})
    return path


def write_train_gold(rows, path: str) -> str:
    if os.path.exists(path):
        raise FileExistsError(f'{path} đã tồn tại — không ghi đè.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_GOLD_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({'review_id': r['review_id'], 'app': r['app'], 'text': r['content'],
                        'gold_label': r['gold_label'], 'ambiguous_flag': r['ambiguous_flag'],
                        'sampling_source': r['sampling_source'], 'split': r['split'],
                        'guideline_version': r['guideline_version']})
    return path


def write_holdout_exclusion(rows, path: str) -> dict:
    """review_ids that GĐ3/GĐ4b must remove from the Hold-out pool (annotators have seen them)."""
    manifest = {
        'stage': 'GĐ4a → exclusion list for the Hold-out pool',
        'reason': 'calibration reviews (incl. keyword-enriched FEATURE–UX set) were seen by annotators; '
                  'Q9: calibration never enters Hold-out, goes to Train after adjudication',
        'n': len(rows),
        'by_set': {s: sum(r['annotation_set'] == s for r in rows) for s in SET_INFO},
        'review_ids': [r['review_id'] for r in rows],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
