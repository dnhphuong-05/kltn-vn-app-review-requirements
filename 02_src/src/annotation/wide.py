"""GĐ4b — checks and agreement for the large annotation round (amendment v1.7).

Double-labelled sets (Hold-out extra, Validation, focused FEATURE/UX, Lazada) are checked and
measured like calibration. Train is single-labelled; its audit subset (15%) is paired with the
main annotator's labels to measure Train label quality — audit disagreements are analysed only,
never used to relabel Train.
"""
import csv

import pandas as pd

from .agreement import agreement, disagreements_by_pair, merge
from .calibration import LABELS

PILOT_KAPPA_FU = 0.79      # κ FEATURE–UX on the pilot (reference for "drops sharply")
KAPPA_GATE = 0.6


def _ambiguous_problems(path, who):
    bad = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if (r.get('ambiguous') or '').strip() not in ('0', '1'):
                bad.append(r['review_id'])
    return [f'{who}: {len(bad)} ô ambiguous không phải 0/1, vd {bad[:3]}'] if bad else []


def check_pair(path_a, path_b, expected_ids):
    m = merge(path_a, path_b, expected_ids=expected_ids)
    m['problems'] += _ambiguous_problems(path_a, 'A') + _ambiguous_problems(path_b, 'B')
    return m


def check_single(path, expected_ids):
    """Single-labelled sheet: same id set as expected, valid labels, ambiguous 0/1."""
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig', newline='')))
    ids = [r['review_id'].strip() for r in rows]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append(f'{len(ids) - len(set(ids))} review_id trùng')
    if set(ids) != set(expected_ids):
        problems.append(f'Lệch so với manifest: thiếu={len(set(expected_ids) - set(ids))}, thừa={len(set(ids) - set(expected_ids))}')
    blank = [r['review_id'] for r in rows if not (r.get('label') or '').strip()]
    invalid = [(r['review_id'], r['label']) for r in rows if (r.get('label') or '').strip()
               and r['label'].strip().upper() not in LABELS]
    if blank:
        problems.append(f'{len(blank)} ô label trống, vd {blank[:3]}')
    if invalid:
        problems.append(f'{len(invalid)} nhãn sai, vd {invalid[:3]}')
    problems += _ambiguous_problems(path, 'train')
    return {'n': len(rows), 'problems': problems}


def audit_frame(main_path, audit_path):
    """Pairs the main annotator (A, from the Train sheet) with the audit annotator (B) on the audit ids."""
    audit_ids = [r['review_id'] for r in csv.DictReader(open(audit_path, encoding='utf-8-sig', newline=''))]
    main_ids = {r['review_id'] for r in csv.DictReader(open(main_path, encoding='utf-8-sig', newline=''))}
    m = merge(main_path, audit_path)          # merge keeps the intersection only
    in_main = len(set(audit_ids) & main_ids)
    if in_main != len(audit_ids):
        m['problems'].append(f'{len(audit_ids) - in_main} review audit không có trong Train')
    # merge() reports the id-set difference (Train has 1.020, audit 153) — expected here, so drop it
    m['problems'] = [p for p in m['problems'] if not p.startswith('Lệch tập review_id')]
    m['problems'] += _ambiguous_problems(audit_path, 'audit')
    m['n_audit_in_train'] = in_main
    m['n_audit'] = len(audit_ids)
    return m


def summary_row(name, res, groups):
    fu = res['feature_ux']
    return {'tập': name, 'n': res['n_valid'], '% đồng thuận': res['percent_agreement'],
            'κ tổng': res['kappa_4class'], 'κ act–OTHER': res['actionable_other']['kappa'],
            'κ FEAT–UX': fu['kappa'], 'n FEAT–UX': fu['n_both_feature_or_ux'],
            'số ca lệch': sum(len(v) for v in groups.values())}


def warnings_for(name, res, is_external=False):
    out = []
    k, kfu = res['kappa_4class'], res['feature_ux']['kappa']
    if k is not None and k < KAPPA_GATE:
        out.append(f'{name}: κ tổng = {k} < {KAPPA_GATE}')
    if kfu is not None and kfu < PILOT_KAPPA_FU - 0.15:
        out.append(f'{name}: κ FEATURE–UX = {kfu} (n={res["feature_ux"]["n_both_feature_or_ux"]}), '
                   f'thấp hơn pilot ({PILOT_KAPPA_FU}) hơn 0,15')
    if is_external and out:
        out = [w + ' — Lazada có phân bố khác Shopee, xem như thông tin, không coi là lỗi' for w in out]
    return out
