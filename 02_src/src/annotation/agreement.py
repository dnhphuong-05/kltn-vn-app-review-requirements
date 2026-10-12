"""GĐ4a — merge two independent calibration sheets and measure agreement (Q10).

Reported: percent agreement, Cohen's κ (4 classes), 4×4 confusion A×B, and two boundary
views where confusion is expected:
  - actionable–OTHER: both labels mapped to ACTIONABLE (BUG∪FEATURE∪UX) / OTHER → this is
    the Tier-1 agreement on every item;
  - FEATURE–UX: items where BOTH annotators chose FEATURE or UX → binary κ on that subset,
    plus the count of FEATURE↔UX cross-labels.
Gate: κ (4 classes) ≥ 0.6 on the gate batch — necessary, not sufficient; disagreements are
printed grouped by label pair for the guideline meeting. Nothing here edits the guideline.
"""
import csv
import os
from collections import defaultdict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .calibration import LABELS

KAPPA_GATE = 0.6
ACTIONABLE = {'BUG', 'FEATURE', 'UX'}


def load_sheet(path: str) -> dict:
    """review_id → {label, ambiguous, note, text}. Labels upper/stripped; raw value kept for errors."""
    out = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            rid = (row.get('review_id') or '').strip()
            if not rid:
                continue
            amb = (row.get('ambiguous') or '').strip().lower()
            out[rid] = {
                'label': (row.get('label') or '').strip().upper(),
                'ambiguous': amb in ('1', 'true', 'x', 'yes', 'có', 'co'),
                'note': (row.get('note') or '').strip(),
                'text': row.get('text') or '',
            }
    return out


def merge(path_a: str, path_b: str, expected_ids=None) -> dict:
    a, b = load_sheet(path_a), load_sheet(path_b)
    ids_a, ids_b = set(a), set(b)
    problems = []
    if ids_a != ids_b:
        problems.append(f'Lệch tập review_id: chỉ A={len(ids_a - ids_b)}, chỉ B={len(ids_b - ids_a)}')
    if expected_ids is not None and ids_a | ids_b != set(expected_ids):
        problems.append(f'Không khớp manifest calibration: thiếu={len(set(expected_ids) - (ids_a | ids_b))}, '
                        f'thừa={len((ids_a | ids_b) - set(expected_ids))}')
    for who, sheet in (('A', a), ('B', b)):
        empty = [r for r, v in sheet.items() if not v['label']]
        invalid = [(r, v['label']) for r, v in sheet.items() if v['label'] and v['label'] not in LABELS]
        if empty:
            problems.append(f'{who}: {len(empty)} ô label trống, vd {empty[:5]}')
        if invalid:
            problems.append(f'{who}: {len(invalid)} nhãn sai, vd {invalid[:5]}')
    rows = []
    for rid in sorted(ids_a & ids_b):
        rows.append({'review_id': rid, 'text': a[rid]['text'] or b[rid]['text'],
                     'label_a': a[rid]['label'], 'label_b': b[rid]['label'],
                     'ambiguous_a': a[rid]['ambiguous'], 'ambiguous_b': b[rid]['ambiguous'],
                     'note_a': a[rid]['note'], 'note_b': b[rid]['note']})
    return {'df': pd.DataFrame(rows), 'problems': problems}


def _kappa(x, y):
    if len(x) == 0:
        return None
    if len(set(x) | set(y)) < 2:      # one category only: κ undefined
        return None
    return round(float(cohen_kappa_score(x, y)), 3)


def agreement(df: pd.DataFrame) -> dict:
    valid = df[df['label_a'].isin(LABELS) & df['label_b'].isin(LABELS)]
    a, b = valid['label_a'], valid['label_b']
    confusion = pd.crosstab(a, b, rownames=['A'], colnames=['B']).reindex(index=LABELS, columns=LABELS, fill_value=0)

    t1a = a.map(lambda l: 'ACTIONABLE' if l in ACTIONABLE else 'OTHER')
    t1b = b.map(lambda l: 'ACTIONABLE' if l in ACTIONABLE else 'OTHER')
    fu = valid[a.isin({'FEATURE', 'UX'}) & b.isin({'FEATURE', 'UX'})]
    fu_cross = int(((a == 'FEATURE') & (b == 'UX')).sum() + ((a == 'UX') & (b == 'FEATURE')).sum())

    return {
        'n_valid': len(valid), 'n_excluded_invalid': len(df) - len(valid),
        'percent_agreement': round(float((a == b).mean() * 100), 1) if len(valid) else None,
        'kappa_4class': _kappa(a, b),
        'confusion': confusion,
        'actionable_other': {'percent_agreement': round(float((t1a == t1b).mean() * 100), 1) if len(valid) else None,
                             'kappa': _kappa(t1a, t1b), 'n_disagree': int((t1a != t1b).sum())},
        'feature_ux': {'n_both_feature_or_ux': len(fu),
                       'percent_agreement': round(float((fu['label_a'] == fu['label_b']).mean() * 100), 1) if len(fu) else None,
                       'kappa': _kappa(fu['label_a'], fu['label_b']), 'n_cross_labels': fu_cross},
        'ambiguous_marked': {'A': int(valid['ambiguous_a'].sum()), 'B': int(valid['ambiguous_b'].sum())},
    }


def disagreements_by_pair(df: pd.DataFrame) -> dict:
    """{(label_a, label_b): [rows]} for every disagreement, largest groups first."""
    groups = defaultdict(list)
    valid = df['label_a'].isin(LABELS) & df['label_b'].isin(LABELS)   # blanks/typos are reported by merge()
    for r in df[valid & (df['label_a'] != df['label_b'])].to_dict('records'):
        groups[(r['label_a'], r['label_b'])].append(r)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


FINALIZE_COLUMNS = ['review_id', 'source', 'text', 'A_label', 'B_label', 'R_rule', 'agreed_label', 'agreed_ambiguous']


def build_finalize_rows(sets: dict, rules: dict, agreed_conflicts: dict) -> list:
    """Rows for the label-finalisation sheet (agreed_label always left blank for the annotators).
    sets: name → (path_a, path_b); rules: review_id prefix → rule note for each disagreement;
    agreed_conflicts: review_id prefix → rule note for A=B labels that contradict the frozen guideline."""
    rows = []
    for name, (pa, pb) in sets.items():
        df = merge(pa, pb)['df']
        for r in df.to_dict('records'):
            pre = r['review_id'][:8]
            if r['label_a'] != r['label_b']:
                source, note = f'{name}_disagreement', rules.get(pre, '')
            elif pre in agreed_conflicts:
                source, note = 'agreed_conflicts_v1.2', agreed_conflicts[pre]
            else:
                continue
            rows.append({'review_id': r['review_id'], 'source': source, 'text': r['text'],
                         'A_label': r['label_a'], 'B_label': r['label_b'], 'R_rule': note, 'agreed_label': '',
                         'agreed_ambiguous': ''})
    missing = (set(rules) | set(agreed_conflicts)) - {r['review_id'][:8] for r in rows}
    if missing:
        raise ValueError(f'Không tìm thấy review cho các mã: {sorted(missing)}')
    return rows


def write_finalize(rows, path: str) -> str:
    """Never overwrites: the annotators fill agreed_label in this file."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} đã tồn tại — không ghi đè (có thể đã điền agreed_label).')
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FINALIZE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def gate_checklist(res: dict, groups: dict, batch: str, is_gate_batch: bool) -> str:
    k = res['kappa_4class']
    fu_dis = sum(len(v) for (x, y), v in groups.items() if {x, y} == {'FEATURE', 'UX'})
    act_dis = res['actionable_other']['n_disagree']
    L = [f'Checklist chốt guideline — {batch}',
         f"  κ tổng (4 lớp)          : {k}  (cổng ≥ {KAPPA_GATE})",
         f"  % đồng thuận            : {res['percent_agreement']}%  (n={res['n_valid']})",
         f"  κ actionable–OTHER      : {res['actionable_other']['kappa']}  | bất đồng: {act_dis}",
         f"  κ FEATURE–UX (tập con)  : {res['feature_ux']['kappa']}  | n tập con: {res['feature_ux']['n_both_feature_or_ux']}"
         f" | bất đồng FEATURE↔UX: {fu_dis}",
         f"  Đánh dấu ambiguous      : A={res['ambiguous_marked']['A']}, B={res['ambiguous_marked']['B']}",
         '  Số ca bất đồng theo cặp (A/B): ' + (', '.join(f'{x}/{y}={len(v)}' for (x, y), v in groups.items()) or 'không có')]
    if not is_gate_batch:
        L.append('  → Batch thảo luận: dùng các ca bất đồng ở trên để họp sửa guideline trước khi gán batch tiếp theo.')
    elif k is not None and k >= KAPPA_GATE:
        L.append(f'  → κ đạt cổng. Người nghiên cứu xem thêm {fu_dis} ca bất đồng FEATURE/UX trước khi kết luận; '
                 'nếu không còn vấn đề hệ thống: guideline sẵn sàng chốt (điền ví dụ thật ở bước riêng, sau xác nhận).')
    else:
        L.append('  → κ chưa đạt cổng: KHÔNG chốt guideline. Họp sửa guideline theo các ca bất đồng ở trên, '
                 'rồi gán một batch calibration mới (calibration lần 2).')
    return '\n'.join(L)
