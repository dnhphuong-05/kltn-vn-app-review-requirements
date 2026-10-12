"""Pilot prevalence sample (ĐC §7.2.3; Q8, Q9; amendment v1.5).

400 Shopee interim reviews drawn by pure random sampling (seed 42) after removing every
review_id in the Hold-out exclusion list (calibration). No enrichment, no near-duplicate or
rating filter: the pilot must mirror the corpus distribution for its prevalence to mean anything.
The pilot is part of the Hold-out (Q9), so its files fall under HOLDOUT_LOCK and its labels are
adjudicated with the frozen guideline only (never used to revise the guideline).

Prevalence is reported with Wilson 95% intervals. Before adjudication the consensus-only shares
are shown together with lower/upper bounds that assign every disagreement against / for a class.
"""
import hashlib
import json
import math
import os
import random

from src.annotation.calibration import LABELS

ACTIONABLE = ('BUG', 'FEATURE', 'UX')


def draw_pilot(records, n: int = 400, exclude_ids=(), seed: int = 42):
    """Pure random sample of `n` records not in `exclude_ids`. Returns (sample, pool_size)."""
    exclude = set(exclude_ids)
    pool = [r for r in records if r['review_id'] not in exclude]
    return random.Random(seed).sample(pool, n), len(pool)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_pilot_manifest(rows, pool_size, source_path, exclusion_path, sheet_path, path, seed: int = 42) -> dict:
    manifest = {
        'stage': 'GĐ3/GĐ4 pilot prevalence',
        'source': source_path, 'source_sha256': _sha256(source_path), 'seed': seed,
        'design': 'pure random sample from Shopee interim after removing the Hold-out exclusion list; '
                  'no enrichment, no near-duplicate / rating / keyword filter',
        'excluded_from_pool': exclusion_path, 'pool_size': pool_size,
        'in_holdout': True, 'holdout_lock': 'files fall under 01_data/HOLDOUT_LOCK.md (Q9: pilot ⊂ Hold-out)',
        'sheet': sheet_path, 'n': len(rows), 'review_ids': [r['review_id'] for r in rows],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    """Wilson score interval for a binomial proportion (well-behaved for small k)."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def prevalence_table(labels, n_total=None) -> list:
    """labels: final label per review. Rows: class, count, share, Wilson 95% CI (+ ACTIONABLE)."""
    n = n_total if n_total is not None else len(labels)
    rows = []
    for cls in LABELS + ('ACTIONABLE',):
        k = sum(1 for l in labels if (l in ACTIONABLE if cls == 'ACTIONABLE' else l == cls))
        lo, hi = wilson(k, n)
        rows.append({'class': cls, 'count': k, 'share': k / n if n else float('nan'), 'ci_low': lo, 'ci_high': hi})
    return rows


def prevalence_bounds(df) -> list:
    """Before adjudication: per class, share among consensus reviews only, plus the range obtained by
    assigning every disagreement against (lower) or towards (upper) that class, over all reviews."""
    n = len(df)
    cons = df[df['label_a'] == df['label_b']]
    dis = df[df['label_a'] != df['label_b']]
    out = []
    for cls in LABELS + ('ACTIONABLE',):
        is_cls = (lambda l: l in ACTIONABLE) if cls == 'ACTIONABLE' else (lambda l, c=cls: l == c)
        k_cons = int(cons['label_a'].map(is_cls).sum())
        k_dis_touch = int((dis['label_a'].map(is_cls) | dis['label_b'].map(is_cls)).sum())
        k_dis_both = int((dis['label_a'].map(is_cls) & dis['label_b'].map(is_cls)).sum())   # e.g. BUG vs UX for ACTIONABLE
        out.append({'class': cls, 'consensus_count': k_cons,
                    'share_among_consensus': k_cons / len(cons) if len(cons) else float('nan'),
                    'lower_bound': (k_cons + k_dis_both) / n, 'upper_bound': (k_cons + k_dis_touch) / n,
                    'n_disagreements_touching': k_dis_touch})
    return out


def rarest(table) -> dict:
    """Rarest actionable class (OTHER and ACTIONABLE are not candidates)."""
    cand = [r for r in table if r['class'] in ACTIONABLE]
    return min(cand, key=lambda r: r['share'])


INSTRUCTIONS = """Pilot prevalence — 400 review ngẫu nhiên, gán bằng guideline v1.2 (FROZEN)
- Nhãn: BUG / FEATURE / UX / OTHER; cột ambiguous: 1 khi ≥2 ý actionable khác lớp không ý nào trội, ngược lại 0; note: lý do khi phân vân.
- Hai người gán độc lập trên hai bản sao Google Sheets riêng (pilot_A / pilot_B), không xem bản của nhau.
- Pilot là một phần của Hold-out: không thảo luận nội dung pilot để sửa guideline; ca lệch được chốt theo guideline v1.2.
- Gán xong tải về pilot_A.csv và pilot_B.csv, đặt trong 07_human_eval/pilot/."""
