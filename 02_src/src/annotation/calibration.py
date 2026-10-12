"""GĐ4a — calibration sample for two independent annotators (ĐC §7.2.1, Q10, Q17).

150 Shopee interim reviews are drawn once (seed 42) and split into two batches of 75:
batch 1 → discuss disagreements, revise the guideline → batch 2 → agreement gate (κ ≥ 0.6).
The sheet shows only review_id + text (the de-identified original, not text_clean, so emoji
and teencode stay as written). rating / date / app_version are never exported (Q17).
Calibration review_ids are recorded in a manifest so they can be excluded from the
Hold-out pool later (Q9: calibration never enters Hold-out; after adjudication → Train).
"""
import csv
import hashlib
import json
import os
import random

LABELS = ('BUG', 'FEATURE', 'UX', 'OTHER')
SHEET_COLUMNS = ['review_id', 'text', 'label', 'ambiguous', 'note']


def draw_calibration(records, n: int = 150, n_batches: int = 2, seed: int = 42) -> dict:
    """Random sample of `n` records, split into equal consecutive batches."""
    rng = random.Random(seed)
    sample = rng.sample(records, n)
    size = n // n_batches
    return {f'batch{i + 1}': sample[i * size:(i + 1) * size] for i in range(n_batches)}


def export_blank(rows, path: str) -> str:
    """CSV for Google Sheets: review_id, text, and empty label / ambiguous / note columns.
    Never overwrites an existing sheet (annotators may already be working on it)."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} đã tồn tại — không ghi đè file gán nhãn.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:   # BOM: Excel-safe, Sheets ignores it
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(SHEET_COLUMNS)
        for r in rows:
            w.writerow([r['review_id'], r['text'], '', '', ''])
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(batches: dict, source_path: str, sheet_paths: dict, path: str, seed: int = 42) -> dict:
    manifest = {
        'stage': 'GĐ4a calibration',
        'source': source_path, 'source_sha256': _sha256(source_path), 'seed': seed,
        'design': 'random 150 from Shopee interim, split 2 x 75; batch1 → revise guideline; '
                  'batch2 → agreement gate (Cohen kappa >= 0.6, Q10)',
        'columns_shown': SHEET_COLUMNS, 'hidden': ['rating', 'review_date', 'app_version'],
        'exclude_from_holdout': True, 'after_adjudication': 'Train (Q9)',
        'batches': {b: {'sheet': sheet_paths[b], 'review_ids': [r['review_id'] for r in rows]}
                    for b, rows in batches.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


# ---------------------------------------------------------------- amendment v1.6
# Random calibration batches contain very few FEATURE/UX reviews, so the FEATURE–UX boundary
# cannot be checked by the gate. A small keyword-selected set is annotated by both people for
# discussion only; it is NOT part of the κ gate (gate stays on random batch 2).
# Keyword groups (variants of one cue share a group). Sampling is balanced across groups:
# a plain random draw over all matches is swamped by "phiền" (~80% of matches, mostly ad
# complaints already settled by R2) and would not exercise the FEATURE–UX boundary.
FEATURE_UX_KEYWORD_GROUPS = {
    'nên thêm': ('nên thêm',), 'mong có': ('mong có',), 'bổ sung': ('bổ sung',), 'khó dùng': ('khó dùng',),
    'khó tìm': ('khó tìm',), 'không tìm thấy': ('không tìm thấy',), 'rối': ('rối',), 'phiền': ('phiền',),
    'cho tắt': ('cho tắt',), 'tùy chọn': ('tùy chọn', 'tuỳ chọn'),
}
FEATURE_UX_KEYWORDS = tuple(k for g in FEATURE_UX_KEYWORD_GROUPS.values() for k in g)


def draw_keyword_set(records, groups=FEATURE_UX_KEYWORD_GROUPS, n: int = 30, exclude_ids=(), seed: int = 42):
    """Balanced draw: n // len(groups) random reviews per keyword group (text_clean, case-insensitive),
    excluding already-used ids and reviews picked for an earlier group; any shortfall is filled at
    random from all remaining matches. Returns (sample, n_candidates, per_group_counts)."""
    rng = random.Random(seed)
    exclude = set(exclude_ids)
    match = lambda r, kws: any(k in r['text_clean'].lower() for k in kws)
    pool = [r for r in records if r['review_id'] not in exclude]
    all_kw = [k for g in groups.values() for k in g]
    cands = [r for r in pool if match(r, all_kw)]
    per_group, chosen, picked = n // len(groups), [], set()
    counts = {}
    for name, kws in groups.items():
        g = [r for r in cands if r['review_id'] not in picked and match(r, kws)]
        take = rng.sample(g, min(per_group, len(g)))
        chosen += take
        picked |= {r['review_id'] for r in take}
        counts[name] = len(take)
    rest = [r for r in cands if r['review_id'] not in picked]
    fill = rng.sample(rest, min(n - len(chosen), len(rest)))
    chosen += fill
    counts['_fill_random'] = len(fill)
    return chosen, len(cands), counts


def write_keyword_manifest(rows, keywords, n_candidates, source_path, sheet_path, path, seed: int = 42) -> dict:
    manifest = {
        'stage': 'GĐ4a calibration — bộ bổ sung FEATURE–UX (amendment v1.6)',
        'source': source_path, 'source_sha256': _sha256(source_path), 'seed': seed,
        'selection': 'keyword match on text_clean (case-insensitive), excluding calibration batch1/batch2 ids',
        'keywords': list(keywords), 'n_candidates': n_candidates,
        'use': 'discussion of the FEATURE–UX boundary only; NOT part of the kappa gate (gate = random batch2)',
        'exclude_from_holdout': True, 'later_use': 'Train as keyword_enriched — to be decided',
        'sheet': sheet_path, 'review_ids': [r['review_id'] for r in rows],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


INSTRUCTIONS = """Hướng dẫn cho người gán nhãn
- Nhãn hợp lệ ở cột `label`: BUG / FEATURE / UX / OTHER (theo `00_protocol/annotation_guideline_v1.md` và `decision_tree.md`).
- Cột `ambiguous`: điền 1 nếu review có hai ý actionable độc lập ngang nhau, không xác định được ý chính; để trống nếu không.
  Vẫn chọn MỘT nhãn tốt nhất ở cột `label`.
- Cột `note`: ghi ngắn lý do khi phân vân.
- Mỗi người làm trên BẢN SAO Google Sheets riêng (A và B), gán độc lập, không xem bản của người kia, không thảo luận khi đang gán.
- Cột `label` dùng Data validation dạng dropdown với 4 giá trị trên; cột `ambiguous` dropdown 1 / trống.
- Gán xong batch1 → tải về calibration_batch1_A.csv / calibration_batch1_B.csv → họp thảo luận bất đồng, sửa guideline
  → mới gán batch2 bằng guideline đã sửa → tải về calibration_batch2_A.csv / calibration_batch2_B.csv."""
