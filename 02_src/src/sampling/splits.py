"""GĐ3/GĐ4 splits (amendment v1.7).

Shopee pool = interim corpus − calibration (180, all in Train) − pilot (400, all in Hold-out).
Draw order (seed 42, one generator, no review in two sets):
  1. Hold-out extra 600   — pure random (+ 400 pilot = 1.000, cap Q8)          double-label, LOCKED
  2. Validation 350       — random at draw time (labels unknown; Q8 stratification is done after
                            labelling by topping up rare classes if needed)    double-label
  3. Focused FEATURE/UX 150 — keyword-enriched, balanced across keyword groups double-label, LOCKED,
                                                                               reported separately
  4. Train extra 1.020    — ~17.5% FEATURE-cue + ~17.5% UX-cue keyword matches, rest random
                            (+ 180 calibration = 1.200); 15% audit subset for a second annotator
Lazada external test 500 — pure random from Lazada interim                     double-label, LOCKED
Keyword matches are only a sampling device — they are never labels.
"""
import csv
import hashlib
import json
import os
import random
from itertools import combinations

from src.annotation.calibration import draw_keyword_set

FEATURE_CUES = {
    'nên thêm': ('nên thêm',), 'mong có': ('mong có',), 'bổ sung': ('bổ sung',),
    'tùy chọn': ('tùy chọn', 'tuỳ chọn'), 'nên có': ('nên có',), 'cho phép': ('cho phép',),
    'thêm chức năng': ('thêm chức năng', 'thêm tính năng'), 'cho tắt': ('cho tắt',),
}
UX_CUES = {
    'khó dùng': ('khó dùng', 'khó sử dụng'), 'khó tìm': ('khó tìm',), 'không tìm thấy': ('không tìm thấy',),
    'rối': ('rối',), 'bất tiện': ('bất tiện',), 'phiền': ('phiền',),
}


def _matches(r, groups):
    t = r['text_clean'].lower()
    return any(k in t for kws in groups.values() for k in kws)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_splits(shopee, lazada, calib_ids, pilot_ids, sizes: dict, seed: int = 42) -> dict:
    """Returns {set_name: {'rows': [...], 'meta': {...}}}. Raises if any draw is short or overlaps."""
    rng = random.Random(seed)
    calib, pilot = set(calib_ids), set(pilot_ids)
    by_id = {r['review_id']: r for r in shopee}
    missing = (calib | pilot) - set(by_id)
    if missing:
        raise ValueError(f'{len(missing)} review_id calibration/pilot không có trong Shopee interim')
    used = set(calib) | set(pilot)
    pool = lambda: [r for r in shopee if r['review_id'] not in used]

    def take(rows):
        used.update(r['review_id'] for r in rows)
        return rows

    out = {}
    # 1. Hold-out extra (pure random)
    hx = take(rng.sample(pool(), sizes['holdout_extra']))
    out['holdout'] = {'rows': [by_id[i] for i in pilot_ids] + hx, 'meta': {
        'size': len(pilot_ids) + len(hx), 'composition': {'pilot (labelled)': len(pilot_ids), 'extra (to label)': len(hx)},
        'sampling': 'pure random', 'label_mode': 'double', 'locked': True, 'to_label': [r['review_id'] for r in hx]}}
    # 2. Validation (random at draw time)
    va = take(rng.sample(pool(), sizes['validation']))
    out['validation'] = {'rows': va, 'meta': {
        'size': len(va), 'sampling': 'random at draw time; Q8 stratification applied after labelling '
        '(top up rare classes from the remaining pool if too few)', 'label_mode': 'double', 'locked': False,
        'to_label': [r['review_id'] for r in va]}}
    # 3. Focused FEATURE/UX set (keyword-enriched, balanced over cue groups)
    groups = {**FEATURE_CUES, **UX_CUES}
    fo, n_cand, per_group = draw_keyword_set(pool(), groups, n=sizes['focused'], seed=rng.randrange(10**9))
    take(fo)
    out['focused'] = {'rows': fo, 'meta': {
        'size': len(fo), 'sampling': 'keyword-enriched, balanced across cue groups', 'cue_groups': per_group,
        'n_candidates': n_cand, 'label_mode': 'double', 'locked': True, 'reported_separately': True,
        'to_label': [r['review_id'] for r in fo]}}
    # 4. Train extra: FEATURE-cue + UX-cue enrichment, rest random
    n_tr = sizes['train_extra']
    n_feat = round(n_tr * sizes['train_cue_share'])
    n_ux = round(n_tr * sizes['train_cue_share'])
    f_pool = [r for r in pool() if _matches(r, FEATURE_CUES)]
    tf = take(rng.sample(f_pool, min(n_feat, len(f_pool))))
    u_pool = [r for r in pool() if _matches(r, UX_CUES)]
    tu = take(rng.sample(u_pool, min(n_ux, len(u_pool))))
    tr = take(rng.sample(pool(), n_tr - len(tf) - len(tu)))
    extra = tf + tu + tr
    audit = random.Random(seed + 1).sample(extra, round(len(extra) * sizes['train_audit_share']))
    out['train'] = {'rows': [by_id[i] for i in calib_ids] + extra, 'meta': {
        'size': len(calib_ids) + len(extra),
        'composition': {'calibration (labelled)': len(calib_ids), 'extra (to label)': len(extra),
                        'extra: keyword FEATURE cues': len(tf), 'extra: keyword UX cues': len(tu), 'extra: random': len(tr)},
        'cue_pool_sizes': {'feature': len(f_pool), 'ux': len(u_pool)},
        'sampling': 'random + keyword enrichment for rare classes (keywords are not labels)',
        'label_mode': 'single (main annotator) + double-label audit on a random 15% of the extra',
        'locked': False, 'to_label': [r['review_id'] for r in extra],
        'sampling_source': {**{r['review_id']: 'keyword_enriched_feature' for r in tf},
                            **{r['review_id']: 'keyword_enriched_ux' for r in tu},
                            **{r['review_id']: 'random' for r in tr},
                            **{i: 'calibration' for i in calib_ids}},
        'audit_ids': [r['review_id'] for r in audit]}}
    # 5. Lazada external test (pure random)
    lz = random.Random(seed).sample(lazada, sizes['lazada'])
    out['lazada_external_test'] = {'rows': lz, 'meta': {
        'size': len(lz), 'sampling': 'pure random from Lazada interim', 'label_mode': 'double', 'locked': True,
        'to_label': [r['review_id'] for r in lz]}}

    for name, s in out.items():
        if len(s['rows']) != s['meta']['size']:
            raise ValueError(f'{name}: số dòng không khớp')
    return out


def rebalance_validation(prov, target: dict, protected_ids, seed: int = 42) -> dict:
    """Amendment v1.8 — top up rare classes in Validation by swapping with Train (both unlocked).
    For each class in `target` ({'FEATURE': 22}): move k = target − current reviews of that class
    Train → Validation and k Validation OTHER reviews → Train, so both sizes stay fixed.
    Train candidates exclude `protected_ids` (calibration + audit stay in Train) and prefer
    randomly sampled reviews over keyword-enriched ones (keeps Validation close to natural).
    prov: DataFrame of provenance. Returns {'to_validation': [...], 'to_train': [...], 'detail': {...}}."""
    rng = random.Random(seed)
    protected = set(protected_ids)
    tr = prov[prov['split'] == 'train']
    va = prov[prov['split'] == 'validation']
    to_val, to_train, detail = [], [], {}
    for cls, want in target.items():
        k = want - int((va['gold_label'] == cls).sum())
        if k <= 0:
            detail[cls] = 'đã đủ'
            continue
        cand = tr[(tr['gold_label'] == cls) & ~tr['review_id'].isin(protected) & ~tr['review_id'].isin(to_val)]
        rand = sorted(cand[cand['sampling_source'] == 'random']['review_id'])
        enr = sorted(cand[cand['sampling_source'] != 'random']['review_id'])
        pick = rng.sample(rand, min(k, len(rand)))
        pick += rng.sample(enr, min(k - len(pick), len(enr)))
        if len(pick) < k:
            raise ValueError(f'Không đủ {cls} trong Train để chuyển ({len(pick)}/{k})')
        back_pool = sorted(va[(va['gold_label'] == 'OTHER') & ~va['review_id'].isin(to_train)]['review_id'])
        back = rng.sample(back_pool, k)
        to_val += pick
        to_train += back
        detail[cls] = {'moved_to_validation': k, 'from_random': min(k, len(rand)), 'from_enriched': k - min(k, len(rand)),
                       'other_moved_to_train': k}
    return {'to_validation': to_val, 'to_train': to_train, 'detail': detail}


def apply_split_moves(prov_path: str, moves: dict) -> int:
    """Rewrite provenance.csv changing only `split` for the moved review_ids (Train ↔ Validation)."""
    with open(prov_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    to_val, to_tr = set(moves['to_validation']), set(moves['to_train'])
    changed = 0
    for r in rows:
        if r['review_id'] in to_val:
            assert r['split'] == 'train', r['review_id']
            r['split'], changed = 'validation', changed + 1
        elif r['review_id'] in to_tr:
            assert r['split'] == 'validation', r['review_id']
            r['split'], changed = 'train', changed + 1
    with open(prov_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return changed


def overlap_matrix(splits: dict, extra_sets: dict) -> dict:
    """Pairwise |intersection| of review_id sets across splits + named extra sets (calibration, pilot)."""
    sets = {k: {r['review_id'] for r in v['rows']} for k, v in splits.items()}
    sets.update({k: set(v) for k, v in extra_sets.items()})
    return {f'{a} ∩ {b}': len(sets[a] & sets[b]) for a, b in combinations(sets, 2)}


def write_split_manifest(splits: dict, overlaps: dict, sources: dict, sheets: dict, path: str, seed: int = 42) -> dict:
    manifest = {
        'stage': 'GĐ3/GĐ4 splits (amendment v1.7)', 'seed': seed,
        'sources': {k: {'path': p, 'sha256': _sha256(p)} for k, p in sources.items()},
        'sets': {k: {**{m: v for m, v in s['meta'].items()},
                     'review_ids': [r['review_id'] for r in s['rows']],
                     'sheet_to_label': sheets.get(k)} for k, s in splits.items()},
        'overlaps': overlaps,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
