"""Paper revision (amendment v4.3): every main table of the manuscript from one source.

Primary estimate = the canonical run of a configuration: seed 42 for trained models, the single deterministic run
for the LLM (temperature 0); hybrids and routers use the canonical runs of their parts. 95% CIs are the paired
percentile bootstraps of `paper_analysis` / `final_eval` (10,000 resamples, seed 42). Means ± SD over the three seeds
appear only in the per-seed appendix table. Configuration names are translated to the English names of the
manuscript. No retraining, no API call, no new prediction: everything is rebuilt from saved predictions, and the
notebook checks every rebuilt number against the files of steps 2, 4 and 5.
"""
import json
import os
import re

import numpy as np
import pandas as pd

from src.classification import fair_controls as FC
from src.classification import paper_analysis as PA
from src.classification.evaluate import FOUR_CLASSES, OTHER_ID, metrics
from src.classification.final_eval import bootstrap_table

TIER2 = ['BUG', 'FEATURE', 'UX']
SEED_GAP_FLAG = 0.02            # per-seed table flags |mean − seed 42| above 2 points of Macro-F1
GATE_TOL = 1e-9

# ---------------------------------------------------------------- names
_EN = [(' (θ_max)', ' (θmax)'), (' (θ_98)', ' (θ98)'), ('LLM few-shot', 'LLM'), ('Bi-LSTM + fastText', 'Bi-LSTM'),
       (' (ngưỡng)', ' (thresholded)'), (' (bất đồng)', ' (disagreement)'),
       (' (độ tin cậy, ngân sách bất đồng)', ' (confidence, disagreement budget)'), (' (hiệu chỉnh)', ' (calibrated)'),
       (' [tỷ lệ Lazada]', ' [Lazada rate]')]
SET_EN = {'holdout': 'Shopee hold-out', 'lazada': 'Lazada', 'focused': 'Shopee focused'}


def en(name):
    """English manuscript name of a configuration (Vietnamese names are used in the result files)."""
    s = str(name)
    if s.startswith('Phẳng '):
        s = 'Flat ' + s[len('Phẳng '):]
    for a, b in _EN:
        s = s.replace(a, b)
    return re.sub(r' \(độ tin cậy, (\d+)%\)', r' (confidence, \1% budget)', s)


def architecture(name):
    s = str(name)
    if s.startswith('Phẳng '):
        return 'flat (thresholded)' if '(ngưỡng)' in s else 'flat'
    if ' ⇢ ' in s:
        return 'routing'
    if s.startswith('Cascade '):
        return 'cascade (calibrated)' if '(hiệu chỉnh)' in s else 'cascade'
    return 'hybrid' if ' → ' in s else 'other'


# ---------------------------------------------------------------- canonical-run metrics
def canon_metrics(y, pred):
    """Four-class Macro-F1, per-class precision/recall/F1, ACTIONABLE precision/recall/F1, forwarded share and
    accuracy of one prediction vector."""
    y, p = np.asarray(y), np.asarray(pred)
    m = metrics(y, p, FOUR_CLASSES)
    g, a = y != OTHER_ID, p != OTHER_ID
    tp = int((g & a).sum())
    rec, prec = tp / max(int(g.sum()), 1), tp / max(int(a.sum()), 1)
    return {'Macro-F1': m['macro_f1'], **{f'F1 {c}': m[f'f1_{c}'] for c in FOUR_CLASSES},
            **{f'P {c}': m[f'precision_{c}'] for c in FOUR_CLASSES}, **{f'R {c}': m[f'recall_{c}'] for c in FOUR_CLASSES},
            'Precision ACT': prec, 'Recall ACT': rec, 'F1 ACT': 2 * rec * prec / max(rec + prec, 1e-12),
            'Forwarded': float(a.mean()), 'Accuracy': float((p == y).mean())}


def e2e_table(y4, configs, mask=None):
    """(table, diffs, y, canon). The table has one row per configuration with canonical-run metrics and the 95% CI
    of Macro-F1; `key` keeps the Vietnamese result-file name used by the gates."""
    tab, diff, y, canon = PA.e2e_scores(y4, configs, mask)
    t = tab.set_index('cấu hình')
    rows = []
    for name in tab['cấu hình']:
        r = canon_metrics(y, canon[name])
        if abs(r['Macro-F1'] - t.loc[name, 'Macro-F1 chuẩn']) > GATE_TOL:
            raise AssertionError(f'{name}: Macro-F1 lần chạy chuẩn lệch bootstrap_table')
        rows.append({'key': name, 'Configuration': en(name), 'Architecture': architecture(name), 'Runs': len(configs[name]),
                     'n': int(len(y)), 'Macro-F1': r.pop('Macro-F1'), 'CI low': t.loc[name, 'CI 2,5%'],
                     'CI high': t.loc[name, 'CI 97,5%'], **r})
    return pd.DataFrame(rows), diff, y, canon


def tier1_table(y4, t1, mask=None):
    """Tier 1 on every review: ACTIONABLE F1, recall and precision (canonical run, 95% CI) and two-class Macro-F1."""
    m = np.ones(len(y4), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y1 = (np.asarray(y4)[m] != OTHER_ID).astype(int)
    canon = {k: np.asarray(PA.canonical(v))[m].astype(int) for k, v in t1.items()}
    out = pd.DataFrame({'key': list(canon), 'Strategy': [en(k) for k in canon], 'n': int(m.sum())})
    for met, col in (('f1', 'F1 ACT'), ('recall', 'Recall ACT'), ('precision', 'Precision ACT')):
        ci = bootstrap_table(y1, canon, 2, met, 1)[0].set_index('cấu hình')
        out[col] = out['key'].map(ci['giá trị'])
        out[f'{col} CI low'] = out['key'].map(ci['CI 2,5%'])
        out[f'{col} CI high'] = out['key'].map(ci['CI 97,5%'])
    out['Macro-F1 (2 classes)'] = out['key'].map(lambda k: metrics(y1, canon[k], ['NOT_ACTIONABLE', 'ACTIONABLE'])['macro_f1'])
    out['Forwarded'] = out['key'].map(lambda k: float(canon[k].mean()))
    return out


def tier2_table(y4, t2, mask=None):
    """Tier 2 on gold-actionable reviews: three-class Macro-F1 (canonical run, 95% CI) and per-class F1."""
    y4 = np.asarray(y4)
    m = (np.ones(len(y4), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)) & (y4 != OTHER_ID)
    y2 = y4[m]
    canon = {k: np.asarray(PA.canonical(v))[m] for k, v in t2.items()}
    for k, p in canon.items():
        if (p < 0).any():
            raise RuntimeError(f'{k}: tầng 2 thiếu dự đoán trên {(p < 0).sum()} review gold-actionable')
    ci = bootstrap_table(y2, canon, 3)[0].set_index('cấu hình')
    rows = []
    for k, p in canon.items():
        r = metrics(y2, p, TIER2)
        rows.append({'key': k, 'Strategy': en(k), 'n': int(m.sum()), 'Macro-F1': ci.loc[k, 'giá trị'],
                     'CI low': ci.loc[k, 'CI 2,5%'], 'CI high': ci.loc[k, 'CI 97,5%'], **{f'F1 {c}': r[f'f1_{c}'] for c in TIER2}})
    return pd.DataFrame(rows)


def per_seed_table(y, configs, mask=None):
    """Macro-F1 of every run of every multi-run configuration, mean ± SD and the gap mean − seed 42."""
    m = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    yy, rows = np.asarray(y)[m], []
    for name, runs in configs.items():
        if len(runs) < 2:
            continue
        f = {k: metrics(yy, np.asarray(p)[m], FOUR_CLASSES)['macro_f1'] for k, p in runs.items()}
        v = np.array(list(f.values()))
        rows.append({'key': name, 'Configuration': en(name), **{f'Seed {k}': x for k, x in f.items()}, 'Mean': v.mean(),
                     'SD': v.std(ddof=1), 'Mean − seed 42': v.mean() - f[42],
                     'Flag (|gap| > 2 points)': bool(abs(v.mean() - f[42]) > SEED_GAP_FLAG)})
    return pd.DataFrame(rows)


def error_table(y, canon):
    """End-to-end errors of the canonical run by stage: missed actionable (gold BUG/FEATURE/UX → OTHER), kept OTHER
    (gold OTHER → BUG/FEATURE/UX) and wrong type (both actionable, different class)."""
    y, rows = np.asarray(y), []
    for name, p in canon.items():
        p = np.asarray(p)
        g, a = y != OTHER_ID, p != OTHER_ID
        c = {'Missed actionable': int((g & ~a).sum()), 'Kept OTHER': int((~g & a).sum()),
             'Wrong type': int((g & a & (p != y)).sum())}
        tot = sum(c.values())
        rows.append({'key': name, 'Configuration': en(name), 'n': len(y), 'Errors': tot, **c,
                     **{f'{k} (% of errors)': v / tot if tot else 0.0 for k, v in c.items()}})
    return pd.DataFrame(rows)


def confusion_table(y, canon, names):
    rows = []
    y = np.asarray(y)
    for name in names:
        p = np.asarray(canon[name])
        for i, g in enumerate(FOUR_CLASSES):
            rows.append({'key': name, 'Configuration': en(name), 'Gold': g,
                         **{f'Predicted {c}': int(((y == i) & (p == j)).sum()) for j, c in enumerate(FOUR_CLASSES)}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pairwise comparisons
def pair_table(y, diff, canon, pairs, family, holm_from=None):
    """Oriented paired bootstrap difference A − B of Macro-F1 (canonical runs) with 95% CI and exact McNemar on
    per-review correctness. p (Holm) comes from `holm_from` (the full registered family, columns A, B, p (Holm))
    when given, otherwise Holm is applied over `pairs` (which is then the registered family)."""
    rows = []
    for a, b in pairs:
        d, sg = PA._pair(diff, a, b)
        lo, hi = sorted([sg * d['CI 2,5%'], sg * d['CI 97,5%']])
        n01, n10, p = PA.mcnemar_exact(y, canon[a], canon[b])
        rows.append({'Family': family, 'key A': a, 'key B': b, 'A': en(a), 'B': en(b), 'Δ Macro-F1 (A − B)': sg * d['chênh (A − B)'],
                     'CI low': lo, 'CI high': hi, 'CI contains 0': bool(lo <= 0 <= hi), 'A right, B wrong': n01,
                     'A wrong, B right': n10, 'p (exact)': p})
    df = pd.DataFrame(rows)
    if holm_from is None:
        df['p (Holm)'] = PA.holm(df['p (exact)'])
    else:
        df['p (Holm)'] = [PA._pair(holm_from, a, b)[0]['p (Holm)'] for a, b in pairs]
    df['Holm family size'] = len(pairs) if holm_from is None else len(holm_from)
    return df


# ---------------------------------------------------------------- B1 / B2 configurations (as notebook 25)
def b1_configs(cfg, flat_probs, thresholds):
    """{name: {seed: prediction}}: flat argmax, thresholded flat and cascade of every trained strategy."""
    out = {}
    for x in FC.TRAINED:
        out[f'Phẳng {x}'] = cfg[f'Phẳng {x}']
        out[f'Phẳng {x} (ngưỡng)'] = {k: FC.flat_threshold_predict(p, thresholds[x]) for k, p in flat_probs[x].items()}
        out[f'Cascade {x}'] = cfg[f'Cascade {x}']
    return out


B2_CONF_MAX, B2_CONF_98 = 'PhoBERT ⇢ LLM (θ_max)', 'PhoBERT ⇢ LLM (θ_98)'
B2_DIS, B2_CONF_DIS = 'PhoBERT ⇢ LLM (bất đồng)', 'PhoBERT ⇢ LLM (độ tin cậy, ngân sách bất đồng)'


def b2_masks(pho, vis, theta, budgets, seeds):
    """{router: {seed: LLM call mask}} of the B2 routing policies (PhoBERT small model)."""
    n = len(next(iter(pho.values())))
    masks = {B2_CONF_MAX: {k: FC.confidence(p) < theta['theta_max'] for k, p in pho.items()},
             B2_CONF_98: {k: FC.confidence(p) < theta['theta_98'] for k, p in pho.items()},
             B2_DIS: {k: FC.disagreement_mask(pho[k], vis[k]) for k in seeds}}
    masks[B2_CONF_DIS] = {k: FC.lowest_confidence_mask(pho[k], int(masks[B2_DIS][k].sum())) for k in seeds}
    for b in budgets:
        masks[f'PhoBERT ⇢ LLM (độ tin cậy, {int(round(b * 100))}%)'] = {k: FC.lowest_confidence_mask(pho[k], int(round(b * n)))
                                                                      for k in seeds}
    return masks


def routed(pho, llm, masks):
    return {name: {k: FC.route_mask(pho[k].argmax(1), llm, m[k]) for k in m} for name, m in masks.items()}


# ---------------------------------------------------------------- data and configuration audit
def class_counts(sets):
    """{set name: DataFrame with gold_label} → counts per class, n and share actionable."""
    rows = []
    for s, d in sets.items():
        c = d['gold_label'].value_counts()
        rows.append({'Set': s, 'n': len(d), **{k: int(c.get(k, 0)) for k in FOUR_CLASSES},
                     'Actionable': int(len(d) - c.get('OTHER', 0)), 'Actionable (%)': (len(d) - c.get('OTHER', 0)) / len(d)})
    return pd.DataFrame(rows)


def id_overlaps(sets):
    names, rows = list(sets), []
    ids = {s: set(d['review_id']) for s, d in sets.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({'Set A': a, 'Set B': b, 'Shared review IDs': len(ids[a] & ids[b])})
    return pd.DataFrame(rows)


def fewshot_audit(prompt_dir, train_ids, other_ids):
    """Exemplars of every few-shot prompt: count, classes, membership in Train and in any other split."""
    rows, union = [], {}
    for v in ('v1', 'v1n'):
        for t in ('tier1', 'tier2', 'flat'):
            f = os.path.join(prompt_dir, f'{t}_exemplars_{v}.json')
            js = json.load(open(f, encoding='utf-8'))
            ex = js['exemplars']
            ids = [e['review_id'] for e in ex]
            union.setdefault(v, {}).update({t: set(ids)})
            c = pd.Series([e['gold_label'] for e in ex]).value_counts()
            rows.append({'Prompt version': v, 'Prompt': t, 'Exemplars': len(ids), 'Distinct IDs': len(set(ids)),
                         **{f'Gold {k}': int(c.get(k, 0)) for k in FOUR_CLASSES},
                         'All in Train': bool(set(ids) <= train_ids), 'In another split': len(set(ids) & other_ids),
                         'Selection seed': js.get('seed'), 'Selection rule': js.get('rule'), 'Source file': js.get('source')})
    df = pd.DataFrame(rows)
    summary = []
    for v, u in union.items():
        tiers = u['tier1'] | u['tier2']
        summary.append({'Prompt version': v, 'Distinct tier-1 + tier-2 exemplars': len(tiers),
                        'Flat exemplars reused from tiers': len(u['flat'] & tiers),
                        'Flat exemplars not in tier prompts': len(u['flat'] - tiers),
                        'Distinct exemplars, all prompts': len(tiers | u['flat'])})
    return df, pd.DataFrame(summary)


# ---------------------------------------------------------------- gates and presentation
def gate(new, old, cols, key_new='key', key_old='cấu hình', scale=1.0, what=''):
    """Assert that every value of `cols` (pairs new_col → old_col) matches the saved table, key by key."""
    o = old.set_index(key_old)
    n = new.set_index(key_new)
    keys = [k for k in n.index if k in o.index]
    if not keys:
        raise AssertionError(f'{what}: không có khóa chung')
    worst = 0.0
    for cn, co in cols.items():
        d = np.abs(n.loc[keys, cn].astype(float).to_numpy() * scale - o.loc[keys, co].astype(float).to_numpy())
        worst = max(worst, float(np.nanmax(d)))
    if worst > GATE_TOL * max(scale, 1.0):
        raise AssertionError(f'{what}: lệch tối đa {worst:.3g}')
    return len(keys), worst


def pct(df, cols, digits=1):
    """Copy with `cols` in percent, rounded."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = (out[c].astype(float) * 100).round(digits)
    return out


def drop_keys(df):
    return df[[c for c in df.columns if not c.startswith('key')]].copy()


def en_code(code):
    """English short code of a configuration (result files use 'X phẳng' for flat models)."""
    return str(code).replace(' phẳng', ' flat').replace('θ_max', 'θmax').replace('θ_98', 'θ98')


COLUMNS = {
    'Set': 'Evaluation set: Shopee hold-out (random sample, 1,000), Lazada (external app, 500) or Shopee focused (FEATURE/UX-enriched, 150).',
    'Configuration': 'Flat X = one four-class model; Cascade X = tier 1 X then tier 2 X; X → Y = tier 1 X, tier 2 Y; X ⇢ LLM = routing: X labels every review and the LLM relabels the reviews selected by the policy in brackets; (thresholded) = flat model with the recall-oriented ACTIONABLE threshold; (calibrated) = temperature-scaled tier 1.',
    'Strategy': 'Classifier family: TF-IDF + linear model, Bi-LSTM with pretrained fastText, ViSoBERT, PhoBERT, LLM (GPT-4o-mini few-shot, temperature 0).',
    'Architecture': 'flat, cascade, hybrid (tier-1 and tier-2 models differ), routing, thresholded flat or calibrated cascade.',
    'Runs': 'Number of runs available (3 seeds for trained models, 1 for the LLM and for hybrids). Only the canonical run enters the main tables.',
    'n': 'Number of reviews scored.',
    'Macro-F1': 'Unweighted mean of the per-class F1 of the canonical run (seed 42; single LLM run), %. Four classes end to end, three classes (BUG, FEATURE, UX on gold-actionable reviews) at tier 2.',
    'CI low': 'Lower bound of the 95% paired percentile bootstrap interval (10,000 resamples of reviews, seed 42), %.',
    'CI high': 'Upper bound of the 95% paired percentile bootstrap interval, %.',
    'F1 BUG': 'F1 of class BUG on the canonical run, %.', 'F1 FEATURE': 'F1 of class FEATURE on the canonical run, %.',
    'F1 UX': 'F1 of class UX on the canonical run, %.', 'F1 OTHER': 'F1 of class OTHER on the canonical run, %.',
    'Precision ACT': 'Share of reviews predicted BUG, FEATURE or UX whose gold label is actionable, canonical run, %.',
    'Recall ACT': 'Share of gold-actionable reviews predicted BUG, FEATURE or UX (any of the three), canonical run, %.',
    'F1 ACT': 'Harmonic mean of Precision ACT and Recall ACT, %.',
    'Forwarded': 'Share of reviews predicted actionable, i.e. forwarded to the development team, canonical run, %.',
    'Accuracy': 'Share of reviews with the correct four-class label, canonical run, %.',
    'Δ Lazada − hold-out': 'Macro-F1 on Lazada minus Macro-F1 on the Shopee hold-out (canonical runs), percentage points.',
    'Macro-F1 (2 classes)': 'Mean F1 of ACTIONABLE and NOT_ACTIONABLE at tier 1, canonical run, %.',
    'Family': 'Holm family registered before scoring; p (Holm) is adjusted over every pair of that family.',
    'A': 'First configuration of the pair.', 'B': 'Second configuration of the pair.',
    'Δ Macro-F1 (A − B)': 'Macro-F1 of A minus Macro-F1 of B on the canonical runs, percentage points.',
    'CI contains 0': 'True when the 95% bootstrap interval of the difference contains zero (no difference detected; this does not establish equivalence).',
    'A right, B wrong': 'Reviews labelled correctly by A and wrongly by B (McNemar discordant count).',
    'A wrong, B right': 'Reviews labelled wrongly by A and correctly by B (McNemar discordant count).',
    'p (exact)': 'Two-sided exact McNemar test on per-review correctness (not on Macro-F1).',
    'p (Holm)': 'Holm-adjusted p within the family.', 'Holm family size': 'Number of pairs in the Holm family.',
    'LLM share': 'Share of reviews sent to the LLM by the routing policy, canonical run, %.',
    'LLM share (3-seed mean)': 'Mean over the three seeds of the share sent to the LLM, %; reported because it varies by seed.',
    'API USD / 1,000 reviews': 'Measured API cost per 1,000 reviews of the LLM part (calls × measured tokens × price), USD.',
    'Random routing mean': 'Mean Macro-F1 of 1,000 random routings with the same number of LLM calls (numpy default_rng 20261121), %.',
    'Random routing SD': 'Standard deviation of Macro-F1 over the 1,000 random routings, percentage points.',
    'Random P2.5': '2.5th percentile of Macro-F1 over the random routings, %.', 'Random P97.5': '97.5th percentile, %.',
    'Share random ≥ router': 'Share of random routings whose Macro-F1 is at least that of the router (empirical one-sided p).',
    'Calls': 'Number of reviews sent to the LLM.',
    'Threshold t': 'ACTIONABLE threshold on P(ACT) = 1 − P(OTHER) chosen on validation (seed 42) with the guarded tier-1 rule: recall ≥ 0.80, then the highest two-class Macro-F1.',
    'Recall floor met on validation': 'Whether the chosen threshold reached ACTIONABLE recall ≥ 0.80 on validation.',
    'Seed 42': 'Macro-F1 of the run with seed 42 (canonical), %.', 'Seed 1337': 'Macro-F1 of the run with seed 1337, %.',
    'Seed 2024': 'Macro-F1 of the run with seed 2024, %.', 'Mean': 'Mean Macro-F1 over the three seeds, %.',
    'SD': 'Sample standard deviation over the three seeds, percentage points.',
    'Mean − seed 42': 'Three-seed mean minus the canonical run, percentage points.',
    'Flag (|gap| > 2 points)': 'True when the three-seed mean and the canonical run differ by more than 2 points.',
    'Errors': 'Number of wrongly labelled reviews of the canonical run.',
    'Missed actionable': 'Gold BUG/FEATURE/UX predicted OTHER.', 'Kept OTHER': 'Gold OTHER predicted BUG, FEATURE or UX.',
    'Wrong type': 'Gold and prediction both actionable but different classes.',
    'Missed actionable (% of errors)': 'Missed actionable as a share of all errors, %.',
    'Kept OTHER (% of errors)': 'Kept OTHER as a share of all errors, %.', 'Wrong type (% of errors)': 'Wrong type as a share of all errors, %.',
}
