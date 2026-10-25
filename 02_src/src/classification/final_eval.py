"""GĐ10 — final RQ1 evaluation helpers (amendment v2.1).

The locked sets (Hold-out, focused, Lazada) reach Kaggle as review_id + text only; gold labels stay local
and are read once, in the scoring notebook. This module covers:
  - text-only export of the locked sets and loading of the frozen configs (checked against the registry);
  - the step-0 gate: retrained models vs the Validation results reported in GĐ6/7/9;
  - vectorised paired bootstrap CIs for Macro-F1 / F1 / recall and for differences between configurations;
  - hybrid 3×3 (Tier 1 of X × Tier 2 of Y) from saved predictions.
"""
import os
import re

import numpy as np
import pandas as pd
import yaml

LOCKED = {'holdout': 'shopee_holdout.csv', 'focused': 'shopee_focused_feature_ux.csv',
          'lazada': 'lazada_external_test.csv'}
TEXT_ONLY = '{}_text_only.csv'
FROZEN_FILES = ('tier1', 'tier2', 'flat')


# ---------------------------------------------------------------- locked sets
def export_text_only(processed_dir, out_dir=None):
    """review_id + text of each locked set for the Kaggle upload. Never returns or prints labels."""
    out_dir = out_dir or processed_dir
    counts = {}
    for name, fname in LOCKED.items():
        df = pd.read_csv(os.path.join(processed_dir, fname), encoding='utf-8-sig', usecols=['review_id', 'text'])
        df.to_csv(os.path.join(out_dir, TEXT_ONLY.format(name)), index=False, encoding='utf-8-sig')
        counts[name] = len(df)
    return counts


# ---------------------------------------------------------------- frozen configs
def load_frozen(repo_dir):
    """{'tier1': {...}, 'tier2': {...}, 'flat': {...}} from 03_configs/frozen/, with prompt hashes checked
    against 04_prompts/prompts_registry.md. Raises on any missing key or mismatch."""
    fz = {}
    for k in FROZEN_FILES:
        path = os.path.join(repo_dir, '03_configs', 'frozen', f'{k}_config.yaml')
        head = open(path, encoding='utf-8').readline()
        if 'FROZEN' not in head or 'v2.1' not in head:
            raise RuntimeError(f'{path}: thiếu header FROZEN v2.1')
        fz[k] = yaml.safe_load(open(path, encoding='utf-8'))[k]
    reg = open(os.path.join(repo_dir, '04_prompts', 'prompts_registry.md'), encoding='utf-8').read()
    for tier, key in ((1, 'tier1'), (2, 'tier2')):
        m = re.search(rf'tier{tier}_system_v1\.txt[^|]*\|\s*v1\s*\|\s*([0-9a-f]{{64}})', reg)
        if not m or m.group(1) != fz[key]['llm']['prompt_hash']:
            raise RuntimeError(f'prompt_hash tầng {tier} trong config frozen khác registry')
    need = {'tier1': ('visobert', 'bilstm', 'threshold_rule', 'seeds'), 'tier2': ('visobert', 'bilstm', 'seeds'),
            'flat': ('visobert', 'seeds')}
    for k, keys in need.items():
        missing = [x for x in keys if x not in fz[k]]
        if missing:
            raise RuntimeError(f'{k}_config.yaml thiếu {missing}')
    return fz


# ---------------------------------------------------------------- step-0 gate
def parse_pm(s):
    """'85.0 ± 0.6' -> (85.0, 0.6) in percentage points."""
    a, b = str(s).split('±')
    return float(a), float(b)


def gate_table(rows, min_points=2.0):
    """rows: dicts {model, ref, ref_std, new_mean, new_std}. Pass iff |new − ref| ≤ max(min_points, ref_std)."""
    df = pd.DataFrame(rows)
    df['chênh'] = df['new_mean'] - df['ref']
    df['ngưỡng'] = np.maximum(min_points, df['ref_std'])
    df['đạt'] = df['chênh'].abs() <= df['ngưỡng']
    return df


# ---------------------------------------------------------------- bootstrap (paired, vectorised)
def bootstrap_index(n, n_boot=10_000, seed=42):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _counts(y, p, idx, k):
    """Confusion counts for every resample: [B, k, k] (rows = gold, cols = prediction; p = −1 is never a class)."""
    y, p = np.asarray(y), np.asarray(p)
    valid = (p >= 0)
    codes = np.where(valid, y * k + np.clip(p, 0, k - 1), k * k)          # invalid prediction -> extra bin
    b = idx.shape[0]
    flat = codes[idx] + (np.arange(b)[:, None] * (k * k + 1))
    c = np.bincount(flat.ravel(), minlength=b * (k * k + 1)).reshape(b, k * k + 1)
    invalid_by_gold = None
    if (~valid).any():                                                    # invalid predictions are false negatives
        gcodes = np.where(valid, k, y)
        gflat = gcodes[idx] + (np.arange(b)[:, None] * (k + 1))
        invalid_by_gold = np.bincount(gflat.ravel(), minlength=b * (k + 1)).reshape(b, k + 1)[:, :k]
    return c[:, :k * k].reshape(b, k, k), invalid_by_gold


def _f1_from_counts(cm, invalid_by_gold=None):
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=2) - tp
    if invalid_by_gold is not None:
        fn = fn + invalid_by_gold
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return f1, rec, prec


def metric_samples(y, p, idx, k, metric='macro_f1', cls=None):
    """Bootstrap distribution of a metric. metric: 'macro_f1' | 'f1' | 'recall' | 'precision' (of class `cls`)."""
    cm, inv = _counts(y, p, idx, k)
    f1, rec, prec = _f1_from_counts(cm, inv)
    if metric == 'macro_f1':
        return f1.mean(axis=1)
    return {'f1': f1, 'recall': rec, 'precision': prec}[metric][:, cls]


def point_metric(y, p, k, metric='macro_f1', cls=None):
    idx = np.arange(len(y))[None, :]
    return float(metric_samples(y, p, idx, k, metric, cls)[0])


def bootstrap_table(y, preds: dict, k, metric='macro_f1', cls=None, n_boot=10_000, seed=42):
    """Point estimate + 95% CI per configuration, and paired differences (row − column) with CI.
    The same resamples are used for every configuration, so differences are paired."""
    idx = bootstrap_index(len(y), n_boot, seed)
    samples = {name: metric_samples(y, p, idx, k, metric, cls) for name, p in preds.items()}
    rows = []
    for name, p in preds.items():
        s = samples[name]
        rows.append({'cấu hình': name, 'giá trị': point_metric(y, p, k, metric, cls),
                     'CI 2,5%': float(np.percentile(s, 2.5)), 'CI 97,5%': float(np.percentile(s, 97.5))})
    diffs = []
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = samples[a] - samples[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diffs.append({'A': a, 'B': b, 'chênh (A − B)': point_metric(y, preds[a], k, metric, cls)
                          - point_metric(y, preds[b], k, metric, cls),
                          'CI 2,5%': float(lo), 'CI 97,5%': float(hi), 'CI chứa 0': bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


# ---------------------------------------------------------------- hybrid 3×3
def hybrid_predictions(tier1_act: dict, tier2_pred: dict):
    """{(X, Y): 4-class ids} for every Tier 1 strategy X × Tier 2 strategy Y. Tier 2 arrays use −1 where the
    strategy was not run; a −1 on a review that Tier 1 kept is an error (coverage gap)."""
    from src.classification.evaluate import cascade_labels
    out = {}
    for x, act in tier1_act.items():
        act = np.asarray(act, dtype=bool)
        for y, t2 in tier2_pred.items():
            t2 = np.asarray(t2)
            gap = int(((t2 < 0) & act).sum())
            if gap:
                raise RuntimeError(f'Tầng 2 {y} thiếu dự đoán cho {gap} review mà tầng 1 {x} giữ lại')
            out[(x, y)] = cascade_labels(act, np.where(t2 < 0, 0, t2))
    return out


# ---------------------------------------------------------------- prediction tables (10a output, 10c input)
P_COLS = {'tier1': ['p_NOT_ACTIONABLE', 'p_ACTIONABLE'], 'tier2': ['p_BUG', 'p_FEATURE', 'p_UX'],
          'flat': ['p_BUG', 'p_FEATURE', 'p_UX', 'p_OTHER']}


def predictions_long(review_ids, runs: dict, set_name):
    """runs: {(tier, model, seed): run_multi output}. Long table tier, model, seed, review_id, p_<label>."""
    frames = []
    for (tier, model, seed), r in runs.items():
        d = pd.DataFrame(r['probs'][set_name], columns=P_COLS[tier])
        d.insert(0, 'review_id', list(review_ids))
        d.insert(0, 'seed', seed)
        d.insert(0, 'model', model)
        d.insert(0, 'tier', tier)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def neural_probs(pred_df, order_ids, tier, model, seeds):
    """{seed: probs aligned to order_ids}; raises if a review is missing."""
    d = pred_df[(pred_df['tier'] == tier) & (pred_df['model'] == model)]
    out = {}
    for s in seeds:
        x = d[d['seed'] == s].set_index('review_id')
        missing = set(order_ids) - set(x.index)
        if missing:
            raise RuntimeError(f'{tier}/{model}/seed {s}: thiếu {len(missing)} review')
        out[s] = x.loc[list(order_ids), P_COLS[tier]].to_numpy()
    return out


def llm_ids(order_ids, llm_df):
    """LLM class ids aligned to order_ids (−1 where the LLM was not called for that review)."""
    m = dict(zip(llm_df['review_id'], llm_df['pred']))
    return np.array([int(m.get(r, -1)) for r in order_ids])


def pm(values, scale=100):
    v = np.asarray(values, dtype=float) * scale
    return f'{v.mean():.1f} ± {v.std(ddof=1):.1f}' if len(v) > 1 else f'{v[0]:.1f}'
