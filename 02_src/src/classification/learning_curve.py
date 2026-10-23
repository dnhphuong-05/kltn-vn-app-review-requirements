"""Learning curve (ĐC §7.2.4; protocol: Tier 1 actionable-F1, Tier 2 per-class F1; 3 seeds).

Subsets are NESTED and stratified: one fixed (subset_seed) ordering of Train is built so that every
prefix keeps the class proportions; size n = first n reviews. The model seed varies (42/1337/2024),
the subsets do not — so the curve reflects data size, not resampling noise. Validation is fixed.
Evaluation uses Validation only (never Hold-out). The orchestration is strategy-agnostic: it takes
`train_fn(df, n_classes, label_names, seed) -> model` and `predict_fn(model, texts) -> pred_ids`.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.classification.tier1_actionable.mapping import TIER1_LABELS, to_tier1
from src.classification.tier2_bfu.mapping import TIER2_LABELS, to_tier2

KEY_METRICS = [('tier1', 'f1_actionable'), ('tier2', 'macro_f1'), ('tier2', 'f1_FEATURE')]


def nested_stratified_order(df, label_col='gold_label', seed=42):
    """Row order whose every prefix is (approximately) stratified by `label_col`."""
    rng = np.random.default_rng(seed)
    keys = pd.Series(0.0, index=df.index)
    for _, idx in df.groupby(label_col).groups.items():
        idx = list(idx)
        perm = rng.permutation(len(idx))
        jitter = rng.random(len(idx))
        keys.loc[[idx[i] for i in perm]] = (np.arange(len(idx)) + jitter) / len(idx)
    return df.loc[keys.sort_values(kind='mergesort').index]


def nested_subsets(df, sizes, label_col='gold_label', seed=42):
    ordered = nested_stratified_order(df, label_col, seed)
    if max(sizes) > len(ordered):
        raise ValueError(f'Mốc {max(sizes)} lớn hơn Train ({len(ordered)})')
    return {n: ordered.iloc[:n] for n in sizes}


def tier1_metrics(y_true, y_pred):
    return {'f1_actionable': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            'recall_actionable': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'precision_actionable': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', labels=[0, 1], zero_division=0)}


def tier2_metrics(y_true, y_pred):
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    out = {'macro_f1': f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2], zero_division=0)}
    out.update({f'f1_{c}': float(v) for c, v in zip(TIER2_LABELS, per)})
    return out


def run_learning_curve(train_df, val_df, sizes, seeds, train_fn, predict_fn, subset_seed=42,
                       tiers=('tier1', 'tier2'), strategy='visobert', log=print, checkpoint_csv=None):
    """Returns a long DataFrame: strategy, size, n_train_tier, seed, tier, metric, value, train_seconds.
    With `checkpoint_csv`, every finished (size, tier, seed) run is appended to that file at once and
    runs already present are skipped — a Kaggle session that stops mid-way can resume."""
    import os
    subsets = nested_subsets(train_df, sizes, seed=subset_seed)
    val = {'tier1': to_tier1(val_df), 'tier2': to_tier2(val_df, drop_ambiguous=False)}
    spec = {'tier1': (to_tier1, TIER1_LABELS, tier1_metrics),
            'tier2': (lambda d: to_tier2(d, drop_ambiguous=True), TIER2_LABELS, tier2_metrics)}
    rows, done = [], set()
    if checkpoint_csv and os.path.exists(checkpoint_csv):
        prev = pd.read_csv(checkpoint_csv)
        prev = prev[prev['strategy'] == strategy]
        rows = prev.to_dict('records')
        done = set(zip(prev['size'], prev['tier'], prev['seed']))
        if done:
            log(f'[{strategy}] tiếp tục từ checkpoint: đã có {len(done)} lần chạy')
    for size in sizes:
        for tier in tiers:
            mapper, names, metric_fn = spec[tier]
            tr = mapper(subsets[size])
            for seed in seeds:
                if (size, tier, seed) in done:
                    continue
                t0 = time.time()
                model = train_fn(tr, len(names), names, seed)
                seconds = time.time() - t0
                pred = predict_fn(model, val[tier]['text'].astype(str).tolist())
                new = [{'strategy': strategy, 'size': size, 'n_train_tier': len(tr), 'seed': seed, 'tier': tier,
                        'metric': metric, 'value': float(value), 'train_seconds': seconds}
                       for metric, value in metric_fn(val[tier]['y'].tolist(), pred).items()]
                rows += new
                if checkpoint_csv:
                    pd.DataFrame(new).to_csv(checkpoint_csv, mode='a', index=False,
                                             header=not os.path.exists(checkpoint_csv))
                key = 'f1_actionable' if tier == 'tier1' else 'macro_f1'
                last = [r['value'] for r in rows if r['size'] == size and r['tier'] == tier
                        and r['seed'] == seed and r['metric'] == key][-1]
                log(f'[{strategy}] size={size} {tier} (n={len(tr)}) seed={seed}: {key}={last:.3f} ({seconds:.0f}s)')
                del model
                _free_gpu()
    return pd.DataFrame(rows)


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def summarize(raw):
    """mean ± std over seeds for each (strategy, size, tier, metric)."""
    g = raw.groupby(['strategy', 'size', 'tier', 'metric'])['value']
    out = g.agg(mean='mean', std='std', n_seeds='count').reset_index()
    out['std'] = out['std'].fillna(0.0)
    return out


def stopping_evidence(summary, strategy='visobert', min_gain=0.01):
    """Facts for the plateau rule (protocol: < ~1 point gain at two consecutive increases, seed bands
    overlapping, and BUG/FEATURE/UX F1 no longer rising materially). Reports — does not decide."""
    s = summary[summary['strategy'] == strategy]
    sizes = sorted(s['size'].unique())
    lines = []
    for tier, metric in KEY_METRICS + [('tier2', 'f1_BUG'), ('tier2', 'f1_UX'), ('tier1', 'recall_actionable')]:
        m = s[(s['tier'] == tier) & (s['metric'] == metric)].set_index('size').reindex(sizes)
        if m['mean'].isna().all():
            continue
        parts = [f"{n}: {m.loc[n, 'mean'] * 100:.1f} ± {m.loc[n, 'std'] * 100:.1f}" for n in sizes]
        gains = []
        for a, b in zip(sizes, sizes[1:]):
            d = m.loc[b, 'mean'] - m.loc[a, 'mean']
            overlap = abs(d) <= m.loc[a, 'std'] + m.loc[b, 'std']
            gains.append(f'{a}→{b}: {d * 100:+.1f} điểm ({"< 1 điểm" if abs(d) < min_gain else "≥ 1 điểm"}, '
                         f'dải ±std {"chồng lấp" if overlap else "không chồng lấp"})')
        lines.append(f'{tier} {metric} — ' + ' | '.join(parts) + '\n    ' + ' ; '.join(gains))
    return lines


def plot_learning_curve(summary, path, strategy='visobert', title=None):
    """Static PNG: x = Train size, y = F1 (%). Tier 1 actionable-F1, Tier 2 macro-F1, Tier 2 FEATURE F1,
    with ±std bands. Colours: validated reference palette slots 1–3 (all-pairs safe)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = [('tier1', 'f1_actionable', 'Tầng 1 — F1 ACTIONABLE', '#2a78d6', 'o'),
              ('tier2', 'macro_f1', 'Tầng 2 — Macro-F1 (3 lớp)', '#eb6834', 's'),
              ('tier2', 'f1_FEATURE', 'Tầng 2 — F1 FEATURE', '#1baf7a', '^')]
    s = summary[summary['strategy'] == strategy]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for tier, metric, label, color, marker in series:
        m = s[(s['tier'] == tier) & (s['metric'] == metric)].sort_values('size')
        if m.empty:
            continue
        x, y, e = m['size'], m['mean'] * 100, m['std'] * 100
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=7, label=label)
        ax.annotate(f'{y.iloc[-1]:.1f}', (x.iloc[-1], y.iloc[-1]), textcoords='offset points', xytext=(6, -3),
                    fontsize=8, color='#52514e')
    ax.set_xticks(sorted(s['size'].unique()))
    ax.set_xlabel('Số review Train (subset lồng nhau, stratify)', color='#52514e')
    ax.set_ylabel('F1 trên Validation (%)', color='#52514e')
    ax.set_title(title or f'Learning curve — {strategy} (trung bình ± độ lệch chuẩn, 3 seed)', fontsize=10, fontweight='bold')
    ax.grid(True, color='#e4e3df', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#e4e3df')
    ax.tick_params(colors='#52514e', labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path
