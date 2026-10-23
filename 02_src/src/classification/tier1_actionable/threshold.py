"""Tier 1 decision threshold (Q5, protocol v1.1): among thresholds with Recall_ACTIONABLE ≥ 0.90 on
Validation, take the one with the highest Macro-F1; if none reaches 0.90, take the threshold with the
highest recall and report that the constraint was not met. Chosen on the canonical seed-42 run (Q6)
and then frozen for the Hold-out evaluation (GĐ10)."""
import numpy as np
import pandas as pd

from src.classification.learning_curve import tier1_metrics

MIN_RECALL = 0.90


def sweep(y_true, p_actionable, lo=0.10, hi=0.90, step=0.01):
    y = np.asarray(y_true)
    p = np.asarray(p_actionable)
    rows = []
    for t in np.round(np.arange(lo, hi + step / 2, step), 4):
        rows.append({'threshold': float(t), **tier1_metrics(y, (p >= t).astype(int))})
    return pd.DataFrame(rows)


def choose(sw, min_recall=MIN_RECALL):
    ok = sw[sw['recall_actionable'] >= min_recall]
    if len(ok):
        best = ok.sort_values(['macro_f1', 'threshold'], ascending=[False, False], kind='mergesort').iloc[0]
        met = True
    else:
        best = sw.sort_values(['recall_actionable', 'macro_f1'], ascending=[False, False], kind='mergesort').iloc[0]
        met = False
    return {'threshold': float(best['threshold']), 'recall_constraint_met': met, 'min_recall': min_recall,
            **{k: float(best[k]) for k in ('recall_actionable', 'precision_actionable', 'f1_actionable', 'macro_f1')}}


def metrics_at(y_true, p_actionable, threshold):
    return tier1_metrics(np.asarray(y_true), (np.asarray(p_actionable) >= threshold).astype(int))


def plot_threshold_curves(sweeps: dict, chosen: dict, path, min_recall=MIN_RECALL):
    """One panel per strategy: precision, recall and Macro-F1 vs threshold, recall floor, chosen threshold."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'recall_actionable': ('#2a78d6', 'Recall ACTIONABLE'), 'precision_actionable': ('#eb6834', 'Precision ACTIONABLE'),
              'macro_f1': ('#1baf7a', 'Macro-F1')}
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.2 * len(sweeps), 3.6), facecolor='#fcfcfb', squeeze=False)
    for ax, (name, sw) in zip(axes[0], sweeps.items()):
        ax.set_facecolor('#fcfcfb')
        for col, (c, lab) in colors.items():
            ax.plot(sw['threshold'], sw[col] * 100, color=c, linewidth=2, label=lab)
        ax.axhline(min_recall * 100, color='#52514e', linewidth=0.8, linestyle='--')
        ax.text(sw['threshold'].min(), min_recall * 100 + 1, f'recall ≥ {min_recall:.2f}', fontsize=7.5, color='#52514e')
        t = chosen[name]['threshold']
        ax.axvline(t, color='#0b0b0b', linewidth=1)
        ax.text(t + 0.01, 5, f'ngưỡng {t:.2f}', fontsize=8, color='#0b0b0b')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Ngưỡng xác suất ACTIONABLE', color='#52514e')
        ax.set_ylabel('%', color='#52514e')
        ax.set_ylim(0, 102)
        ax.grid(True, color='#e4e3df', linewidth=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors='#52514e', labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path
