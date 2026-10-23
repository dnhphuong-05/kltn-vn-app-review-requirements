"""Re-apply the Tier 1 threshold rule to saved Validation probabilities (no retraining).

Used after amendment v2.0 (recall ≥ 0.80, sweep 0.01–0.90): the trained models are unchanged, so the
threshold sweep, the chosen threshold, the results table and the curve plot are rebuilt from
tier1_<strategy>_val_predictions.csv (columns review_id, gold_y, seed, p_NOT_ACTIONABLE, p_ACTIONABLE).
"""
import os

import pandas as pd

from src.classification.tier1_actionable.threshold import (MIN_RECALL, choose, plot_threshold_curves,
                                                           summary_row, sweep)

DISPLAY = {'visobert': 'ViSoBERT', 'bilstm': 'Bi-LSTM + fastText'}
CANONICAL_SEED = 42


def runs_from_predictions(path, label_cols=('p_NOT_ACTIONABLE', 'p_ACTIONABLE')):
    """(y_true, runs) with the canonical seed first; every seed must cover the same reviews in the same order."""
    p = pd.read_csv(path)
    ref = p[p['seed'] == CANONICAL_SEED].reset_index(drop=True)
    if ref.empty:
        raise ValueError(f'{path}: không có seed {CANONICAL_SEED}')
    seeds = [CANONICAL_SEED] + sorted(s for s in p['seed'].unique() if s != CANONICAL_SEED)
    runs = []
    for s in seeds:
        g = p[p['seed'] == s].reset_index(drop=True)
        if not (g['review_id'].equals(ref['review_id']) and g['gold_y'].equals(ref['gold_y'])):
            raise ValueError(f'{path}: seed {s} không cùng danh sách review/nhãn với seed {CANONICAL_SEED}')
        runs.append({'seed': int(s), 'probs': g[list(label_cols)].to_numpy()})
    return ref['gold_y'].to_numpy(), runs


def rethreshold(pred_dir, out_dir, overrides: dict, min_recall=MIN_RECALL, lo=0.01, hi=0.90, step=0.01):
    """overrides = {strategy: winning config}. Writes tier1_<s>_threshold_sweep.csv, tier1_validation_results.csv
    and tier1_threshold_curves.png into out_dir. Returns (results DataFrame, {strategy: chosen})."""
    rows, sweeps, chosen = [], {}, {}
    for s, ov in overrides.items():
        y, runs = runs_from_predictions(os.path.join(pred_dir, f'tier1_{s}_val_predictions.csv'))
        sw = sweep(y, runs[0]['probs'][:, 1], lo=lo, hi=hi, step=step)       # seed 42 = lần chạy chuẩn (Q6)
        thr = choose(sw, min_recall=min_recall)
        sw.to_csv(os.path.join(out_dir, f'tier1_{s}_threshold_sweep.csv'), index=False)
        rows.append(summary_row(y, runs, thr, DISPLAY.get(s, s), ov, min_recall=min_recall))
        sweeps[DISPLAY.get(s, s)], chosen[s] = sw, thr
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'tier1_validation_results.csv'), index=False)
    plot_threshold_curves(sweeps, {DISPLAY.get(s, s): c for s, c in chosen.items()},
                          os.path.join(out_dir, 'tier1_threshold_curves.png'), min_recall=min_recall)
    return res, chosen
