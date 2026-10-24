"""Evaluation shared by every tier and strategy: multi-class P/R/F1 tables (mean ± std over seeds), confusion
matrices, end-to-end 4 classes and the cascade combination (Tier 1 decision + Tier 2 label → BUG/FEATURE/UX/OTHER).

Class ids: Tier 2 uses BUG=0, FEATURE=1, UX=2; the 4-class space adds OTHER=3, so a Tier 2 id is already a
valid 4-class id. A run is {'seed', 'probs'} (prediction = argmax) or {'seed', 'pred'} (class ids).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

FOUR_CLASSES = ['BUG', 'FEATURE', 'UX', 'OTHER']
OTHER_ID = FOUR_CLASSES.index('OTHER')


def metrics(y_true, y_pred, labels):
    ids = list(range(len(labels)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    out = {'macro_f1': float(f.mean())}
    for i, c in enumerate(labels):
        out[f'f1_{c}'], out[f'precision_{c}'], out[f'recall_{c}'] = float(f[i]), float(p[i]), float(r[i])
    return out


def run_pred(run):
    return np.asarray(run['pred']) if 'pred' in run else np.asarray(run['probs']).argmax(1)


def runs_metrics(y_true, runs, labels):
    return pd.DataFrame([metrics(y_true, run_pred(r), labels) for r in runs])


def _pm(s):
    return f"{s.mean() * 100:.1f} ± {s.std(ddof=1) * 100:.1f}" if len(s) > 1 else f"{s.iloc[0] * 100:.1f}"


def summary_row(y_true, runs, strategy, overrides, set_name, labels):
    """Macro-F1 + per-class F1, mean ± std (%) over seeds."""
    m = runs_metrics(y_true, runs, labels)
    counts = pd.Series(y_true).map(dict(enumerate(labels))).value_counts()
    row = {'tập': set_name, 'strategy': strategy,
           'cấu hình': overrides if isinstance(overrides, str) else json.dumps(overrides, sort_keys=True),
           'n': len(y_true), **{f'n_{c}': int(counts.get(c, 0)) for c in labels}}
    row['macro_f1'] = _pm(m['macro_f1'])
    for c in labels:
        row[f'f1_{c}'] = _pm(m[f'f1_{c}'])
    return row


def per_class_table(y_true, runs, strategy, set_name, labels):
    """Precision / recall / F1 per class, mean ± std (%) over seeds."""
    m = runs_metrics(y_true, runs, labels)
    return pd.DataFrame([{'tập': set_name, 'strategy': strategy, 'lớp': c,
                          'precision': _pm(m[f'precision_{c}']), 'recall': _pm(m[f'recall_{c}']),
                          'f1': _pm(m[f'f1_{c}'])} for c in labels])


def confusion(y_true, y_pred, labels):
    """Gold (rows) × predicted (columns) from class ids."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return pd.DataFrame(cm, index=[f'gold {c}' for c in labels], columns=[f'đoán {c}' for c in labels])


def confusion_sum(y_true, runs, labels):
    """Confusion summed over runs (divide by len(runs) for the per-run average)."""
    return sum(confusion(y_true, run_pred(r), labels) for r in runs)


def cascade_labels(tier1_actionable, tier2_pred):
    """4-class ids: Tier 1 NOT_ACTIONABLE → OTHER, otherwise the Tier 2 label (BUG/FEATURE/UX ids)."""
    act = np.asarray(tier1_actionable, dtype=bool)
    return np.where(act, np.asarray(tier2_pred), OTHER_ID)
