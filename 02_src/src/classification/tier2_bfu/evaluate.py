"""Tier 2 evaluation on gold-actionable reviews: Macro-F1 (3 classes) + per-class P/R/F1 (protocol, CLAUDE.md §6).

The same functions serve every evaluation set: Validation-actionable in GĐ7, and the locked Hold-out and
focused FEATURE/UX set in GĐ10 (from the checkpoints saved in GĐ7). Each set is reported in its own table.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.classification.tier2_bfu.mapping import TIER2_LABELS

LABEL_IDS = list(range(len(TIER2_LABELS)))


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS, zero_division=0)
    out = {'macro_f1': float(f.mean())}
    for i, c in enumerate(TIER2_LABELS):
        out[f'f1_{c}'], out[f'precision_{c}'], out[f'recall_{c}'] = float(f[i]), float(p[i]), float(r[i])
    return out


def argmax_metrics(y_true, probs):
    """metric_fn for rq1.grid_search: Tier 2 has no threshold, the predicted class is the argmax."""
    return metrics(np.asarray(y_true), np.asarray(probs).argmax(1))


def _runs_metrics(y_true, runs):
    return pd.DataFrame([argmax_metrics(y_true, r['probs']) for r in runs])


def _pm(s):
    return f"{s.mean() * 100:.1f} ± {s.std(ddof=1) * 100:.1f}" if len(s) > 1 else f"{s.iloc[0] * 100:.1f}"


def summary_row(y_true, runs, strategy, overrides, set_name):
    """Macro-F1 + per-class F1, mean ± std (%) over seeds."""
    m = _runs_metrics(y_true, runs)
    counts = pd.Series(y_true).map(dict(enumerate(TIER2_LABELS))).value_counts()
    row = {'tập': set_name, 'strategy': strategy, 'cấu hình': json.dumps(overrides, sort_keys=True),
           'n': len(y_true), **{f'n_{c}': int(counts.get(c, 0)) for c in TIER2_LABELS}}
    row['macro_f1'] = _pm(m['macro_f1'])
    for c in TIER2_LABELS:
        row[f'f1_{c}'] = _pm(m[f'f1_{c}'])
    return row


def per_class_table(y_true, runs, strategy, set_name):
    """Precision / recall / F1 per class, mean ± std (%) over seeds."""
    m = _runs_metrics(y_true, runs)
    return pd.DataFrame([{'tập': set_name, 'strategy': strategy, 'lớp': c,
                          'precision': _pm(m[f'precision_{c}']), 'recall': _pm(m[f'recall_{c}']),
                          'f1': _pm(m[f'f1_{c}'])} for c in TIER2_LABELS])


def confusion(y_true, probs):
    """Gold (rows) × predicted (columns) for one run."""
    cm = confusion_matrix(y_true, np.asarray(probs).argmax(1), labels=LABEL_IDS)
    return pd.DataFrame(cm, index=[f'gold {c}' for c in TIER2_LABELS], columns=[f'đoán {c}' for c in TIER2_LABELS])
