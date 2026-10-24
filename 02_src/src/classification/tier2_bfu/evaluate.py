"""Tier 2 evaluation on gold-actionable reviews: Macro-F1 (3 classes) + per-class P/R/F1 (protocol, CLAUDE.md §6).

The same functions serve every evaluation set: Validation-actionable in GĐ7, and the locked Hold-out and
focused FEATURE/UX set in GĐ10 (from the checkpoints saved in GĐ7). Each set is reported in its own table.
Thin wrappers over `src.classification.evaluate` with the Tier 2 classes.
"""
import numpy as np

from src.classification import evaluate as E
from src.classification.tier2_bfu.mapping import TIER2_LABELS


def metrics(y_true, y_pred):
    return E.metrics(y_true, y_pred, TIER2_LABELS)


def argmax_metrics(y_true, probs):
    """metric_fn for rq1.grid_search: Tier 2 has no threshold, the predicted class is the argmax."""
    return metrics(np.asarray(y_true), np.asarray(probs).argmax(1))


def summary_row(y_true, runs, strategy, overrides, set_name):
    """Macro-F1 + per-class F1, mean ± std (%) over seeds."""
    return E.summary_row(y_true, runs, strategy, overrides, set_name, TIER2_LABELS)


def per_class_table(y_true, runs, strategy, set_name):
    """Precision / recall / F1 per class, mean ± std (%) over seeds."""
    return E.per_class_table(y_true, runs, strategy, set_name, TIER2_LABELS)


def confusion(y_true, probs):
    """Gold (rows) × predicted (columns) for one run."""
    return E.confusion(y_true, np.asarray(probs).argmax(1), TIER2_LABELS)
