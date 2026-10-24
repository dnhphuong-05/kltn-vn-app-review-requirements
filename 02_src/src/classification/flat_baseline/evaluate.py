"""Flat baseline evaluation: Macro-F1 over 4 classes + per-class P/R/F1 + 4×4 confusion (end-to-end metric,
the same one used to score the cascade). Thin wrappers over `src.classification.evaluate`."""
import numpy as np

from src.classification import evaluate as E
from src.classification.flat_baseline.mapping import FLAT_LABELS


def metrics(y_true, y_pred):
    return E.metrics(y_true, y_pred, FLAT_LABELS)


def argmax_metrics(y_true, probs):
    """metric_fn for rq1.grid_search (no threshold: argmax over 4 classes)."""
    return metrics(np.asarray(y_true), np.asarray(probs).argmax(1))


def summary_row(y_true, runs, strategy, overrides, set_name):
    return E.summary_row(y_true, runs, strategy, overrides, set_name, FLAT_LABELS)


def per_class_table(y_true, runs, strategy, set_name):
    return E.per_class_table(y_true, runs, strategy, set_name, FLAT_LABELS)


def confusion_sum(y_true, runs):
    return E.confusion_sum(y_true, runs, FLAT_LABELS)
