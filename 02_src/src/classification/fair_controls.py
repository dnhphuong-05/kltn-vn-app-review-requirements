"""Paper revision (amendment v4.2): fair controls requested by the supervisor, from saved predictions only.

B1 — thresholded flat model: P(ACT) = 1 − P(OTHER) of the flat 4-class model; a review with P(ACT) ≥ t takes the most
probable of BUG/FEATURE/UX, otherwise OTHER. t is chosen on Validation (seed 42) by the guarded Tier 1 rule, so the
flat model and the cascade work under the same recall-oriented decision policy.

B2 — routing policies with the flat LLM as the expensive model and flat PhoBERT as the small one: confidence routing
(v4.1 thresholds), disagreement routing (PhoBERT and ViSoBERT flat argmax differ), random routing with the same number
of LLM calls, and confidence vs random routing at fixed budgets.

No retraining and no API call: every prediction comes from the saved files of steps 2 and 4.
"""
import os

import numpy as np
import pandas as pd

from src.classification.evaluate import FOUR_CLASSES, OTHER_ID, metrics
from src.classification.final_eval import neural_probs
from src.classification.tier1_actionable.threshold import choose_guarded, sweep

BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
RANDOM_DRAWS = 1000
RANDOM_SEED = 20261121
NEAR_MAIN, NEAR_SENS = 0.90, 0.80
TRAINED = {'TF-IDF': 'tfidf', 'Bi-LSTM + fastText': 'bilstm', 'ViSoBERT': 'visobert', 'PhoBERT': 'phobert'}
MAIN = 'PhoBERT'
ACT_COLS = [i for i in range(len(FOUR_CLASSES)) if i != OTHER_ID]


# ---------------------------------------------------------------- saved flat probabilities
def _flat_source(res_dir, set_name, model):
    if model == 'visobert':
        return os.path.join(res_dir, f'gd10_pred_{set_name}.csv')
    if model == 'tfidf':
        return os.path.join(res_dir, 'paper_tfidf_val_predictions.csv' if set_name == 'validation'
                            else f'paper_pred_tfidf_{set_name}.csv')
    return os.path.join(res_dir, f'paper_pred_neural_{set_name}.csv')      # phobert, flat bilstm (step 2)


def flat_probs(res_dir, set_name, ids, model, seeds):
    """{seed: 4-class probabilities (BUG, FEATURE, UX, OTHER) aligned to ids} of a saved flat model."""
    pred = pd.read_csv(_flat_source(res_dir, set_name, model), encoding='utf-8-sig')
    pred = pred[[c for c in pred.columns if c != 'gold_label']]
    return neural_probs(pred, ids, 'flat', model, seeds)


# ---------------------------------------------------------------- B1: thresholded flat model
def p_actionable(p4):
    return 1.0 - np.asarray(p4)[:, OTHER_ID]


def flat_threshold_predict(p4, t):
    """OTHER when P(ACT) < t, otherwise the most probable actionable class."""
    p = np.asarray(p4)
    act_cls = np.asarray(ACT_COLS)[p[:, ACT_COLS].argmax(1)]
    return np.where(p_actionable(p) >= t, act_cls, OTHER_ID)


def choose_flat_threshold(y4, p4):
    """Guarded Tier 1 rule (v2.0 + v3.5) applied to P(ACT) of a flat model on Validation."""
    y1 = (np.asarray(y4) != OTHER_ID).astype(int)
    return choose_guarded(sweep(y1, p_actionable(p4)))


# ---------------------------------------------------------------- B2: routing policies
def confidence(p4):
    return np.asarray(p4).max(1)


def route_mask(pred_small, llm_pred, mask):
    return np.where(mask, np.asarray(llm_pred), np.asarray(pred_small))


def disagreement_mask(p_small, p_other):
    return np.asarray(p_small).argmax(1) != np.asarray(p_other).argmax(1)


def lowest_confidence_mask(p4, k):
    """The k reviews with the lowest confidence (ties broken by position, stable)."""
    m = np.zeros(len(p4), dtype=bool)
    if k > 0:
        m[np.argsort(confidence(p4), kind='mergesort')[:k]] = True
    return m


def random_masks(n, k, draws=RANDOM_DRAWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        m = np.zeros(n, dtype=bool)
        if k > 0:
            m[rng.choice(n, size=k, replace=False)] = True
        yield m


def random_baseline(y, pred_small, llm_pred, k, ref_f1=None, draws=RANDOM_DRAWS, seed=RANDOM_SEED):
    """Macro-F1 of random routing with k LLM calls: mean, SD, 2.5/97.5 percentiles and the share of draws at least as
    good as ref_f1 (an empirical one-sided p-value for 'the router beats random routing')."""
    f = np.array([metrics(y, route_mask(pred_small, llm_pred, m), FOUR_CLASSES)['macro_f1']
                  for m in random_masks(len(y), k, draws, seed)])
    out = {'k': int(k), 'random mean': float(f.mean()), 'random SD': float(f.std(ddof=1)),
           'random P2.5': float(np.percentile(f, 2.5)), 'random P97.5': float(np.percentile(f, 97.5))}
    if ref_f1 is not None:
        out['share random ≥ router'] = float((f >= ref_f1 - 1e-12).mean())
    return out


# ---------------------------------------------------------------- shared
def forward_rate(runs):
    """Share of reviews predicted actionable (sent to the development team), mean over runs."""
    return float(np.mean([np.mean(np.asarray(p) != OTHER_ID) for p in runs.values()]))
