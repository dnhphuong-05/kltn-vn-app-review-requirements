"""Paper, step 4 (amendment v4.1): confidence-based routing and temperature-scaled Tier 1 thresholds.

Only saved predictions are used: no retraining, no API call. Both analyses are selected on Validation alone; the
selection is saved (with its SHA-256) before any label of a locked set is read.

Routing: a small flat 4-class model labels every review; reviews whose confidence c(x) = max class probability is
below θ take the label of the flat LLM instead. θ_max maximises Validation Macro-F1 (ties: fewer LLM calls); θ_98 is
the smallest θ whose Validation Macro-F1 reaches 98% of the flat LLM's (θ_max if none does).

Calibration: per seed, a temperature T is fitted on Validation by minimising the log-loss of σ(logit(p)/T); the
guarded Tier 1 rule (v2.0 + v3.5) is applied to the calibrated seed-42 probabilities, Tier 2 is unchanged.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.classification.evaluate import FOUR_CLASSES, OTHER_ID, metrics
from src.classification.final_eval import neural_probs
from src.classification.tier1_actionable.threshold import choose_guarded, sweep

THETAS = np.round(np.arange(0.30, 0.99 + 1e-9, 0.01), 2)
RETAIN = 0.98
SMALL = {'PhoBERT': 'phobert', 'TF-IDF': 'tfidf', 'ViSoBERT': 'visobert'}     # routing: small flat models (main first)
TIER1 = {'TF-IDF': 'tfidf', 'Bi-LSTM + fastText': 'bilstm', 'ViSoBERT': 'visobert', 'PhoBERT': 'phobert'}
EPS = 1e-6
LBL = {l: i for i, l in enumerate(FOUR_CLASSES)}


# ---------------------------------------------------------------- saved probabilities
def _source(res_dir, set_name, model):
    """Prediction file holding `model` for a set: GĐ10 (ViSoBERT, Bi-LSTM), TF-IDF or step-2 neural (PhoBERT)."""
    if model in ('visobert', 'bilstm'):
        return os.path.join(res_dir, f'gd10_pred_{set_name}.csv')
    if model == 'tfidf':
        return os.path.join(res_dir, 'paper_tfidf_val_predictions.csv' if set_name == 'validation'
                            else f'paper_pred_tfidf_{set_name}.csv')
    return os.path.join(res_dir, f'paper_pred_neural_{set_name}.csv')


def probabilities(res_dir, set_name, ids, tier, model, seeds):
    """{seed: probability matrix aligned to ids} of a saved model ('tier1': [P(NOT), P(ACT)], 'flat': 4 classes)."""
    pred = pd.read_csv(_source(res_dir, set_name, model), encoding='utf-8-sig')
    pred = pred[[c for c in pred.columns if c != 'gold_label']]          # the TF-IDF Validation file carries labels
    return neural_probs(pred, ids, tier, model, seeds)


def flat_llm_validation(calls_jsonl, ids):
    """Flat LLM class ids on Validation from the call log (last attempt per review; OTHER when nothing parsed)."""
    last = {}
    for line in open(calls_jsonl, encoding='utf-8'):
        r = json.loads(line)
        if r.get('tier') == 'flat' and (r['review_id'] not in last or r['attempt'] >= last[r['review_id']]['attempt']):
            last[r['review_id']] = r
    missing = [i for i in ids if i not in last]
    if missing:
        raise RuntimeError(f'flat LLM Validation: thiếu {len(missing)} review')
    return np.array([LBL.get(last[i].get('parsed'), OTHER_ID) for i in ids])


# ---------------------------------------------------------------- routing
def route(small_probs, llm_pred, theta):
    """4-class prediction of the routed configuration and the boolean mask of reviews sent to the LLM."""
    p = np.asarray(small_probs)
    to_llm = p.max(axis=1) < theta
    return np.where(to_llm, np.asarray(llm_pred), p.argmax(axis=1)), to_llm


def routing_sweep(y, small_probs, llm_pred, thetas=THETAS):
    """Macro-F1 and LLM call rate of the routed configuration for every θ."""
    rows = []
    for t in thetas:
        pred, to_llm = route(small_probs, llm_pred, t)
        rows.append({'theta': float(t), 'macro_f1': metrics(np.asarray(y), pred, FOUR_CLASSES)['macro_f1'],
                     'llm_rate': float(to_llm.mean())})
    return pd.DataFrame(rows)


def choose_thetas(sw, llm_macro_f1, retain=RETAIN):
    """θ_max: highest Macro-F1, ties broken by fewer LLM calls (lower θ); θ_98: smallest θ reaching retain × LLM."""
    best = sw.sort_values(['macro_f1', 'llm_rate', 'theta'], ascending=[False, True, True], kind='mergesort').iloc[0]
    ok = sw[sw['macro_f1'] >= retain * llm_macro_f1].sort_values('theta', kind='mergesort')
    t98 = ok.iloc[0] if len(ok) else best
    return {'theta_max': float(best['theta']), 'theta_max_val_macro_f1': float(best['macro_f1']),
            'theta_max_val_llm_rate': float(best['llm_rate']), 'theta_98': float(t98['theta']),
            'theta_98_val_macro_f1': float(t98['macro_f1']), 'theta_98_val_llm_rate': float(t98['llm_rate']),
            'theta_98_reached': bool(len(ok)), 'llm_val_macro_f1': float(llm_macro_f1), 'retain': retain}


def routed_configs(small: dict, llm_pred, selection):
    """{name: {seed: prediction}} and {name: {seed: LLM call mask}} for every small model and both operating points.
    small: {strategy: {seed: flat probabilities}}."""
    cfg, rate = {}, {}
    for x, runs in small.items():
        for point in ('theta_max', 'theta_98'):
            t = selection[x][point]
            name = f'{x} ⇢ LLM ({"θ_max" if point == "theta_max" else "θ_98"})'
            cfg[name], rate[name] = {}, {}
            for s, p in runs.items():
                cfg[name][s], rate[name][s] = route(p, llm_pred, t)
    return cfg, rate


# ---------------------------------------------------------------- temperature scaling
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def calibrate(p, temperature):
    return 1.0 / (1.0 + np.exp(-_logit(p) / temperature))


def fit_temperature(y, p, lo=0.05, hi=20.0):
    """T minimising the binary log-loss of σ(logit(p)/T) on (y, p)."""
    y, z = np.asarray(y, dtype=float), _logit(p)

    def nll(t):
        q = np.clip(1.0 / (1.0 + np.exp(-z / t)), EPS, 1 - EPS)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    return float(minimize_scalar(nll, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4}).x)


def ece(y, p, bins=15):
    """Expected calibration error of the positive-class probability with equal-width bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def calibrated_threshold(y1, probs_by_seed, canonical_seed=42):
    """Per-seed temperatures and the guarded threshold chosen on the calibrated canonical run (Validation).
    Also returns the raw-probability threshold equivalent to τ_cal on the canonical seed."""
    temps = {s: fit_temperature(y1, p[:, 1]) for s, p in probs_by_seed.items()}
    q = calibrate(probs_by_seed[canonical_seed][:, 1], temps[canonical_seed])
    chosen = choose_guarded(sweep(y1, q))
    t = chosen['threshold']
    raw = float(1.0 / (1.0 + np.exp(-temps[canonical_seed] * _logit(t))))   # p with σ(logit(p)/T) = t
    return {'temperatures': temps, 'tau_cal': t, 'tau_raw_equivalent_seed42': raw, 'val_choice': chosen}


# ---------------------------------------------------------------- bookkeeping
def save_selection(obj, path):
    """Write the Validation-only selection and return its SHA-256 (recorded before any locked label is read)."""
    txt = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True)
    open(path, 'w', encoding='utf-8').write(txt)
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def routed_cost_rows(names_rates, small_cost, llm_cost, macro_f1, grid_runs, hardware, labels_initial):
    """Cost components of routed configurations in the format of cost_model.components().
    names_rates: {name: (small strategy, LLM call rate on Hold-out seed 42)}; small_cost: {strategy: (sec/review,
    train sec/run)}; llm_cost: (sec/review, USD/review) of the flat LLM; macro_f1: {name: % on Hold-out seed 42}."""
    rows = []
    for name, (x, r) in names_rates.items():
        sec, train = small_cost[x]
        rows.append({'cấu hình': name, 'mã': name.replace('PhoBERT', 'P').replace('TF-IDF', 'T').replace('ViSoBERT', 'V')
                     .replace(' ⇢ LLM ', '⇢L ').replace('(', '').replace(')', ''),
                     'kiến trúc': 'định tuyến', 'Macro-F1': macro_f1[name], 'gọi API': True, 'có huấn luyện': True,
                     'phần cứng': hardware[x], 'nhãn ban đầu': labels_initial['trained'], 'lần chạy dò lưới': grid_runs[x],
                     'giây train / lần chạy': train, 'giây suy luận / review': sec + r * llm_cost[0],
                     'USD API / review': r * llm_cost[1]})
    return pd.DataFrame(rows)
