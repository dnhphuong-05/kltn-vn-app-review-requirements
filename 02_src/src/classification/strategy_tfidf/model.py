"""Strategy D (paper, amendment v3.2): TF-IDF + linear classifier, the traditional machine-learning baseline.

CPU only. One grid per tier (Tier 1, Tier 2, flat 4-class): feature set × classifier × C, class_weight 'balanced'
as for the neural strategies. LinearSVC is wrapped in 5-fold sigmoid calibration so that every configuration
outputs probabilities (the Tier 1 threshold rule of amendment v2.0 needs P(ACTIONABLE)). Grid order matters:
ties keep the first configuration in GRID (stable sort), as for the other strategies.
"""
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

FEATURES = ('word', 'char', 'word+char')
CLASSIFIERS = ('logreg', 'linearsvc')
C_VALUES = (0.1, 1.0, 10.0)
GRID = [{'features': f, 'classifier': c, 'C': C} for f in FEATURES for c in CLASSIFIERS for C in C_VALUES]
FEATURE_SPEC = {'word': "TfidfVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'(?u)\\b\\w+\\b', sublinear_tf=True)",
                'char': "TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)"}


def prep(s):
    """NFC + lower case (the vectorisers get this as preprocessor)."""
    return unicodedata.normalize('NFC', str(s)).lower()


def _word():
    return TfidfVectorizer(preprocessor=prep, analyzer='word', ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
                           sublinear_tf=True)


def _char():
    return TfidfVectorizer(preprocessor=prep, analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def make_model(cfg, seed=42):
    parts = {'word': [('word', _word())], 'char': [('char', _char())],
             'word+char': [('word', _word()), ('char', _char())]}[cfg['features']]
    if cfg['classifier'] == 'logreg':
        clf = LogisticRegression(C=cfg['C'], class_weight='balanced', max_iter=5000, random_state=seed)
    elif cfg['classifier'] == 'linearsvc':
        clf = CalibratedClassifierCV(LinearSVC(C=cfg['C'], class_weight='balanced', max_iter=20000, random_state=seed),
                                     method='sigmoid', cv=5)
    else:
        raise ValueError(cfg['classifier'])
    return Pipeline([('tfidf', FeatureUnion(parts)), ('clf', clf)])


def fit(cfg, texts, y, seed=42):
    """Fitted pipeline and training seconds."""
    m = make_model(cfg, seed)
    t0 = time.perf_counter()
    m.fit(list(texts), np.asarray(y))
    return m, time.perf_counter() - t0


def predict_proba(model, texts):
    """Class probabilities and inference seconds."""
    t0 = time.perf_counter()
    p = model.predict_proba(list(texts))
    return p, time.perf_counter() - t0


def grid_search(train_texts, y_train, val_texts, y_val, metric_fn, grid=GRID, seed=42, select='macro_f1'):
    """Every configuration trained once (seed 42) and scored on Validation at argmax. Returns (results sorted by
    `select`, stable so ties keep grid order; best configuration)."""
    rows = []
    for i, cfg in enumerate(grid):
        m, secs = fit(cfg, train_texts, y_train, seed)
        p, _ = predict_proba(m, val_texts)
        rows.append({'thứ tự lưới': i, **cfg, **metric_fn(np.asarray(y_val), p.argmax(1)), 'giây train': round(secs, 3)})
    res = pd.DataFrame(rows).sort_values(select, ascending=False, kind='mergesort').reset_index(drop=True)
    best = {k: (float(res.loc[0, k]) if k == 'C' else str(res.loc[0, k])) for k in ('features', 'classifier', 'C')}
    return res, best


def run_seeds(cfg, train_texts, y_train, sets: dict, seeds, timing_set=None):
    """Train the chosen configuration once per seed and predict every set. sets: {name: texts}.
    Returns {seed: {'probs': {name: array}, 'train_seconds': s, 'infer_seconds_per_1000': s}} where inference
    time is measured on `timing_set` (default: the first set)."""
    timing_set = timing_set or next(iter(sets))
    out = {}
    for s in seeds:
        m, tsec = fit(cfg, train_texts, y_train, s)
        probs, isec = {}, None
        for name, texts in sets.items():
            p, secs = predict_proba(m, texts)
            probs[name] = p
            if name == timing_set:
                isec = secs / len(p) * 1000
        out[s] = {'probs': probs, 'train_seconds': tsec, 'infer_seconds_per_1000': isec}
    return out
