"""RQ1 experiment helpers shared by Tier 1 and Tier 2 (Train for fitting, Validation for tuning only;
Hold-out is never touched here).

A *runner* wraps one strategy behind two calls so grid search, multi-seed runs and cost logging are
strategy-agnostic:
    runner.train(df, n_classes, label_names, seed, overrides) -> model
    runner.predict_proba(model, texts) -> (probs [n × k], seconds)
"""
import json
import time

import numpy as np
import pandas as pd


def _peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    except ImportError:
        pass
    return None


def _free():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class VisobertRunner:
    def __init__(self, base_config: dict, model_name=None):
        self.base, self.model_name, self._tok = dict(base_config), model_name, None

    def config(self, overrides=None):
        return {**self.base, **(overrides or {})}

    def train(self, df, n_classes, names, seed, overrides=None):
        from src.classification.strategy_visobert.train import train_visobert
        model, tok, _ = train_visobert(df, n_classes, names, self.config(overrides), seed, model_name=self.model_name)
        self._tok, self._max_len = tok, self.config(overrides)['max_len']
        return model

    def predict_proba(self, model, texts):
        from src.classification.strategy_visobert.predict import predict_visobert
        _, probs, seconds = predict_visobert(model, self._tok, texts, self._max_len)
        return np.asarray(probs), seconds


class BilstmRunner:
    def __init__(self, base_config: dict, vocab: dict, emb_matrix):
        self.base, self.vocab, self.emb = dict(base_config), vocab, emb_matrix

    def config(self, overrides=None):
        return {**self.base, **(overrides or {})}

    def train(self, df, n_classes, names, seed, overrides=None):
        from src.classification.strategy_bilstm_fasttext.train import train_bilstm
        cfg = self.config(overrides)
        self._max_len = int(cfg['max_len'])
        return train_bilstm(df, n_classes, names, cfg, seed, self.vocab, self.emb)[0]

    def predict_proba(self, model, texts):
        from src.classification.strategy_bilstm_fasttext.train import predict_bilstm
        _, probs, seconds = predict_bilstm(model, texts, self.vocab, self._max_len)
        return np.asarray(probs), seconds


def run_one(runner, train_df, val_df, n_classes, names, seed, overrides=None):
    """Train once, predict Validation. Returns probs + cost (train s, inference s, s/1000 reviews, peak GPU GB)."""
    _peak_reset()
    t0 = time.time()
    model = runner.train(train_df, n_classes, names, seed, overrides)
    train_s = time.time() - t0
    probs, infer_s = runner.predict_proba(model, val_df['text'].astype(str).tolist())
    out = {'seed': seed, 'overrides': overrides or {}, 'probs': probs, 'train_seconds': train_s,
           'infer_seconds': infer_s, 'infer_seconds_per_1000': infer_s / len(val_df) * 1000,
           'peak_gpu_gb': _peak_gb(), 'n_train': len(train_df)}
    del model
    _free()
    return out


def grid_search(runner, grid, train_df, val_df, n_classes, names, metric_fn, select='macro_f1', seed=42,
                log=print, strategy=''):
    """One seed per configuration (fast tuning). metric_fn(y_true, probs) -> dict.
    Returns (results DataFrame sorted by `select`, best overrides)."""
    rows = []
    y = val_df['y'].tolist()
    for i, ov in enumerate(grid, 1):
        r = run_one(runner, train_df, val_df, n_classes, names, seed, ov)
        m = metric_fn(y, r['probs'])
        rows.append({'strategy': strategy, 'config': json.dumps(ov, sort_keys=True), **m,
                     'train_seconds': r['train_seconds'], 'infer_seconds_per_1000': r['infer_seconds_per_1000'],
                     'peak_gpu_gb': r['peak_gpu_gb']})
        log(f'[{strategy}] {i}/{len(grid)} {ov}: {select}={m[select]:.4f} ({r["train_seconds"]:.0f}s)')
    res = pd.DataFrame(rows).sort_values(select, ascending=False, kind='mergesort').reset_index(drop=True)
    return res, json.loads(res.loc[0, 'config'])


def run_seeds(runner, overrides, seeds, train_df, val_df, n_classes, names, log=print, strategy=''):
    """Winning configuration re-run with every seed. Returns list of run_one results."""
    out = []
    for s in seeds:
        r = run_one(runner, train_df, val_df, n_classes, names, s, overrides)
        log(f'[{strategy}] seed={s}: train {r["train_seconds"]:.0f}s, inference {r["infer_seconds_per_1000"]:.2f}s/1000')
        out.append(r)
    return out


def predictions_frame(val_df, runs, label_names, strategy, tier):
    """Long table of Validation probabilities per seed (kept for cross-tier combination later)."""
    frames = []
    for r in runs:
        d = pd.DataFrame(r['probs'], columns=[f'p_{n}' for n in label_names])
        d.insert(0, 'review_id', val_df['review_id'].values)
        d.insert(1, 'gold_y', val_df['y'].values)
        d.insert(2, 'seed', r['seed'])
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, 'tier', tier)
    out.insert(0, 'strategy', strategy)
    return out


def save_dev_config(path, content: dict, header: str = ''):
    """Write a tuned (dev, not frozen) configuration as YAML for 03_configs/dev/."""
    import os
    import yaml
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(''.join(f'# {line}\n' for line in header.splitlines()))
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    return path


def cost_row(runs, strategy):
    """mean ± std of the cost indicators over seeds."""
    df = pd.DataFrame([{k: r[k] for k in ('train_seconds', 'infer_seconds_per_1000', 'peak_gpu_gb', 'n_train')} for r in runs])
    row = {'strategy': strategy, 'n_train': int(df['n_train'].iloc[0])}
    for c in ('train_seconds', 'infer_seconds_per_1000', 'peak_gpu_gb'):
        if df[c].notna().any():
            row[f'{c}_mean'], row[f'{c}_std'] = float(df[c].mean()), float(df[c].std(ddof=1) if len(df) > 1 else 0.0)
    return row
