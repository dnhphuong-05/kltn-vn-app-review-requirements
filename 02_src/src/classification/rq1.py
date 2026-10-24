"""RQ1 experiment helpers shared by Tier 1 and Tier 2 (Train for fitting, Validation for tuning only;
Hold-out is never touched here).

A *runner* wraps one strategy behind two calls so grid search, multi-seed runs and cost logging are
strategy-agnostic:
    runner.train(df, n_classes, label_names, seed, overrides) -> model
    runner.predict_proba(model, texts) -> (probs [n × k], seconds)
and, to evaluate the locked sets later (GĐ10) with exactly the models reported on Validation:
    runner.save(model, path) / runner.load(path) -> model
"""
import json
import os
import shutil
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

    def save(self, model, path):
        """HF checkpoint (fp32, unchanged weights) + the sentencepiece model, so VisobertTokenizer(path) works."""
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        shutil.copy(self._tok.spm_path, os.path.join(path, os.path.basename(self._tok.spm_path)))
        with open(os.path.join(path, 'runner.json'), 'w', encoding='utf-8') as f:
            json.dump({'max_len': int(self._max_len)}, f)

    def load(self, path, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification
        from src.classification.strategy_visobert.tokenizer import VisobertTokenizer, self_check
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tok = VisobertTokenizer(path)
        self_check(self._tok)
        self._max_len = json.load(open(os.path.join(path, 'runner.json'), encoding='utf-8'))['max_len']
        return AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()


class BilstmRunner:
    def __init__(self, base_config: dict, vocab: dict, emb_matrix):
        self.base, self.vocab, self.emb = dict(base_config), vocab, emb_matrix

    def config(self, overrides=None):
        return {**self.base, **(overrides or {})}

    def train(self, df, n_classes, names, seed, overrides=None):
        from src.classification.strategy_bilstm_fasttext.train import train_bilstm
        cfg = self.config(overrides)
        self._max_len, self._cfg, self._n = int(cfg['max_len']), cfg, n_classes
        return train_bilstm(df, n_classes, names, cfg, seed, self.vocab, self.emb)[0]

    def predict_proba(self, model, texts):
        from src.classification.strategy_bilstm_fasttext.train import predict_bilstm
        _, probs, seconds = predict_bilstm(model, texts, self.vocab, self._max_len)
        return np.asarray(probs), seconds

    def save(self, model, path):
        """state_dict (includes the frozen fastText matrix) + config + vocab in one file."""
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'config': self._cfg, 'n_classes': self._n, 'vocab': self.vocab},
                   os.path.join(path, 'bilstm.pt'))

    def load(self, path, device=None):
        import torch
        from src.classification.strategy_bilstm_fasttext.train import _model
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(os.path.join(path, 'bilstm.pt'), map_location='cpu', weights_only=False)
        self.vocab, self._cfg, self._n = ck['vocab'], ck['config'], ck['n_classes']
        self._max_len = int(self._cfg['max_len'])
        emb = ck['state_dict']['emb.weight'].numpy()
        model = _model(emb, self._n, self._cfg)
        model.load_state_dict(ck['state_dict'])
        return model.to(device).eval()


RELOAD_PROB_TOL = 1e-2


def verify_reload(new_probs, saved_probs, name='', prob_tol=RELOAD_PROB_TOL):
    """Check a reloaded checkpoint against the probabilities saved when it was trained. On another GPU / library
    version probabilities drift by float noise (seen: 3.35e-3 between GĐ7 and GĐ9); what drives every metric is the
    predicted label. Passes iff argmax labels agree 100% AND max |Δp| < prob_tol (a broken checkpoint drifts far more)."""
    new, old = np.asarray(new_probs), np.asarray(saved_probs)
    if new.shape != old.shape:
        raise AssertionError(f'{name}: kích thước xác suất khác nhau {new.shape} vs {old.shape}')
    diff = float(np.abs(new - old).max())
    same = new.argmax(1) == old.argmax(1)
    msg = (f'{name}: nhãn khớp {same.mean():.1%} ({int(same.sum())}/{len(same)}) | '
           f'chênh xác suất lớn nhất {diff:.2e} (ngưỡng {prob_tol:.0e})')
    if not same.all() or diff >= prob_tol:
        raise AssertionError('Checkpoint nạp lại không khớp — ' + msg)
    return {'label_agreement': float(same.mean()), 'max_prob_diff': diff, 'n': int(len(same)), 'message': msg}


def find_checkpoints(strategy, seeds, root='/kaggle/input', contains=''):
    """Paths <...>/checkpoints/<strategy>/seed<s> in the given seed order, e.g. a previous notebook's output
    attached as input. `contains` narrows the search (e.g. 'rq1_tier2') when several outputs are attached."""
    found = {}
    for dirpath, _, _ in os.walk(root):
        if (os.path.basename(dirpath) == strategy and os.path.basename(os.path.dirname(dirpath)) == 'checkpoints'
                and contains in dirpath):
            for s in seeds:
                p = os.path.join(dirpath, f'seed{s}')
                if os.path.isdir(p):
                    found.setdefault(s, p)
    missing = [s for s in seeds if s not in found]
    if missing:
        raise FileNotFoundError(f'Không tìm thấy checkpoints/{strategy}/seed{missing} (lọc "{contains}") trong {root} — '
                                'cần gắn output của notebook đã lưu checkpoint làm input.')
    return [found[s] for s in seeds]


def run_one(runner, train_df, val_df, n_classes, names, seed, overrides=None, save_dir=None):
    """Train once, predict Validation. Returns probs + cost (train s, inference s, s/1000 reviews, peak GPU GB).
    With `save_dir`, the trained model is kept at <save_dir>/seed<seed> (runner.save)."""
    _peak_reset()
    t0 = time.time()
    model = runner.train(train_df, n_classes, names, seed, overrides)
    train_s = time.time() - t0
    probs, infer_s = runner.predict_proba(model, val_df['text'].astype(str).tolist())
    out = {'seed': seed, 'overrides': overrides or {}, 'probs': probs, 'train_seconds': train_s,
           'infer_seconds': infer_s, 'infer_seconds_per_1000': infer_s / len(val_df) * 1000,
           'peak_gpu_gb': _peak_gb(), 'n_train': len(train_df)}
    if save_dir:
        out['checkpoint'] = os.path.join(save_dir, f'seed{seed}')
        runner.save(model, out['checkpoint'])
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


def run_seeds(runner, overrides, seeds, train_df, val_df, n_classes, names, log=print, strategy='', save_dir=None):
    """Winning configuration re-run with every seed. Returns list of run_one results."""
    out = []
    for s in seeds:
        r = run_one(runner, train_df, val_df, n_classes, names, s, overrides, save_dir=save_dir)
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
