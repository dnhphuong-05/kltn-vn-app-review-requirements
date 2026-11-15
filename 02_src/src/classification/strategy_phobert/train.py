"""Strategy E (paper, amendment v3.4) — fine-tuned PhoBERT (vinai/phobert-base-v2) on word-segmented text.

The training loop is the one of ViSoBERT (weighted cross-entropy, AdamW, linear warmup, fp16 autocast on GPU,
gradient clipping 1.0, fresh classification head). Only the tokenizer differs: PhoBERT's own BPE tokenizer over
segmented words, checked before training by the <unk> rate and by masked-token recovery with the pretrained MLM
head (a wrong tokenizer collapses recovery, as seen with ViSoBERT in GĐ6).
"""
import time

import numpy as np

from src.classification.common import balanced_class_weights, seed_everything

MODEL_NAME = 'vinai/phobert-base-v2'
_TOKENIZERS = {}


def load_tokenizer(name=MODEL_NAME):
    from transformers import AutoTokenizer
    if name not in _TOKENIZERS:
        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


def unk_rate(tok, texts):
    """(share of <unk> tokens, mean tokens per text) on segmented texts, without special tokens."""
    ids = [tok(str(t), add_special_tokens=False)['input_ids'] for t in texts]
    n = sum(len(i) for i in ids)
    unk = sum(x == tok.unk_token_id for i in ids for x in i)
    return unk / max(n, 1), n / max(len(ids), 1)


def mlm_alignment(tok, name, texts, max_tokens=40, device=None):
    """Mask each token in turn and let the pretrained MLM head recover it (same check as ViSoBERT, GĐ6)."""
    import torch
    from transformers import AutoModelForMaskedLM
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mlm = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
    hit = tot = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(str(t), truncation=True, max_length=max_tokens + 2)['input_ids']
            n = len(ids) - 2
            if n < 1:
                continue
            x = torch.tensor([ids] * n, device=device)
            pos = torch.arange(1, n + 1, device=device)
            x[torch.arange(n), pos] = tok.mask_token_id
            pred = mlm(input_ids=x, attention_mask=torch.ones_like(x)).logits[torch.arange(n), pos].argmax(-1)
            hit += int((pred == torch.tensor(ids[1:-1], device=device)).sum())
            tot += n
    del mlm
    return {'mlm_top1': hit / max(tot, 1), 'n_tokens': tot}


def _loader(texts, labels, tok, max_len, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader
    data = list(zip(texts, labels))

    def collate(batch):
        enc = tok([t for t, _ in batch], truncation=True, max_length=max_len, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor([y for _, y in batch], dtype=torch.long)
        return enc
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def train_phobert(texts, y, n_classes, label_names, config, seed, device=None):
    """texts: already segmented. Returns (model, history)."""
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    seed_everything(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = config['model_name']
    tok = load_tokenizer(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=n_classes, id2label=dict(enumerate(label_names)),
        label2id={l: i for i, l in enumerate(label_names)}, ignore_mismatched_sizes=True).to(device)
    y = [int(v) for v in y]
    loader = _loader(list(texts), y, tok, int(config['max_len']), int(config['batch_size']), True, seed)
    weights = balanced_class_weights(y, n_classes) if config.get('class_weight') == 'balanced' else [1.0] * n_classes
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config.get('weight_decay', 0.0)))
    steps = len(loader) * int(config['epochs'])
    sched = get_linear_schedule_with_warmup(opt, int(steps * float(config.get('warmup_ratio', 0.0))), steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = {'epoch_loss': [], 'n_train': len(y), 'class_weights': [float(w) for w in weights]}
    t0 = time.time()
    model.train()
    for _ in range(int(config['epochs'])):
        total, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(labels), n + len(labels)
        history['epoch_loss'].append(total / max(n, 1))
    history['train_seconds'] = time.time() - t0
    return model, history


def predict_phobert(model, tok, texts, max_len, batch_size=64):
    """texts: already segmented. Returns (probs [n × k], seconds)."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    probs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok([str(t) for t in texts[i:i + batch_size]], truncation=True, max_length=max_len,
                      padding=True, return_tensors='pt').to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(**enc).logits.float()
            probs += torch.softmax(logits, dim=-1).cpu().tolist()
    return np.asarray(probs), time.time() - t0


class PhobertRunner:
    """Runner for `src.classification.rq1` (grid_search / run_one / run_multi). Texts arrive raw and are segmented
    here (cached), so the same Train/Validation frames serve every strategy. predict_proba times the model only;
    segmentation time is measured separately in the notebook."""

    def __init__(self, base_config: dict, segmenter):
        self.base, self.seg = dict(base_config), segmenter

    def config(self, overrides=None):
        return {**self.base, **(overrides or {})}

    def train(self, df, n_classes, names, seed, overrides=None):
        cfg = self.config(overrides)
        self._max_len, self._name = int(cfg['max_len']), cfg['model_name']
        return train_phobert(self.seg(df['text'].astype(str).tolist()), df['y'].tolist(), n_classes, names, cfg, seed)[0]

    def predict_proba(self, model, texts):
        return predict_phobert(model, load_tokenizer(self._name), self.seg(list(texts)), self._max_len)
