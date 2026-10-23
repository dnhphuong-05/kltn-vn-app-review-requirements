"""Strategy A — Bi-LSTM over pretrained Vietnamese fastText (ĐC §7.3.1): frozen 300-d embeddings,
1-layer Bi-LSTM, concatenated mean/max pooling, dropout, linear head; weighted CE from Train.
Hyperparameters: 03_configs/frozen/classification.yaml (strategy_a_bilstm)."""
import time

import numpy as np

from src.classification.common import balanced_class_weights, seed_everything
from src.classification.strategy_bilstm_fasttext.embeddings import UNK, tokenize


def _encode(texts, vocab, max_len):
    ids = [[vocab.get(t, vocab[UNK]) for t in tokenize(s)][:max_len] or [vocab[UNK]] for s in texts]
    lens = [len(x) for x in ids]
    arr = np.zeros((len(ids), max(lens)), dtype='int64')
    for i, x in enumerate(ids):
        arr[i, :len(x)] = x
    return arr, np.asarray(lens)


def _model(emb_matrix, n_classes, cfg):
    import torch
    from torch import nn

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=bool(cfg['freeze_embedding']),
                                                    padding_idx=0)
            self.lstm = nn.LSTM(emb_matrix.shape[1], int(cfg['hidden_size']), num_layers=int(cfg['num_layers']),
                                batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(float(cfg['dropout']))
            self.out = nn.Linear(4 * int(cfg['hidden_size']), n_classes)

        def forward(self, ids, lens):
            h, _ = self.lstm(self.emb(ids))                       # [B, T, 2H]
            mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lens[:, None]).unsqueeze(-1)
            mean = (h * mask).sum(1) / lens[:, None].clamp(min=1)
            mx = h.masked_fill(~mask, -1e4).max(1).values
            return self.out(self.drop(torch.cat([mean, mx], dim=-1)))
    return BiLSTMClassifier()


def train_bilstm(train_df, n_classes: int, label_names, config: dict, seed: int, vocab: dict, emb_matrix,
                 text_col: str = 'text', label_col: str = 'y', device=None):
    """Returns (model, history)."""
    import torch
    seed_everything(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, L = _encode(train_df[text_col].astype(str).tolist(), vocab, int(config['max_len']))
    y = train_df[label_col].astype(int).to_numpy()
    model = _model(emb_matrix, n_classes, config).to(device)
    w = balanced_class_weights(y, n_classes) if config.get('class_weight') == 'balanced' else np.ones(n_classes)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(config['lr']))
    g = torch.Generator().manual_seed(seed)
    bs, history = int(config['batch_size']), {'epoch_loss': [], 'n_train': len(y), 'class_weights': [float(v) for v in w]}
    t0 = time.time()
    model.train()
    for _ in range(int(config['epochs'])):
        perm = torch.randperm(len(y), generator=g).numpy()
        total = 0.0
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            ids = torch.tensor(X[b], device=device)
            lens = torch.tensor(L[b], device=device)
            loss = loss_fn(model(ids, lens), torch.tensor(y[b], device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        history['epoch_loss'].append(total / len(y))
    history['train_seconds'] = time.time() - t0
    return model, history


def predict_bilstm(model, texts, vocab: dict, max_len: int, batch_size: int = 256, device=None):
    """Returns (pred_ids, probs, seconds)."""
    import torch
    device = device or next(model.parameters()).device
    X, L = _encode([str(t) for t in texts], vocab, max_len)
    model.eval()
    preds, probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            p = torch.softmax(model(torch.tensor(X[i:i + batch_size], device=device),
                                    torch.tensor(L[i:i + batch_size], device=device)), dim=-1)
            probs += p.cpu().tolist()
            preds += p.argmax(-1).cpu().tolist()
    return preds, probs, time.time() - t0
