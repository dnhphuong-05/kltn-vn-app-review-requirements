"""Strategy B — fine-tuned ViSoBERT (ĐC §7.3.2). Hyperparameters come from
03_configs/frozen/classification.yaml (section strategy_b_visobert)."""
import time

from src.classification.common import balanced_class_weights, seed_everything
from src.classification.strategy_visobert.tokenizer import VisobertTokenizer, self_check


def _loader(texts, labels, tokenizer, max_len, batch_size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader
    data = list(zip(texts, labels))

    def collate(batch):
        enc = tokenizer([t for t, _ in batch], truncation=True, max_length=max_len, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor([y for _, y in batch], dtype=torch.long)
        return enc
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def train_visobert(train_df, n_classes: int, label_names, config: dict, seed: int,
                   text_col: str = 'text', label_col: str = 'y', device=None, model_name=None):
    """Fine-tune a sequence classifier. Returns (model, tokenizer, history); pass the returned tokenizer to
    predict_visobert so train and predict share one tokenization.
    history = {'epoch_loss': [...], 'train_seconds': float, 'n_train': int, 'class_weights': [...],
               'mean_tokens': float}."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    seed_everything(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = model_name or config['model_name']
    tokenizer = VisobertTokenizer(name)     # sentencepiece trực tiếp, không dùng AutoTokenizer (tách sai)
    self_check(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=n_classes, id2label=dict(enumerate(label_names)),
        label2id={l: i for i, l in enumerate(label_names)},
        ignore_mismatched_sizes=True).to(device)   # always a fresh head sized to this tier

    texts, y = train_df[text_col].astype(str).tolist(), train_df[label_col].astype(int).tolist()
    mean_tokens = float(np.mean([len(tokenizer.encode(t)) for t in texts]))
    print(f'[visobert] tokenizer sentencepiece: {mean_tokens:.1f} token/review trên {len(texts)} review train')
    loader = _loader(texts, y, tokenizer, config['max_len'], config['batch_size'], True, seed)
    weights = balanced_class_weights(y, n_classes) if config.get('class_weight') == 'balanced' else [1.0] * n_classes
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config.get('weight_decay', 0.0)))
    steps = len(loader) * int(config['epochs'])
    sched = get_linear_schedule_with_warmup(opt, int(steps * float(config.get('warmup_ratio', 0.0))), steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {'epoch_loss': [], 'n_train': len(y), 'class_weights': [float(w) for w in weights],
               'mean_tokens': mean_tokens}
    if use_amp:
        torch.cuda.synchronize()
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
    if use_amp:
        torch.cuda.synchronize()
    history['train_seconds'] = time.time() - t0
    return model, tokenizer, history
