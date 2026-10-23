"""Strategy B — prediction."""
import time


def predict_visobert(model, tokenizer, texts, max_len: int, batch_size: int = 64, device=None):
    """Returns (pred_ids, probs [n × k], seconds)."""
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer([str(t) for t in texts[i:i + batch_size]], truncation=True, max_length=max_len,
                            padding=True, return_tensors='pt').to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(**enc).logits.float()
            p = torch.softmax(logits, dim=-1)
            probs += p.cpu().tolist()
            preds += p.argmax(-1).cpu().tolist()
    return preds, probs, time.time() - t0
