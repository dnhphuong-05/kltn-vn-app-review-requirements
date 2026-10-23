"""Strategy B — ViSoBERT tokenization straight from its SentencePiece model (used by BOTH train and predict).

uitnlp/visobert ships only `sentencepiece.bpe.model` (a BPE model, no tokenizer.json). transformers 5.x
(checked 5.16.1 on Kaggle, 5.20.2 locally; use_fast True and False alike) rebuilds it as an XLM-R *Unigram*
tokenizer from the BPE ranks and over-segments: "thủ tục" -> "▁thủ ▁t ục", only 8.2% of Train reviews match
the original pieces (32.9 vs 20.7 tokens/review). So we encode with sentencepiece directly.

Piece ids -> model ids follow the fairseq/XLM-R convention: <s>=0 <pad>=1 </s>=2 <unk>=3, spm piece id i (≥3)
-> i + 1, <mask> = piece_size + 1. Verified against the ViSoBERT MLM head (GĐ6, 60 Train reviews, each token
masked in turn): top-1 recovery 42.2% with offset +1, 1.1% with raw spm ids, 8.4% with the transformers
tokenizer. `embedding_alignment` re-runs that check.
"""
import os

SPM_FILE = 'sentencepiece.bpe.model'
OFFSET = 1                                    # fairseq: spm ids shifted by one to make room for <pad>
SPECIALS = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}
CHECK_SAMPLES = (
    'chạy chậm',
    'thủ tục hủy hàng rắc rối',
    'shopee',
    'app lag vl, đặt hàg mãi k đc 😡😡',
    'sp ok nhưng giao hơi lâu 👍 ko bik bao h mới có voucher',
    'QUÁ NHIỀU QUẢNG CÁO, POP UP LIÊN TỤC',
)
WHOLE_WORDS = ('chạy', 'chậm', 'thủ', 'tục', 'hủy', 'hàng', 'rắc', 'rối', 'shopee')


class VisobertTokenizer:
    """Callable like a transformers tokenizer: tok(texts, truncation, max_length, padding, return_tensors='pt')
    -> BatchEncoding(input_ids, attention_mask)."""

    def __init__(self, model_name: str):
        import sentencepiece as spm
        from transformers import AutoConfig
        if os.path.isdir(model_name):
            path = os.path.join(model_name, SPM_FILE)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_name, SPM_FILE)
        self.sp = spm.SentencePieceProcessor(model_file=path)
        cfg = AutoConfig.from_pretrained(model_name)
        # special ids from the model config; must agree with the fairseq layout the embedding was trained with
        self.bos_id, self.pad_id, self.eos_id = cfg.bos_token_id, cfg.pad_token_id, cfg.eos_token_id
        self.unk_id = SPECIALS['<unk>']
        self.mask_id = self.sp.get_piece_size() + OFFSET
        if (self.bos_id, self.pad_id, self.eos_id) != (SPECIALS['<s>'], SPECIALS['<pad>'], SPECIALS['</s>']):
            raise ValueError(f'Special ids trong config ({self.bos_id}, {self.pad_id}, {self.eos_id}) khác bố cục '
                             f'fairseq <s>=0 <pad>=1 </s>=2 — không dùng được ánh xạ offset +1.')
        if (self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()) != (0, 1, 2):
            raise ValueError('sentencepiece model không có <unk>=0 <s>=1 </s>=2 như XLM-R.')
        if cfg.vocab_size <= self.mask_id:
            raise ValueError(f'vocab_size của model ({cfg.vocab_size}) không chứa đủ id spm + offset ({self.mask_id}).')
        self._inv = {v: k for k, v in SPECIALS.items()}
        self._inv[self.mask_id] = '<mask>'

    def pieces(self, text):
        return self.sp.encode(str(text), out_type=str)

    def encode(self, text):
        """Piece ids mapped to model ids, without <s>/</s>."""
        unk = self.sp.unk_id()
        return [self.unk_id if i == unk else i + OFFSET for i in self.sp.encode(str(text))]

    def convert_ids_to_tokens(self, ids):
        return [self._inv[i] if i in self._inv else self.sp.id_to_piece(i - OFFSET) for i in ids]

    def __call__(self, texts, truncation=True, max_length=256, padding=True, return_tensors='pt'):
        import torch
        from transformers import BatchEncoding
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            ids = self.encode(t)
            if truncation:
                ids = ids[:max_length - 2]               # room for <s> and </s>
            rows.append([self.bos_id] + ids + [self.eos_id])
        width = max(len(r) for r in rows) if padding else None
        if width is not None:
            mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
            rows = [r + [self.pad_id] * (width - len(r)) for r in rows]
        else:
            mask = [[1] * len(r) for r in rows]
        if return_tensors == 'pt':
            return BatchEncoding({'input_ids': torch.tensor(rows, dtype=torch.long),
                                  'attention_mask': torch.tensor(mask, dtype=torch.long)})
        return BatchEncoding({'input_ids': rows, 'attention_mask': mask})


def self_check(tok: VisobertTokenizer, samples=CHECK_SAMPLES, whole_words=WHOLE_WORDS):
    """Defensive check before any training: our input_ids must decode back to exactly spm.encode's pieces,
    wrapped in <s> … </s>, and common words must stay whole (the transformers tokenizer splits them)."""
    for s in samples:
        ref = tok.sp.encode(s, out_type=str)
        ids = tok([s], max_length=10_000)['input_ids'][0].tolist()
        got = tok.convert_ids_to_tokens(ids[1:-1])
        if ids[0] != tok.bos_id or ids[-1] != tok.eos_id or got != ref:
            raise RuntimeError(f'Tokenization ViSoBERT lệch spm gốc với câu {s!r}:\n  spm : {ref}\n  ours: {got}')
    split = {w: tok.pieces(w) for w in whole_words if tok.pieces(w) != ['▁' + w]}
    if split:
        raise RuntimeError(f'Tokenizer tách vụn từ thông dụng (dấu hiệu dùng nhầm tokenizer transformers): {split}')
    return True


def embedding_alignment(tok: VisobertTokenizer, model_name: str, texts, max_tokens: int = 40, device=None):
    """Mask each token in turn and let the pretrained MLM head recover it. High top-1 (~40%) means our ids
    point at the embedding rows the model was pretrained with; a wrong offset collapses it to ~1%."""
    import torch
    from transformers import AutoModelForMaskedLM
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    hit = tot = 0
    with torch.no_grad():
        for t in texts:
            ids = [tok.bos_id] + tok.encode(t)[:max_tokens] + [tok.eos_id]
            n = len(ids) - 2
            if n < 1:
                continue
            x = torch.tensor([ids] * n, device=device)
            pos = torch.arange(1, n + 1, device=device)
            x[torch.arange(n), pos] = tok.mask_id
            pred = mlm(input_ids=x).logits[torch.arange(n), pos].argmax(-1)
            hit += int((pred == torch.tensor(ids[1:-1], device=device)).sum())
            tot += n
    del mlm
    return {'mlm_top1': hit / max(tot, 1), 'n_tokens': tot}
