"""Pretrained Vietnamese fastText (Common Crawl + Wikipedia, cc.vi.300) — static embeddings (CLAUDE.md §3).

Tokenisation is by syllable (lower-cased word characters): the cc.vi vocabulary is whitespace-separated
syllables, so underscore-joined word segmentation ("thanh_toán") would be mostly out of vocabulary.
Only vectors for the corpus vocabulary are kept, read by streaming the .vec.gz (no full 2M-word load).
"""
import gzip
import os
import re
import urllib.request

import numpy as np

_TOKEN = re.compile(r'\w+', re.UNICODE)
PAD, UNK = '<pad>', '<unk>'


def tokenize(text: str):
    return _TOKEN.findall(str(text).lower())


def build_vocab(texts, min_count: int = 1):
    from collections import Counter
    c = Counter(t for s in texts for t in tokenize(s))
    words = [w for w, n in c.most_common() if n >= min_count]
    return {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(words)}}


def download(url: str, dest: str, log=print) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    log(f'Tải {url} …')
    urllib.request.urlretrieve(url, dest + '.part')
    os.replace(dest + '.part', dest)
    return dest


def load_vectors(vec_gz_path: str, vocab: dict, dim: int = 300, seed: int = 42):
    """Embedding matrix [len(vocab) × dim]: fastText vector when available, small random otherwise
    (PAD = 0). Returns (matrix, coverage) where coverage = share of vocab found in fastText."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 0.1, (len(vocab), dim)).astype('float32')
    mat[vocab[PAD]] = 0.0
    found = 0
    with gzip.open(vec_gz_path, 'rt', encoding='utf-8', errors='ignore') as f:
        next(f)                                  # header: n_words dim
        for line in f:
            w, rest = line.split(' ', 1)
            idx = vocab.get(w)
            if idx is not None:
                mat[idx] = np.asarray(rest.split(), dtype='float32')[:dim]
                found += 1
    return mat, found / max(len(vocab) - 2, 1)
