"""GĐ11 — cascade inference on the whole Shopee corpus with the downstream strategy.

Amendment v2.2: the downstream classifier is the ViSoBERT cascade (Tier 1 threshold + Tier 2 argmax) with the frozen
v2.1 configs, seed 42 (Q6). The corpus is the scrubbed interim file; the models see `text_clean`, the same field the
gold CSVs carry as `text`. Reviews of the labelled sets stay in the corpus (clustering runs on every actionable review)
but are tagged with their split so any downstream number can be traced back to them.
"""
import json

import numpy as np
import pandas as pd

CORPUS_FILE = 'shopee_clean_T0_2026-10-06.jsonl'
SPLITS_MANIFEST = 'splits_manifest_T0_2026-10-06.json'
SPLIT_ORDER = ('train', 'validation', 'holdout', 'focused')
TIER2_NAMES = ('BUG', 'FEATURE', 'UX')
LABEL_COLS = ('gold_label', 'label', 'labels')            # must never appear in the corpus file
PILOT_PREVALENCE = {'actionable': 0.205, 'ci95': (0.168, 0.247), 'n': 400}   # amendment v1.7 (Wilson)


def load_corpus(path):
    """review_id, text (= text_clean), review_date, app_version, rating, flags. Raises on labels or duplicate ids."""
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    bad = sorted({k for r in rows for k in r if k in LABEL_COLS})
    if bad:
        raise RuntimeError(f'File corpus có cột nhãn {bad} — không được dùng ở GĐ11')
    df = pd.DataFrame(rows)
    df = df.rename(columns={'text': 'text_raw'}).rename(columns={'text_clean': 'text'})
    df['flags'] = df['flags'].apply(lambda x: ';'.join(x) if isinstance(x, list) else '')
    if not df['review_id'].is_unique:
        raise RuntimeError('review_id trùng trong corpus')
    return df[['review_id', 'text', 'review_date', 'app_version', 'rating', 'flags']]


def split_tags(manifest_path, review_ids):
    """'train' | 'validation' | 'holdout' | 'focused' | 'none' per review (sets are disjoint, v1.7)."""
    sets = json.load(open(manifest_path, encoding='utf-8'))['sets']
    tag = {}
    for name in SPLIT_ORDER:
        for rid in sets[name]['review_ids']:
            if rid in tag:
                raise RuntimeError(f'{rid} thuộc cả {tag[rid]} và {name}')
            tag[rid] = name
    return pd.Series([tag.get(r, 'none') for r in review_ids], index=range(len(review_ids)), name='split')


def cascade_table(review_ids, split, p_tier1, p_tier2, threshold):
    """One row per review: Tier 1 probability/label, Tier 2 probabilities/label (only where Tier 1 kept the review)
    and the 4-class label. p_tier1: [n, 2] (NOT_ACTIONABLE, ACTIONABLE); p_tier2: [n, 3] (BUG, FEATURE, UX)."""
    p1, p2 = np.asarray(p_tier1), np.asarray(p_tier2)
    act = p1[:, 1] >= threshold
    t2 = np.array(TIER2_NAMES)[p2.argmax(1)]
    out = pd.DataFrame({'review_id': list(review_ids), 'split': list(split), 'p_ACTIONABLE': p1[:, 1],
                        'tier1_label': np.where(act, 'ACTIONABLE', 'NOT_ACTIONABLE')})
    for i, n in enumerate(TIER2_NAMES):
        out[f'p_{n}'] = p2[:, i]
    out['tier2_label'] = np.where(act, t2, '')
    out['label4'] = np.where(act, t2, 'OTHER')
    return out


def class_summary(table):
    """Counts and shares of the 4-class cascade label, overall and per split."""
    order = list(TIER2_NAMES) + ['OTHER']
    rows = []
    for name, g in [('toàn corpus', table)] + [(s, table[table['split'] == s]) for s in SPLIT_ORDER + ('none',)]:
        c = g['label4'].value_counts().reindex(order, fill_value=0)
        rows.append({'phạm vi': name, 'n': len(g), **{k: int(v) for k, v in c.items()},
                     'ACTIONABLE': int(c[list(TIER2_NAMES)].sum()),
                     'tỷ lệ ACTIONABLE': float(c[list(TIER2_NAMES)].sum() / max(len(g), 1))})
    return pd.DataFrame(rows)


def expected_positive_rate(prevalence, recall, precision):
    """Share of reviews a classifier is expected to call positive: prevalence × recall / precision."""
    return prevalence * recall / precision


def agreement(new_probs, ref_probs, threshold=None):
    """Label agreement (threshold on column 1 for Tier 1, argmax otherwise) and max |Δp| between two runs."""
    a, b = np.asarray(new_probs), np.asarray(ref_probs)
    if threshold is None:
        la, lb = a.argmax(1), b.argmax(1)
    else:
        la, lb = a[:, 1] >= threshold, b[:, 1] >= threshold
    return {'n': len(a), 'nhãn khớp': float((la == lb).mean()), 'chênh xác suất lớn nhất': float(np.abs(a - b).max())}


def actionable_export(table, corpus):
    """GĐ12 input: actionable reviews with text and the metadata needed for temporal validation (GĐ13)."""
    keep = table.loc[table['tier1_label'] == 'ACTIONABLE', ['review_id', 'tier2_label', 'split']]
    out = keep.merge(corpus[['review_id', 'text', 'review_date', 'app_version']], on='review_id', how='left',
                     validate='one_to_one')
    if out['text'].isna().any():
        raise RuntimeError('Thiếu text cho một số review actionable')
    return out[['review_id', 'text', 'tier2_label', 'review_date', 'app_version', 'split']]
