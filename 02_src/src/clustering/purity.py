"""RQ2 — manual topic purity (research_protocol, mục RQ2; amendment v2.4).

After the clustering result is frozen, ~10 random members of every topic (outliers excluded, seed 42) are judged by two
people independently: does the review talk about the issue the topic stands for (its name and keywords)? Purity of a
topic = share of 'yes' judgements, per rater and averaged over the two raters; agreement = % agreement and Cohen's κ on
the yes/no judgements. Raters see the topic name, its keywords and the review text only — not the Tier 2 label nor the
dominant type, so the judgement is about the topic, not the classifier.

This is an evaluation of a frozen result: low-purity topics are reported and flagged for the BA, the clustering is not
re-tuned on these judgements.
"""
import os

import numpy as np
import pandas as pd

SHEET_COLS = ['item_no', 'topic_id', 'topic_name', 'keywords', 'review_id', 'text', 'relevant', 'note']
YES, NO = {'1', 'có', 'co', 'yes', 'y', 'x'}, {'0', 'không', 'khong', 'no', 'n'}
FLAG_BELOW = 0.60                 # amendment v2.4: topic purity below this → flagged for the BA


def sample_items(review_topic_map, actionable, topics, k=10, seed=42, n_keywords=6):
    """k random members per topic (−1 excluded), grouped by topic in id order, shuffled inside a topic."""
    rng = np.random.default_rng(seed)
    text = actionable.set_index('review_id')['text']
    t = topics[topics['topic_id'] != -1].set_index('topic_id')
    rows = []
    for tid in sorted(t.index):
        ids = review_topic_map.loc[review_topic_map['topic_id'] == tid, 'review_id'].to_numpy()
        for rid in rng.choice(ids, size=min(k, len(ids)), replace=False):
            name = t.loc[tid, 'name_vi'] if isinstance(t.loc[tid].get('name_vi'), str) else t.loc[tid, 'keyword_name']
            rows.append({'topic_id': int(tid), 'topic_name': name,
                         'keywords': ', '.join(str(t.loc[tid, 'top_keywords']).split(', ')[:n_keywords]),
                         'review_id': rid, 'text': text[rid], 'relevant': '', 'note': ''})
    out = pd.DataFrame(rows)
    out.insert(0, 'item_no', range(1, len(out) + 1))
    return out[SHEET_COLS]


def write_blank(items, out_dir, name='topic_purity_blank.csv'):
    """Never overwrites an existing sheet (raters may already be working on a copy)."""
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        return path, False
    os.makedirs(out_dir, exist_ok=True)
    items.to_csv(path, index=False, encoding='utf-8-sig')
    return path, True


def read_judgements(path, blank):
    """`relevant` as 1/0 aligned to the blank sheet; raises on a missing or unreadable answer, or an edited row."""
    d = pd.read_csv(path, encoding='utf-8-sig', dtype=str).fillna('')
    if list(d['review_id']) != list(blank['review_id']) or list(d['topic_id'].astype(int)) != list(blank['topic_id']):
        raise RuntimeError(f'{os.path.basename(path)}: review_id / topic_id khác phiếu gốc (đã sắp xếp lại hoặc sửa dòng?)')
    v = d['relevant'].str.strip().str.lower()
    bad = d.loc[~v.isin(YES | NO), 'item_no'].tolist()
    if bad:
        raise RuntimeError(f'{os.path.basename(path)}: {len(bad)} dòng chưa chấm hoặc không đọc được (item_no {bad[:10]}…)')
    return v.isin(YES).astype(int).to_numpy()


def score(blank, a, b, topics):
    """Per-topic purity (rater A, rater B, mean), agreement, κ; plus overall macro / size-weighted purity."""
    from sklearn.metrics import cohen_kappa_score
    d = blank[['topic_id']].copy()
    d['A'], d['B'] = a, b
    size = topics.set_index('topic_id')['size']
    rows = []
    for tid, g in d.groupby('topic_id'):
        pa, pb = g['A'].mean(), g['B'].mean()
        rows.append({'topic_id': int(tid), 'n': len(g), 'purity_A': pa, 'purity_B': pb, 'purity': (pa + pb) / 2,
                     'agreement': float((g['A'] == g['B']).mean()), 'size': int(size[tid]),
                     'flag_ba': (pa + pb) / 2 < FLAG_BELOW})
    per = pd.DataFrame(rows)
    kappa = float(cohen_kappa_score(d['A'], d['B'])) if d['A'].nunique() + d['B'].nunique() > 2 else float('nan')
    overall = {'n_review': len(d), 'n_topic': len(per), 'purity_macro': float(per['purity'].mean()),
               'purity_weighted': float((per['purity'] * per['size']).sum() / per['size'].sum()),
               'agreement': float((d['A'] == d['B']).mean()), 'cohen_kappa': kappa,
               'n_topic_flagged': int(per['flag_ba'].sum())}
    return per, overall
