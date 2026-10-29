"""RQ2 (GĐ12) — BERTopic on all actionable reviews, dominant type and type purity per topic.

Design (research_protocol, RQ2; CLAUDE.md §1, §7):
  - one clustering run on EVERY actionable review; BUG/FEATURE/UX (the Tier 2 label) is attached afterwards;
  - sentence embedding of the raw text → UMAP → HDBSCAN → c-TF-IDF on word-segmented text (underthesea);
  - min_topic_size chosen on the Q12 grid by coherence C_v, configurations that are degenerate (one giant topic,
    too many topics) are excluded;
  - outlier reduction (embedding similarity) and merging of near-duplicate topics on the chosen configuration;
  - dominant type: a type with ≥ 60% of the topic's reviews, otherwise `mixed` (+ the mixed-origin types of Q14).
Names never change membership: keyword names here, LLM names in `naming.py`.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

TYPES = ('BUG', 'FEATURE', 'UX')
_LETTER = re.compile(r'[^\W\d_]', re.UNICODE)


# ---------------------------------------------------------------- config
def load_config(repo_dir, stage='dev'):
    path = os.path.join(repo_dir, '03_configs', stage, 'rq2_topics_dev_config.yaml' if stage == 'dev'
                        else 'rq2_topics_config.yaml')
    return yaml.safe_load(open(path, encoding='utf-8'))['rq2_topics']


# ---------------------------------------------------------------- text for c-TF-IDF / coherence
def segment(texts):
    """underthesea word segmentation, lower-cased; compounds joined by '_' (e.g. 'giao_hàng')."""
    from underthesea import word_tokenize
    return [word_tokenize(str(t).lower(), format='text') for t in texts]


def informative_mask(tokens, min_content_tokens):
    """Reviews with at least min_content_tokens tokens left after stopword filtering. Only these shape the topics
    (BERTopic fit); shorter reviews are attached afterwards by embedding similarity (amendment v2.3)."""
    return np.array([len(t) >= min_content_tokens for t in tokens])


def tokens_for_topics(segmented, stopwords, min_chars=2):
    """Token lists used by BOTH the c-TF-IDF vectorizer and the coherence model: drop stopwords, tokens without a
    letter (numbers, punctuation, emoji) and tokens shorter than min_chars."""
    stop = set(stopwords)
    out = []
    for s in segmented:
        out.append([w for w in s.split() if w not in stop and _LETTER.search(w) and len(w.replace('_', '')) >= min_chars])
    return out


# ---------------------------------------------------------------- embeddings (cached)
def _ids_hash(review_ids):
    return hashlib.sha256('\n'.join(map(str, review_ids)).encode('utf-8')).hexdigest()[:16]


def embed(texts, review_ids, ecfg, cache_dir, tag):
    """Normalised sentence embeddings [n, d], cached as <tag>.npy + <tag>.json (model, revision, ids hash)."""
    npy, meta = os.path.join(cache_dir, f'{tag}.npy'), os.path.join(cache_dir, f'{tag}.json')
    want = {'model_name': ecfg['model_name'], 'revision': ecfg['revision'], 'n': len(texts), 'ids': _ids_hash(review_ids)}
    if os.path.exists(npy) and os.path.exists(meta) and json.load(open(meta, encoding='utf-8')) == want:
        return np.load(npy)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ecfg['model_name'], revision=ecfg['revision'])
    emb = model.encode(list(texts), batch_size=int(ecfg['batch_size']), normalize_embeddings=bool(ecfg['normalize']),
                       show_progress_bar=False, convert_to_numpy=True)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, emb)
    json.dump(want, open(meta, 'w', encoding='utf-8'), indent=1)
    return emb


# ---------------------------------------------------------------- BERTopic
def _vectorizer(cfg):
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None,
                           ngram_range=tuple(cfg['vectorizer']['ngram_range']))


def _ctfidf(cfg):
    from bertopic.vectorizers import ClassTfidfTransformer
    return ClassTfidfTransformer(reduce_frequent_words=bool(cfg['vectorizer']['reduce_frequent_words']))


def fit_bertopic(docs, embeddings, min_topic_size, cfg):
    """docs: token lists joined by spaces. Returns (model, topics)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    u, h = cfg['umap'], cfg['hdbscan']
    model = BERTopic(
        umap_model=UMAP(n_neighbors=u['n_neighbors'], n_components=u['n_components'], min_dist=u['min_dist'],
                        metric=u['metric'], random_state=u['random_state']),
        hdbscan_model=HDBSCAN(min_cluster_size=int(min_topic_size), metric=h['metric'],
                              cluster_selection_method=h['cluster_selection_method'], prediction_data=h['prediction_data']),
        vectorizer_model=_vectorizer(cfg), ctfidf_model=_ctfidf(cfg), top_n_words=int(cfg['selection']['top_n_words']),
        min_topic_size=int(min_topic_size), calculate_probabilities=False, verbose=False,
        language='multilingual')        # 'english' makes BERTopic strip every non-ASCII character (Vietnamese diacritics, '_')
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    return model, np.asarray(topics)


def topic_words(model, top_n):
    """{topic_id: [word, ...]} for every non-outlier topic (empty placeholder words removed)."""
    return {t: [w for w, _ in model.get_topic(t)[:top_n] if w] for t in sorted(model.get_topics()) if t != -1}


def coherence_cv(words_by_topic, tokens, dictionary=None):
    """Mean C_v and C_v per topic (gensim, single process). A topic with fewer than 2 keywords (its reviews have no
    token left after stopword filtering) has no C_v: it is NaN per topic and left out of the mean."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    dictionary = dictionary or Dictionary(tokens)
    words = {t: [w for w in ws if w in dictionary.token2id] for t, ws in words_by_topic.items()}
    ids = [t for t, ws in words.items() if len(ws) >= 2]
    per = {t: float('nan') for t in words}
    if ids:
        cm = CoherenceModel(topics=[words[t] for t in ids], texts=tokens, dictionary=dictionary, coherence='c_v',
                            processes=1)
        per.update(zip(ids, map(float, cm.get_coherence_per_topic())))
    return float(np.nanmean([per[t] for t in ids])) if ids else float('nan'), per


def topic_stats(topics):
    t = np.asarray(topics)
    clustered = t[t != -1]
    sizes = pd.Series(clustered).value_counts()
    return {'số topic': int(sizes.size), 'tỷ lệ outlier': float((t == -1).mean()),
            'topic lớn nhất (% review đã gom)': float(sizes.max() / len(clustered)) if len(clustered) else float('nan'),
            'kích thước trung vị': float(sizes.median()) if len(sizes) else float('nan'),
            'kích thước nhỏ nhất': int(sizes.min()) if len(sizes) else 0}


def grid(docs, tokens, embeddings, cfg, log=print):
    """Q12 grid: one fit per min_topic_size, C_v + secondary signals. Returns (table, {mts: (model, topics)})."""
    from gensim.corpora import Dictionary
    dct = Dictionary(tokens)
    rows, fits = [], {}
    for mts in cfg['min_topic_size_grid']:
        model, topics = fit_bertopic(docs, embeddings, mts, cfg)
        cv, per = coherence_cv(topic_words(model, cfg['selection']['top_n_words']), tokens, dct)
        st = topic_stats(topics)
        degenerate = (st['topic lớn nhất (% review đã gom)'] > cfg['selection']['max_topic_share']
                      or st['số topic'] > cfg['selection']['max_topics'] or st['số topic'] < 2)
        rows.append({'min_topic_size': mts, 'C_v': cv, **st, 'topic không có C_v': int(np.isnan(list(per.values())).sum()),
                     'suy biến': degenerate})
        fits[mts] = (model, topics)
        log(f"min_topic_size {mts}: C_v {cv:.4f} | {st['số topic']} topic | outlier {st['tỷ lệ outlier']:.1%}")
    return pd.DataFrame(rows), fits


def choose_min_topic_size(table):
    """Highest C_v among non-degenerate configurations (ties → larger topics, fewer for the BA to review)."""
    ok = table[~table['suy biến']]
    if ok.empty:
        raise RuntimeError('Mọi cấu hình đều suy biến — cần xem lại lưới trước khi chọn')
    return int(ok.sort_values(['C_v', 'min_topic_size'], ascending=[False, False]).iloc[0]['min_topic_size'])


# ---------------------------------------------------------------- outliers, merging
def centroids(embeddings, topics):
    t = np.asarray(topics)
    ids = sorted(x for x in set(t) if x != -1)
    c = np.stack([embeddings[t == i].mean(0) for i in ids])
    return ids, c / np.linalg.norm(c, axis=1, keepdims=True)


def member_similarity(embeddings, topics):
    """Cosine of every clustered review to its own topic centroid."""
    ids, c = centroids(embeddings, topics)
    t = np.asarray(topics)
    row = {tid: i for i, tid in enumerate(ids)}
    m = t != -1
    return np.einsum('ij,ij->i', embeddings[m], c[[row[x] for x in t[m]]])


def outlier_threshold(embeddings, topics, cfg):
    """outliers.threshold, or with 'auto' the outliers.member_percentile-th percentile of member-to-centroid cosine:
    an outlier joins a topic only if it is at least as close to the centroid as that share of the topic's own members."""
    o = cfg['outliers']
    if o['threshold'] != 'auto':
        return float(o['threshold'])
    return float(np.percentile(member_similarity(embeddings, topics), o['member_percentile']))


def reduce_outliers(model, docs, topics, embeddings, cfg, threshold):
    """Reassign an outlier to its nearest topic only if cosine(doc, topic) ≥ threshold; refresh c-TF-IDF."""
    new = model.reduce_outliers(docs, list(topics), strategy=cfg['outliers']['strategy'], embeddings=embeddings,
                                threshold=threshold)
    model.update_topics(docs, topics=new, vectorizer_model=_vectorizer(cfg), ctfidf_model=_ctfidf(cfg),
                        top_n_words=int(cfg['selection']['top_n_words']))
    return np.asarray(model.topics_)


def outlier_similarity(embeddings, topics):
    """Cosine of each outlier to its nearest topic centroid (to show what the reassignment threshold keeps out)."""
    ids, c = centroids(embeddings, topics)
    t = np.asarray(topics)
    out = embeddings[t == -1]
    return (out @ c.T).max(1) if len(out) else np.array([])


def merge_duplicates(model, docs, topics, embeddings, cfg):
    """Merge pairs of topics whose centroids have cosine ≥ merge.topic_similarity. Pairs only, most similar first, each
    topic in at most one pair — no chaining, so near-duplicates cannot snowball into one giant topic.
    Returns (topics, [(a, b, cosine), ...])."""
    ids, c = centroids(embeddings, topics)
    sim = c @ c.T
    thr = cfg['merge']['topic_similarity']
    cand = sorted(((sim[a, b], ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))
                   if sim[a, b] >= thr), reverse=True)
    used, pairs = set(), []
    for s, a, b in cand:
        if a not in used and b not in used:
            used |= {a, b}
            pairs.append((int(a), int(b), float(s)))
    if pairs:
        model.merge_topics(docs, [[a, b] for a, b, _ in pairs])
    return np.asarray(model.topics_), pairs


def attach_by_similarity(embeddings, fit_topics, fit_mask, threshold):
    """Topics for EVERY review: fitted reviews keep their topic; each other review joins the nearest topic centroid
    (centroids from fitted reviews only) if cosine ≥ threshold, otherwise −1. Returns (topics, attached mask)."""
    ids, c = centroids(embeddings[fit_mask], fit_topics)
    out = np.full(len(embeddings), -1)
    out[fit_mask] = fit_topics
    rest = np.where(~fit_mask)[0]
    attached = np.zeros(len(embeddings), dtype=bool)
    if len(rest):
        s = embeddings[rest] @ c.T
        best = s.argmax(1)
        ok = s[np.arange(len(rest)), best] >= threshold
        out[rest[ok]] = np.asarray(ids)[best[ok]]
        attached[rest[ok]] = True
    return out, attached


def random_members(topics, k, seed=42):
    """{topic: k random member indices} (non-outlier topics), to show the BA / the LLM more than the most typical reviews."""
    rng = np.random.default_rng(seed)
    t = np.asarray(topics)
    return {int(i): rng.choice(np.where(t == i)[0], size=min(k, int((t == i).sum())), replace=False).tolist()
            for i in sorted(set(t)) if i != -1}


# ---------------------------------------------------------------- dominant type
def dominant_types(topics, tier2_labels, cfg):
    """Per topic: counts per type, dominant type (≥ threshold) or `mixed`, type purity and the Q14 mixed-origin types."""
    d = cfg['dominant_type']
    df = pd.DataFrame({'topic': np.asarray(topics), 'type': np.asarray(tier2_labels)})
    rows = []
    for t, g in df.groupby('topic'):
        c = g['type'].value_counts().reindex(TYPES, fill_value=0)
        share = c / len(g)
        top = share.idxmax()
        dominant = top if share[top] >= d['threshold'] else 'mixed'
        origin = [k for k in TYPES if c[k] >= d['mixed_origin_min_reviews'] or share[k] >= d['mixed_origin_min_share']]
        rows.append({'topic': int(t), 'size': len(g), **{f'n_{k}': int(c[k]) for k in TYPES},
                     'dominant_type': dominant, 'type_purity': float(share[top]),
                     'cờ mixed': dominant == 'mixed', 'mixed_origin_types': ';'.join(origin) if dominant == 'mixed' else ''})
    return pd.DataFrame(rows)


def type_purity_overall(dom):
    """Size-weighted mean purity over non-outlier topics."""
    d = dom[dom['topic'] != -1]
    return float((d['type_purity'] * d['size']).sum() / d['size'].sum())


# ---------------------------------------------------------------- tables
def representative_index(embeddings, topics, k):
    """{topic: indices of the k reviews closest to the topic centroid} (non-outlier topics)."""
    ids, c = centroids(embeddings, topics)
    t = np.asarray(topics)
    out = {}
    for i, tid in enumerate(ids):
        idx = np.where(t == tid)[0]
        out[tid] = idx[np.argsort(-(embeddings[idx] @ c[i]))[:k]].tolist()
    return out


def topics_table(model, topics, dom, cv_per_topic, n_actionable, cfg):
    """topics.csv: one row per topic, no review text."""
    words = topic_words(model, cfg['selection']['top_n_words'])
    k = cfg['naming']['n_keywords']
    rows = []
    for _, r in dom.iterrows():
        t = int(r['topic'])
        w = words.get(t, [])
        name = 'outlier' if t == -1 else (' / '.join(w[:k]) or '(không còn từ khóa sau khi lọc từ dừng)')
        rows.append({'topic_id': t, 'keyword_name': name,
                     'size': int(r['size']), 'frequency': r['size'] / n_actionable,
                     'dominant_type': '' if t == -1 else r['dominant_type'], 'type_purity': r['type_purity'],
                     'flag_mixed': bool(r['cờ mixed']) and t != -1, 'mixed_origin_types': r['mixed_origin_types'],
                     **{f'n_{x}': int(r[f'n_{x}']) for x in TYPES},
                     'c_v': cv_per_topic.get(t, np.nan), 'top_keywords': ', '.join(w)})
    return pd.DataFrame(rows).sort_values(['topic_id']).reset_index(drop=True)
