"""GĐ16 — D6 structured output: five tool-independent relational tables (CSV, UTF-8) in 06_output_data_model/.

Keys: reviews.topic_id → topics.topic_id ; requirements.topic_id → topics.topic_id ;
      requirement_evidence.requirement_id → requirements.requirement_id ; requirement_evidence.review_id → reviews.review_id.
All tables are built from results already locked in 05_experiments/ — nothing is recomputed or re-predicted here.
"""
import json
import re

import numpy as np
import pandas as pd

SOURCE_APP = 'shopee'                     # GĐ11 corpus = Shopee (Lazada is the external test set of RQ1 only)
TIMEZONE = 'Asia/Ho_Chi_Minh'
_PM = re.compile(r'^\s*(?P<m>-?\d+(?:\.\d+)?)\s*(?:±\s*(?P<s>\d+(?:\.\d+)?))?\s*$')


def mean_sd(x):
    """'65.9 ± 0.9' → (65.9, 0.9); '81.7' → (81.7, nan)."""
    m = _PM.match(str(x))
    return (float(m['m']), float(m['s']) if m['s'] else np.nan) if m else (np.nan, np.nan)


def reviews_table(corpus, preds, rmap, pool, units, temporal_groups):
    """One row per review of the GĐ11 corpus: metadata, text, tier-1/tier-2 predictions, topic, evidence status and the
    temporal status of the issue group the review belongs to (its unit if any, else its topic)."""
    d = corpus[['review_id', 'review_date', 'app_version', 'rating', 'text']].merge(
        preds[['review_id', 'split', 'p_ACTIONABLE', 'tier1_label', 'tier2_label']], on='review_id', how='left', validate='one_to_one')
    d = d.merge(rmap[['review_id', 'topic_id']], on='review_id', how='left', validate='one_to_one')
    d = d.merge(pool[['review_id', 'unit_id', 'exclusion_reason', 'used_for_generation']], on='review_id', how='left',
                validate='one_to_one')
    d['review_date'] = pd.to_datetime(d['review_date'], utc=True).dt.tz_convert(TIMEZONE).map(lambda x: x.isoformat())
    d.loc[d['tier1_label'] != 'ACTIONABLE', 'tier2_label'] = ''
    d['evidence_status'] = np.select([d['used_for_generation'].eq(True), d['exclusion_reason'].notna(), d['unit_id'].notna()],
                                     ['evidence_pool', d['exclusion_reason'].astype(str), 'temporal_insufficient'], default='')
    unit_status = units.set_index('unit_id')['temporal_status']
    topic_status = temporal_groups[temporal_groups['type'] == 'ALL'].set_index('topic_id')['status']
    d['temporal_status'] = d['unit_id'].map(unit_status).fillna(d['topic_id'].map(topic_status)).fillna('')
    d['topic_id'] = d['topic_id'].astype('Int64')
    d.insert(1, 'source_app', SOURCE_APP)
    return d[['review_id', 'source_app', 'review_date', 'app_version', 'rating', 'text', 'split', 'p_ACTIONABLE', 'tier1_label',
              'tier2_label', 'topic_id', 'unit_id', 'evidence_status', 'temporal_status']]


def topics_table(topics, purity, temporal_groups, cfg):
    """One row per topic (outlier -1 included): names, dominant type, purities, temporal status and how GĐ14 treated it."""
    sel = cfg['selection']
    step = {t: k for k in ('ba_manual', 'scope_filter', 'type_filter', 'q14_split', 'normal') for t in sel[k]}
    st = temporal_groups[temporal_groups['type'] == 'ALL'].set_index('topic_id')['status']
    p = purity.set_index('topic_id')['purity']
    t = topics.copy()
    return pd.DataFrame({
        'topic_id': t['topic_id'], 'label': t['name_vi'].fillna('outlier'), 'label_en': t['name_en'].fillna('outlier'),
        'dominant_type': t['dominant_type'].fillna(''), 'type_purity': t['type_purity'], 'manual_purity': t['topic_id'].map(p),
        'size': t['size'], 'frequency': t['frequency'], 'top_keywords': t['top_keywords'].fillna(''),
        'scope': t['scope'].fillna(''), 'temporal_status': t['topic_id'].map(st).fillna(''),
        'generation_step': t['topic_id'].map(step).fillna('outlier'),
        'flag_ba': t['topic_id'].isin(sel['ba_manual'])})


def requirements_table(reqs, qus, human=None):
    """Candidate Requirements with their automatic QUS result and, when available, the human evaluation scores."""
    r = reqs.rename(columns={'cluster_id': 'topic_id'}).merge(qus[['requirement_id', 'n_defects']].rename(
        columns={'n_defects': 'qus_defects'}), on='requirement_id', how='left')
    r['status'] = 'candidate'                 # BA duyệt, sửa hoặc loại (amendment v2.8)
    cols = ['requirement_id', 'topic_id', 'unit_id', 'type', 'title_vi', 'problem_vi', 'user_story', 'acceptance_criteria',
            'temporal_status', 'issue_frequency', 'mixed_origin_flag', 'scope_note', 'n_evidence_pool', 'n_evidence_cited',
            'qus_defects', 'status']
    if human is not None:
        h = human[['requirement_id', 'fidelity', 'actionability', 'ac_testability', 'hallucination', 'type_correct',
                   'evidence_precision']].add_prefix('eval_').rename(columns={'eval_requirement_id': 'requirement_id'})
        r = r.merge(h, on='requirement_id', how='left')
        cols += list(h.columns[1:])
    return r[cols].sort_values('issue_frequency', ascending=False, kind='stable', ignore_index=True)


def evidence_table(ev, reviews, human_ev=None):
    e = ev.merge(reviews[['review_id', 'source_app', 'review_date']], on='review_id', how='left', validate='many_to_one')
    if human_ev is not None:
        e = e.merge(human_ev[['requirement_id', 'review_id', 'supports']].rename(columns={'supports': 'eval_supports'}),
                    on=['requirement_id', 'review_id'], how='left')
    return e


def model_comparison(rq1):
    """RQ1 results of GĐ10 (Hold-out, opened once) in long, numeric form: one row per strategy × tier.
    rq1: dict of the GĐ10 tables (tier1, tier2, e2e, e2e_lazada, cost)."""
    cost = rq1['cost'].set_index('chiến lược')
    name = {'Cascade ViSoBERT': 'ViSoBERT', 'Cascade Bi-LSTM + fastText': 'Bi-LSTM + fastText', 'Cascade LLM few-shot': 'LLM few-shot',
            'Baseline phẳng ViSoBERT': 'Baseline phẳng ViSoBERT'}
    laz = rq1['e2e_lazada'].assign(s=lambda x: x['cấu hình'].map(name)).set_index('s')['Macro-F1 (4 lớp)']
    rows = []

    def add(strategy, tier, main, main_name, per_class, extra, n):
        c = cost.loc[strategy]
        m, sd = mean_sd(main)
        rows.append({'strategy': strategy, 'tier': tier, 'metric': main_name, 'macro_f1': m, 'macro_f1_sd': sd,
                     'per_class_f1': json.dumps({k: mean_sd(v)[0] for k, v in per_class.items()}, ensure_ascii=False),
                     **extra, 'time_per_1k': float(c['suy luận / 1.000 review (giây)']),
                     'api_cost': float(c['USD API / 1.000 review']), 'train_time': float(c['huấn luyện / seed (giây, tầng 1 + tầng 2)']),
                     'hardware': f"GPU tối đa {float(c['GPU tối đa (GB)']):.2f} GB" if float(c['GPU tối đa (GB)']) else 'không cần GPU (API)',
                     'external_test_f1': mean_sd(laz.get(strategy, np.nan))[0] if tier == 'end_to_end' else np.nan, 'n_runs': int(n)})

    for _, r in rq1['tier1'].iterrows():
        add(r['chiến lược'], 'tier1', r['Macro-F1 (2 lớp)'], 'Macro-F1 (2 lớp)', {'ACTIONABLE': r['F1 ACT']},
            {'actionable_recall': mean_sd(r['Recall ACT'])[0], 'actionable_f1': mean_sd(r['F1 ACT'])[0]}, r['số lần chạy'])
    for _, r in rq1['tier2'].iterrows():
        add(r['chiến lược'], 'tier2', r['Macro-F1'], 'Macro-F1 (3 lớp)', {k: r[f'F1 {k}'] for k in ('BUG', 'FEATURE', 'UX')},
            {'actionable_recall': np.nan, 'actionable_f1': np.nan}, r['số lần chạy'])
    for _, r in rq1['e2e'].iterrows():
        add(name[r['cấu hình']], 'end_to_end', r['Macro-F1 (4 lớp)'], 'Macro-F1 (4 lớp)',
            {k: r[f'F1 {k}'] for k in ('BUG', 'FEATURE', 'UX', 'OTHER')}, {'actionable_recall': np.nan, 'actionable_f1': np.nan},
            r['số lần chạy'])
    out = pd.DataFrame(rows)
    out.loc[out['strategy'] == 'Baseline phẳng ViSoBERT', 'tier'] = 'flat_baseline_reference'
    return out


def check_keys(tables):
    """Referential integrity of the five tables; returns one row per foreign key."""
    rv, tp, rq, ev = tables['reviews'], tables['topics'], tables['requirements'], tables['requirement_evidence']
    checks = [
        ('reviews.review_id duy nhất', rv['review_id'].is_unique),
        ('reviews.topic_id → topics.topic_id', rv['topic_id'].dropna().isin(tp['topic_id']).all()),
        ('requirements.topic_id → topics.topic_id', rq['topic_id'].isin(tp['topic_id']).all()),
        ('requirement_evidence.requirement_id → requirements', ev['requirement_id'].isin(rq['requirement_id']).all()),
        ('requirement_evidence.review_id → reviews', ev['review_id'].isin(rv['review_id']).all()),
        ('mọi requirement có evidence', rq['requirement_id'].isin(ev['requirement_id']).all()),
    ]
    return pd.DataFrame(checks, columns=['ràng buộc', 'đạt'])
