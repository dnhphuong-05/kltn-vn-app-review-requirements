"""GĐ15 — evaluation of the Candidate Requirements (RQ3).

Automatic, structural (protocol RQ3):
  - `qus_defects`: user-story quality rules of the QUS framework, re-implemented after the defect types that AQUSA detects
    automatically (Lucassen et al., 2016): well-formed, atomic, minimal, explicit dependencies, uniform, unique. This is a
    re-implementation of the rules, not the original AQUSA tool.
  - `ac_validity`: Given/When/Then form and count of the acceptance criteria.
  - `near_duplicates`: semantic near-duplicate pairs of user stories (cosine of sentence embeddings), reported next to the
    exact-duplicate rule `unique`.
  - evidence-ID validity and traceability reuse `src.generation.core.verify_evidence`.
Human, semantic (protocol RQ3, amendment v2.9):
  - `write_sheets` / `read_sheets` / `score`: requirement-level ratings (fidelity, actionability, AC testability 1–5;
    hallucination and type correctness 0/1) and evidence-level judgements (does the cited review state the requirement's
    problem, 1/0), two raters, weighted κ for the 1–5 scales and Cohen κ for 0/1 judgements.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

TEMPLATE_RE = re.compile(r'^As an? (?P<role>[^,]+), I want (?P<means>.+?), so that (?P<ends>.+?)\.?$')
GWT_RE = re.compile(r'^Given (?P<given>.+?), when (?P<when>.+?), then (?P<then>.+?)\.?$', re.IGNORECASE)
CONJ_RE = re.compile(r'\b(and|or)\b|&|/|\+', re.IGNORECASE)
MINIMAL_RE = re.compile(r'[()\[\]{}]|\b(e\.g\.|i\.e\.|etc)\b|\.\.\.|…')
DEPENDENCY_RE = re.compile(r'#\d+|\b(see|after|before|depends on|requires) (story|requirement|us)\b', re.IGNORECASE)
QUS_RULES = ['well_formed', 'atomic', 'minimal', 'explicit_dependencies', 'uniform', 'unique']


SKELETON_RE = re.compile(r'^(?P<r>As an?|As)\b[^,]*,\s*(?P<m>I want|I need|I can|I would like)\b.*?(?P<e>,\s*so that\b.*)?$')


def _template(story):
    """Skeleton of the story template by position: role indicator | means indicator | ends indicator (or '' if absent)."""
    m = SKELETON_RE.match(story.strip())
    if not m:
        return 'non-template'
    return '|'.join(['As a', m['m'], 'so that' if m['e'] else ''])


def qus_defects(reqs):
    """One row per requirement: role/means/ends and one boolean column per QUS rule (True = defect)."""
    rows = []
    templates = reqs['user_story'].map(_template)
    modal = Counter(templates).most_common(1)[0][0]
    means_of = lambda s: m['means'].lower() if (m := TEMPLATE_RE.match(s.strip())) else ''
    means_count = Counter(reqs['user_story'].map(means_of))
    full_count = Counter(reqs['user_story'].str.strip().str.lower())
    for (_, r), tpl in zip(reqs.iterrows(), templates):
        s = r['user_story'].strip()
        m = TEMPLATE_RE.match(s)
        role, means, ends = (m['role'], m['means'], m['ends']) if m else ('', '', '')
        sentences = [x for x in re.split(r'(?<=[.!?])\s+', s) if x]
        rows.append({'requirement_id': r['requirement_id'], 'role': role, 'means': means, 'ends': ends,
                     'well_formed': not (m and role.strip() and means.strip()),
                     'atomic': bool(CONJ_RE.search(means)) or bool(CONJ_RE.search(ends)),
                     'minimal': bool(MINIMAL_RE.search(s)) or len(sentences) > 1,
                     'explicit_dependencies': bool(DEPENDENCY_RE.search(s)),
                     'uniform': tpl != modal,
                     'unique': full_count[s.lower()] > 1 or (bool(means) and means_count[means.lower()] > 1)})
    out = pd.DataFrame(rows)
    out['n_defects'] = out[QUS_RULES].sum(axis=1)
    return out


def qus_summary(defects):
    n = len(defects)
    rows = [{'rule': k, 'n_defect': int(defects[k].sum()), 'n_story': n, 'defect_rate': float(defects[k].mean())} for k in QUS_RULES]
    rows.append({'rule': 'any', 'n_defect': int((defects['n_defects'] > 0).sum()), 'n_story': n,
                 'defect_rate': float((defects['n_defects'] > 0).mean())})
    return pd.DataFrame(rows)


def ac_validity(reqs, ac_range=(2, 3)):
    """One row per acceptance criterion (GWT form, non-empty parts) and one row per requirement (count in range, all valid)."""
    crit, per = [], []
    for _, r in reqs.iterrows():
        ac = json.loads(r['acceptance_criteria'])
        ok = []
        for j, a in enumerate(ac, 1):
            m = GWT_RE.match(a.strip())
            valid = bool(m) and all(m[k].strip() for k in ('given', 'when', 'then'))
            ok.append(valid)
            crit.append({'requirement_id': r['requirement_id'], 'ac_no': j, 'gwt_valid': valid,
                         'n_then_clauses': len(re.findall(r'\bthen\b', a, re.IGNORECASE))})
        per.append({'requirement_id': r['requirement_id'], 'n_ac': len(ac), 'count_in_range': ac_range[0] <= len(ac) <= ac_range[1],
                    'all_gwt_valid': all(ok)})
    return pd.DataFrame(crit), pd.DataFrame(per)


def near_duplicates(reqs, embeddings, threshold=0.85):
    """Pairs of user stories with cosine ≥ threshold (embeddings normalised, same row order as reqs)."""
    sim = embeddings @ embeddings.T
    ids = reqs['requirement_id'].tolist()
    rows = [{'requirement_a': ids[i], 'requirement_b': ids[j], 'cosine': float(sim[i, j]),
             'cluster_a': int(reqs['cluster_id'].iloc[i]), 'cluster_b': int(reqs['cluster_id'].iloc[j])}
            for i in range(len(ids)) for j in range(i + 1, len(ids)) if sim[i, j] >= threshold]
    return pd.DataFrame(rows, columns=['requirement_a', 'requirement_b', 'cosine', 'cluster_a', 'cluster_b']).sort_values(
        'cosine', ascending=False, ignore_index=True)


# ---------------------------------------------------------------- human evaluation
REQ_SCALES = ['fidelity', 'actionability', 'ac_testability']
REQ_BINARY = ['hallucination', 'type_correct']
REQ_COLS = ['item_no', 'requirement_id', 'type', 'title_vi', 'problem_vi', 'user_story', 'acceptance_criteria', 'cited_reviews',
            *REQ_SCALES, *REQ_BINARY, 'hallucination_note', 'comment']
EV_COLS = ['item_no', 'requirement_id', 'problem_vi', 'user_story', 'evidence_label', 'review_id', 'text', 'supports', 'note']


def write_sheets(reqs, ev, text, out_dir, max_chars=600):
    """Blank sheets (never overwritten): requirement level (one row per requirement, cited review texts included so the
    rater judges against the Vietnamese sources) and evidence level (one row per cited review). Order is the order of
    `reqs`; raters must not reorder rows."""
    paths = {'req': os.path.join(out_dir, 'rq3_requirement_eval_blank.csv'), 'ev': os.path.join(out_dir, 'rq3_evidence_eval_blank.csv')}
    if all(os.path.exists(p) for p in paths.values()):
        return paths, False
    clip = lambda s: ' '.join(str(s).split())[:max_chars]
    e = ev.sort_values(['requirement_id', 'centroid_rank'])
    cited = e.groupby('requirement_id').apply(
        lambda g: '\n'.join(f"{l}: {clip(text[v])}" for l, v in zip(g['evidence_label'], g['review_id'])), include_groups=False)
    r = reqs.assign(acceptance_criteria=reqs['acceptance_criteria'].map(lambda s: '\n'.join(json.loads(s))),
                    cited_reviews=reqs['requirement_id'].map(cited))
    for c in [*REQ_SCALES, *REQ_BINARY, 'hallucination_note', 'comment']:
        r[c] = ''
    r.insert(0, 'item_no', range(1, len(r) + 1))
    r[REQ_COLS].to_csv(paths['req'], index=False, encoding='utf-8-sig')
    order = {k: i for i, k in enumerate(reqs['requirement_id'])}
    x = e.merge(reqs[['requirement_id', 'problem_vi', 'user_story']], on='requirement_id')
    x = x.sort_values(['requirement_id', 'centroid_rank'], key=lambda s: s.map(order) if s.name == 'requirement_id' else s)
    x = x.assign(text=x['review_id'].map(lambda v: clip(text[v])), supports='', note='')
    x.insert(0, 'item_no', range(1, len(x) + 1))
    x[EV_COLS].to_csv(paths['ev'], index=False, encoding='utf-8-sig')
    return paths, True


def read_sheets(out_dir, raters=('A', 'B')):
    """{rater: (requirement sheet, evidence sheet)} for the raters whose two sheets exist and are fully filled."""
    out = {}
    for k in raters:
        pr, pe = os.path.join(out_dir, f'rq3_requirement_eval_{k}.csv'), os.path.join(out_dir, f'rq3_evidence_eval_{k}.csv')
        if not (os.path.exists(pr) and os.path.exists(pe)):
            continue
        r, e = pd.read_csv(pr, encoding='utf-8-sig'), pd.read_csv(pe, encoding='utf-8-sig')
        if r[REQ_SCALES + REQ_BINARY].isna().any().any() or e['supports'].isna().any():
            continue
        for c in REQ_SCALES:
            assert r[c].isin([1, 2, 3, 4, 5]).all(), f'{k}: {c} phải là 1–5'
        for c in REQ_BINARY:
            assert r[c].isin([0, 1]).all(), f'{k}: {c} phải là 0/1'
        assert e['supports'].isin([0, 1]).all(), f'{k}: supports phải là 0/1'
        out[k] = (r.sort_values('item_no', ignore_index=True), e.sort_values('item_no', ignore_index=True))
    return out


def score(sheets, reqs):
    """(summary per criterion, per requirement table, agreement table or None). Uses the mean of the available raters."""
    from sklearn.metrics import cohen_kappa_score
    raters = list(sheets)
    r0 = sheets[raters[0]][0][['item_no', 'requirement_id']]
    per = r0.merge(reqs[['requirement_id', 'type', 'unit_id', 'cluster_id', 'mixed_origin_flag', 'issue_frequency']], on='requirement_id')
    for c in REQ_SCALES + REQ_BINARY:
        per[c] = np.mean([sheets[k][0][c].to_numpy(float) for k in raters], axis=0)
    ev = sheets[raters[0]][1][['item_no', 'requirement_id', 'review_id']].copy()
    ev['supports'] = np.mean([sheets[k][1]['supports'].to_numpy(float) for k in raters], axis=0)
    per = per.merge(ev.groupby('requirement_id')['supports'].mean().rename('evidence_precision'), on='requirement_id')
    rows = [{'criterion': c, 'scale': '1–5', 'mean': per[c].mean(), 'median': per[c].median(),
             'share_≥4': float((per[c] >= 4).mean())} for c in REQ_SCALES]
    rows += [{'criterion': f'{c} (tỷ lệ = 1)', 'scale': '0/1', 'mean': per[c].mean(), 'median': np.nan, 'share_≥4': np.nan}
             for c in REQ_BINARY]
    rows.append({'criterion': 'evidence precision (review trích nêu đúng vấn đề)', 'scale': '0/1 mỗi review',
                 'mean': float(ev['supports'].mean()), 'median': np.nan, 'share_≥4': np.nan})
    summary = pd.DataFrame(rows)
    agree = None
    if len(raters) >= 2:
        a, b = raters[:2]
        A, B = sheets[a], sheets[b]
        rows = [{'criterion': c, 'kappa': 'weighted (quadratic)',
                 'value': cohen_kappa_score(A[0][c], B[0][c], weights='quadratic', labels=[1, 2, 3, 4, 5]),
                 'exact_agreement': float((A[0][c] == B[0][c]).mean()), 'n': len(A[0])} for c in REQ_SCALES]
        rows += [{'criterion': c, 'kappa': 'Cohen', 'value': cohen_kappa_score(A[0][c], B[0][c], labels=[0, 1]),
                  'exact_agreement': float((A[0][c] == B[0][c]).mean()), 'n': len(A[0])} for c in REQ_BINARY]
        rows.append({'criterion': 'supports', 'kappa': 'Cohen', 'value': cohen_kappa_score(A[1]['supports'], B[1]['supports'], labels=[0, 1]),
                     'exact_agreement': float((A[1]['supports'] == B[1]['supports']).mean()), 'n': len(A[1])})
        agree = pd.DataFrame(rows)
    return summary, per, agree


def rater_comparison(sheets):
    """Descriptive comparison of the two raters (supplementary, not pre-registered): per 1–5 scale the mean of each
    rater, the mean difference, how often A scores below / equal / above B and the Spearman rank correlation; per 0/1
    judgement the rate of each rater, the rate flagged by both (lower bound) and by at least one (upper bound)."""
    from scipy.stats import spearmanr
    a, b = list(sheets)[:2]
    A, B = sheets[a][0], sheets[b][0]
    rows = []
    for c in REQ_SCALES:
        d = A[c] - B[c]
        rho, p = spearmanr(A[c], B[c])
        rows.append({'criterion': c, f'mean_{a}': A[c].mean(), f'mean_{b}': B[c].mean(), 'mean_diff': d.mean(),
                     f'{a}<{b}': int((d < 0).sum()), f'{a}={b}': int((d == 0).sum()), f'{a}>{b}': int((d > 0).sum()),
                     'spearman': rho, 'spearman_p': p})
    for c, (x, y) in [*((c, (A[c], B[c])) for c in REQ_BINARY), ('supports', (sheets[a][1]['supports'], sheets[b][1]['supports']))]:
        rows.append({'criterion': c, f'mean_{a}': x.mean(), f'mean_{b}': y.mean(), 'mean_diff': (x - y).mean(),
                     'both_1': float(((x == 1) & (y == 1)).mean()), 'either_1': float(((x == 1) | (y == 1)).mean())})
    return pd.DataFrame(rows)
