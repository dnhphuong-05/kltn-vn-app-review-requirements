"""GĐ13 — Temporal Issue Validation (ĐC §7.5; rule pre-registered in amendment v2.5, 03_configs/frozen/temporal_config.yaml).

For every issue group (each non-outlier topic; for a `mixed` topic also each Q14 mixed-origin (topic × type) subgroup):
  N          reviews in the 12 windows of 30 days ending at T0 (W0 = last 30 days ... W11)
  c30/60/90  reviews in the last 30 / 60 / 90 days
  R90        (c90 / N) / B90 — recency relative to the whole actionable corpus
  n_win      windows with ≥ presence_min_reviews reviews of the group
Rule, in order: N < min_n → INSUFFICIENT_EVIDENCE; c30 ≥ min_c30 and R90 ≥ min_r90 → RECURRING if n_win ≥
recurring_min_windows else ACTIVE; R90 < min_r90 → POSSIBLY_RESOLVED; otherwise INSUFFICIENT_EVIDENCE.
Dates are the evidence; app versions are only reported (first/last). Never FIXED, no causal reading of an update.
"""
import os

import numpy as np
import pandas as pd
import yaml

STATUSES = ('ACTIVE', 'RECURRING', 'POSSIBLY_RESOLVED', 'INSUFFICIENT_EVIDENCE')


def load_config(repo_dir):
    path = os.path.join(repo_dir, '03_configs', 'frozen', 'temporal_config.yaml')
    head = open(path, encoding='utf-8').readline()
    if 'FROZEN' not in head or 'v2.5' not in head:
        raise RuntimeError(f'{path}: thiếu header FROZEN v2.5')
    return yaml.safe_load(open(path, encoding='utf-8'))['temporal']


def t0(cfg):
    return pd.Timestamp(cfg['t0'], tz=cfg['timezone'])


def local_dates(review_date, cfg):
    return pd.to_datetime(review_date, utc=True).dt.tz_convert(cfg['timezone'])


def age_days(dates, cfg):
    return (t0(cfg) - dates).dt.total_seconds().to_numpy() / 86400


def window_index(age, cfg):
    """0 = last window_days before T0 ... n_windows−1; −1 for reviews older than the windows (or after T0)."""
    w = np.floor(age / cfg['window_days']).astype(int)
    return np.where((age >= 0) & (w < cfg['n_windows']), w, -1)


def corpus_b90(dates, cfg):
    """Share of all actionable reviews dated in the last 90 days (the pre-registered B90 is checked against this)."""
    return float((age_days(dates, cfg) <= 90).mean())


def _versions(versions, dates):
    v = pd.DataFrame({'v': versions, 'd': dates}).dropna(subset=['v']).sort_values('d')
    return (v['v'].iloc[0], v['v'].iloc[-1]) if len(v) else (None, None)


def group_stats(dates, versions, cfg, n_actionable):
    """Evidence of one issue group (dates: tz-aware Series)."""
    age = age_days(dates, cfg)
    win = window_index(age, cfg)
    in_hist = win >= 0
    per_win = np.bincount(win[in_hist], minlength=cfg['n_windows'])
    n = int(in_hist.sum())
    c = {k: int(((age >= 0) & (age <= k)).sum()) for k in (30, 60, 90)}
    first_v, last_v = _versions(versions, dates)
    return {'N': n, 'n_all': len(dates), 'historical_frequency': len(dates) / n_actionable,
            'c30': c[30], 'c60': c[60], 'c90': c[90], 'R90': (c[90] / n) / cfg['b90'] if n else float('nan'),
            'n_win': int((per_win >= cfg['rule']['presence_min_reviews']).sum()),
            'first_seen': dates.min(), 'last_seen': dates.max(), 'first_version': first_v, 'last_version': last_v,
            'version_coverage': float(pd.Series(versions).notna().mean()), 'per_window': per_win.tolist()}


def status(st, rule):
    if st['N'] < rule['min_n']:
        return 'INSUFFICIENT_EVIDENCE'
    if st['c30'] >= rule['min_c30'] and st['R90'] >= rule['min_r90']:
        return 'RECURRING' if st['n_win'] >= rule['recurring_min_windows'] else 'ACTIVE'
    if st['R90'] < rule['min_r90']:
        return 'POSSIBLY_RESOLVED'
    return 'INSUFFICIENT_EVIDENCE'


def issue_groups(review_topic_map, topics):
    """[(group_id, topic_id, type or 'ALL', mask over the map rows)] — topics, plus Q14 subgroups of mixed topics."""
    groups = []
    t = topics[topics['topic_id'] != -1].set_index('topic_id')
    for tid in sorted(t.index):
        in_topic = (review_topic_map['topic_id'] == tid).to_numpy()
        groups.append((f'T{tid}', int(tid), 'ALL', in_topic))
        if t.loc[tid, 'dominant_type'] == 'mixed' and isinstance(t.loc[tid, 'mixed_origin_types'], str):
            for typ in t.loc[tid, 'mixed_origin_types'].split(';'):
                groups.append((f'T{tid}-{typ}', int(tid), typ, in_topic & (review_topic_map['tier2_label'] == typ).to_numpy()))
    return groups


def evaluate(review_topic_map, actionable, topics, cfg, rule=None):
    """One row per issue group: evidence + status + eligibility for generation."""
    rule = rule or cfg['rule']
    a = review_topic_map.merge(actionable[['review_id', 'review_date', 'app_version']], on='review_id', how='left', validate='one_to_one')
    dates = local_dates(a['review_date'], cfg)
    n_act = len(actionable)
    rows = []
    for gid, tid, typ, mask in issue_groups(review_topic_map, topics):
        st = group_stats(dates[mask], a.loc[mask, 'app_version'].to_numpy(), cfg, n_act)
        s = status(st, rule)
        rows.append({'group_id': gid, 'topic_id': tid, 'type': typ, **st, 'status': s,
                     'eligible_for_generation': s in cfg['statuses_for_generation']})
    return pd.DataFrame(rows)


def sensitivity(review_topic_map, actionable, topics, cfg):
    """One parameter changed at a time; counts per status and number of groups whose status differs from the main rule."""
    base = evaluate(review_topic_map, actionable, topics, cfg)
    rows = [{'tham số': 'luật chính', 'giá trị': '', **base['status'].value_counts().reindex(STATUSES, fill_value=0).to_dict(),
             'số group đổi trạng thái': 0}]
    for param, values in cfg['sensitivity'].items():
        for v in values:
            alt = evaluate(review_topic_map, actionable, topics, cfg, rule={**cfg['rule'], param: v})
            rows.append({'tham số': param, 'giá trị': v,
                         **alt['status'].value_counts().reindex(STATUSES, fill_value=0).to_dict(),
                         'số group đổi trạng thái': int((alt['status'].to_numpy() != base['status'].to_numpy()).sum())})
    return base, pd.DataFrame(rows)


def window_labels(cfg):
    """'dd/mm–dd/mm' for W11 ... W0 (oldest first)."""
    end = t0(cfg)
    out = []
    for k in range(cfg['n_windows'] - 1, -1, -1):
        hi = end - pd.Timedelta(days=cfg['window_days'] * k)
        lo = hi - pd.Timedelta(days=cfg['window_days']) + pd.Timedelta(seconds=1)
        out.append(f"{lo:%d/%m}–{hi:%d/%m}")
    return out
