"""Loaders for the thesis tables/figures: every number comes from a file already produced by the pipeline.

Locked sets: nothing here reads gold labels of Hold-out / focused / Lazada. Their label counts and agreement figures are
taken from results already reported (notebook 04 outputs, gd10_* tables) — reading those is reporting, not re-opening.
"""
import json
import os
import re

import pandas as pd


def _outputs(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        for o in c.get('outputs', []):
            yield ''.join(o.get('text', '')) or ''.join(o.get('data', {}).get('text/plain', ''))


def agreement_from_nb04(nb_path):
    """Inter-annotator agreement as printed by notebook 04: the two calibration batches, the FEATURE/UX discussion set and
    the summary table of the double-labelled sets. Returns one row per set."""
    rows = []
    for t in _outputs(nb_path):
        for m in re.finditer(r'===== (batch\d): n=(\d+) \| % đồng thuận=([\d.]+) \| κ 4 lớp=([\d.]+)\n\s+actionable–OTHER: '
                             r'% đồng thuận=[\d.]+ \| κ=([\d.]+)\n\s+FEATURE–UX \(cả hai chọn FEATURE/UX\): n=(\d+) \| '
                             r'% đồng thuận=[\d.]+ \| κ=([\d.]+)', t):
            rows.append({'tập': f'calibration {m.group(1)}', 'n': int(m.group(2)), '% đồng thuận': float(m.group(3)),
                         'κ 4 lớp': float(m.group(4)), 'κ act–OTHER': float(m.group(5)), 'κ FEATURE–UX': float(m.group(7)),
                         'n FEATURE–UX': int(m.group(6))})
        m = re.search(r'n=(\d+) \| % đồng thuận=([\d.]+) \| κ 4 lớp=([\d.]+)\nFEATURE–UX \(cả hai chọn FEATURE/UX\): '
                      r'n=(\d+) \| % đồng thuận=[\d.]+ \| κ=([\d.]+)', t)
        if m and 'KHÔNG phải κ cổng' in t:
            rows.append({'tập': 'calibration FEATURE/UX (thảo luận)', 'n': int(m.group(1)), '% đồng thuận': float(m.group(2)),
                         'κ 4 lớp': float(m.group(3)), 'κ act–OTHER': float('nan'), 'κ FEATURE–UX': float(m.group(5)),
                         'n FEATURE–UX': int(m.group(4))})
        if 'Bảng tổng' in t:
            for line in t.splitlines():
                m = re.match(r'\s*(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$', line)
                if m:
                    rows.append({'tập': m.group(1), 'n': int(m.group(2)), '% đồng thuận': float(m.group(3)),
                                 'κ 4 lớp': float(m.group(4)), 'κ act–OTHER': float(m.group(5)),
                                 'κ FEATURE–UX': float(m.group(6)), 'n FEATURE–UX': int(m.group(7)),
                                 'số ca lệch': int(m.group(8))})
    seen, out = set(), []
    for r in rows:                                          # the same text can be printed by several cells
        if r['tập'] not in seen:
            seen.add(r['tập'])
            out.append(r)
    return pd.DataFrame(out)


def crawl_summary(repo):
    """Per app: crawl (manifest r2), out-of-window and no-letter drops, reviews kept (interim files)."""
    man = json.load(open(os.path.join(repo, '01_data', 'manifests', 'gd1_crawl_manifest_T0_2026-10-06_r2.json'), encoding='utf-8'))
    rows = []
    for r in man['runs']:
        if 'app' not in r:                  # bookkeeping entries of the manifest
            continue
        app = r['app']
        n_lines = lambda kind: sum(1 for _ in open(os.path.join(repo, '01_data', 'interim', f'{app}_{kind}_T0_2026-10-06.jsonl'),
                                                   encoding='utf-8'))
        clean, dropped = n_lines('clean'), n_lines('dropped')
        started = r.get('started_utc') or r['strata'][0]['started_utc']
        rows.append({'app': app.capitalize(), 'mode': r['mode'], 'ngày cào (UTC)': started[:10], 'cào được': r['n_reviews'],
                     'ngoài cửa sổ 12 tháng': r['n_reviews'] - clean - dropped, 'loại (không có chữ cái)': dropped,
                     'đưa vào nghiên cứu': clean, 'thiếu app_version': r['app_version_null_rate'],
                     'rating_counts': r['rating_counts']})
    return pd.DataFrame(rows)


def load_interim(repo, app):
    """review_id, rating, review_date (datetime, +07:00), words (word count of text_clean) of the clean interim file."""
    rows = [json.loads(l) for l in open(os.path.join(repo, '01_data', 'interim', f'{app}_clean_T0_2026-10-06.jsonl'), encoding='utf-8')]
    d = pd.DataFrame(rows)[['review_id', 'rating', 'review_date', 'text_clean']]
    d['review_date'] = pd.to_datetime(d['review_date'], utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
    d['words'] = d['text_clean'].astype(str).str.split().str.len()
    return d.drop(columns='text_clean')


def splits(repo):
    m = json.load(open(os.path.join(repo, '01_data', 'manifests', 'splits_manifest_T0_2026-10-06.json'), encoding='utf-8'))
    return {k: int(v['size']) for k, v in m['sets'].items()}


def label_distribution(repo):
    """4-class label counts: Train and Validation from the gold CSVs (not locked); Hold-out from the GĐ10 confusion
    table (row sums of one configuration = gold counts, already reported)."""
    order = ['BUG', 'FEATURE', 'UX', 'OTHER']
    out = {}
    for name, f in (('Train', 'shopee_train.csv'), ('Validation', 'shopee_validation.csv')):
        d = pd.read_csv(os.path.join(repo, '01_data', 'processed', f), encoding='utf-8-sig', usecols=['gold_label'])
        out[name] = d['gold_label'].value_counts().reindex(order, fill_value=0)
    cm = pd.read_csv(os.path.join(repo, '05_experiments', 'rq1_classification', 'gd10_confusion_holdout.csv'), index_col=0)
    first = cm[cm['cấu_hình'] == cm['cấu_hình'].iloc[0]].drop(columns='cấu_hình')
    out['Hold-out'] = first.sum(axis=1).rename(lambda s: s.replace('gold ', '')).reindex(order).astype(int)
    return pd.DataFrame(out).T
