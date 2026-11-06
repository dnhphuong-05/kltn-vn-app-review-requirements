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


def method_facts(repo):
    """Numbers printed on the method diagrams (notebook 17), read from saved results and frozen configs.
    Pilot κ and pilot class shares are the values reported in amendment v1.7 (pilot labels belong to the locked Hold-out)."""
    import yaml
    from src.reporting.style import pct, vn
    J = lambda *p: json.load(open(os.path.join(repo, *p), encoding='utf-8'))
    Y = lambda *p: yaml.safe_load(open(os.path.join(repo, *p), encoding='utf-8'))
    C = lambda *p, **k: pd.read_csv(os.path.join(repo, *p), **k)
    runs = {r['app']: r for r in J('01_data', 'manifests', 'gd2_interim_manifest_T0_2026-10-06.json')['runs']}
    sp = splits(repo)
    ag = agreement_from_nb04(os.path.join(repo, 'notebooks', '04_annotation_agreement.ipynb')).set_index('tập')
    t1, t2 = Y('03_configs', 'frozen', 'tier1_config.yaml')['tier1'], Y('03_configs', 'frozen', 'tier2_config.yaml')['tier2']
    ex = lambda t: len(J('04_prompts', 'classification_fewshot', f'tier{t}_exemplars_v1.json')['exemplars'])
    rq2 = os.path.join('05_experiments', 'rq2_topics')
    rq3 = os.path.join('05_experiments', 'rq3_requirements')
    topics = C(rq2, 'topics.csv', encoding='utf-8-sig')
    ps = J(rq2, 'gd12_manual_purity_summary.json')
    fz12 = Y('03_configs', 'frozen', 'rq2_topics_config.yaml')['rq2_topics']
    tg = C(rq2, 'gd13_temporal_groups.csv')
    g3 = Y('03_configs', 'frozen', 'rq3_generation_config.yaml')['rq3_generation']
    qa = C(rq3, 'gd14_scope_screening_qa_summary.csv').set_index('topic_id').loc['all']
    units = C(rq3, 'gd14_generation_units.csv', encoding='utf-8-sig')
    pool = C(rq3, 'gd14_evidence_pool.csv')
    reqs, ev = C(rq3, 'requirements.csv', encoding='utf-8-sig'), C(rq3, 'requirement_evidence.csv')
    f = {}
    for app in ('shopee', 'lazada'):
        s = runs[app]['steps']
        f[f'crawl_{app}'] = s['raw']
        f[f'out_window_{app}'] = s['outside_window_before'] + s['outside_window_after']
        f[f'no_letter_{app}'] = s['drop_empty'] + s['drop_no_letters']
        f[f'n_{app}'] = s['after_filter']
        f[f'dup_flag_{app}'] = runs[app]['flag_counts']['near_duplicate']
        f[f'lang_flag_{app}'] = runs[app]['flag_counts']['non_vi'] + runs[app]['flag_counts']['lang_und']
    f.update(n_train=sp['train'], n_val=sp['validation'], n_holdout=sp['holdout'], n_focused=sp['focused'],
             n_lazada_test=sp['lazada_external_test'],
             n_calib_batch=int(ag.loc['calibration batch2', 'n']), kappa_calib=vn(ag.loc['calibration batch2', 'κ 4 lớp'], 3),
             n_calib_fu=int(ag.loc['calibration FEATURE/UX (thảo luận)', 'n']),
             n_pilot=400, kappa_pilot='0,830', prev_actionable='20,5%', prev_feature='2,5%')
    f['n_calib_train'] = int(ag.loc['calibration batch1', 'n'] + ag.loc['calibration batch2', 'n'] + f['n_calib_fu'])
    f.update(thr_visobert=vn(t1['visobert']['threshold'], 2), thr_bilstm=vn(t1['bilstm']['threshold'], 2),
             min_recall=vn(t1['threshold_rule']['min_recall'], 2), fewshot_t1=ex(1), fewshot_t2=ex(2))
    f.update(n_actionable=int(topics['size'].sum()), n_topics=int((topics['topic_id'] != -1).sum()),
             n_outlier=int(topics.loc[topics['topic_id'] == -1, 'size'].sum()), min_topic_size=fz12['chosen']['min_topic_size'],
             n_purity_items=ps['n_review'], kappa_purity=vn(ps['cohen_kappa'], 2), n_flag_ba=ps['n_topic_flagged'],
             b90=vn(Y('03_configs', 'frozen', 'temporal_config.yaml')['temporal']['b90'], 4),
             n_recurring=int(((tg['type'] == 'ALL') & (tg['status'] == 'RECURRING')).sum()))
    f.update(ba_topics=', '.join(map(str, g3['selection']['ba_manual'])), n_ba_reviews=int((pool['exclusion_reason'] == 'ba_manual').sum()),
             qa_agree=pct(qa['agreement'], 1), qa_kappa=vn(qa['cohen_kappa'], 2), n_units=int(units['eligible'].sum()),
             n_evidence_pool=int(pool['used_for_generation'].sum()), k_shown=g3['evidence']['k_shown'],
             n_requirements=len(reqs), n_evidence_links=len(ev))
    return f
