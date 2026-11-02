"""GĐ14 — Candidate Requirement generation (RQ3): User Story + Acceptance Criteria + Evidence IDs.

Pipeline (notebook 14 imports these functions):
  1. `build_units` — apply amendment v2.6: which clusters generate requirements, which reviews are evidence. A
     *generation unit* is a cluster, or a (cluster × type) subgroup when the evidence set is mixed (Q14). Every
     actionable review of a cluster gets either a unit or an exclusion reason, so the structured output is complete.
  2. `unit_temporal` — frozen temporal rule (v2.5) recomputed on the exact evidence set of each unit.
  3. `shown_evidence` — the k reviews closest to the centroid of the unit's evidence set (mpnet, GĐ12), labelled E1..Ek.
  4. `generate` — one LLM call per unit (locked prompt, strict JSON schema whose evidence ids are an enum E1..Ek;
     prompt v2 decodes the evidence ids and the shared problem before the story),
     backend validation of every requirement, one corrective retry, log without review text.
  5. `requirements_table` / `evidence_table` — requirements.csv and requirement_evidence.csv.
"""
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import yaml

from src.utils.kaggle_paths import _repo_root

PROMPT_DIR = os.path.join(_repo_root(), '04_prompts', 'requirement_generation')
TYPES = ['BUG', 'FEATURE', 'UX']
API_SEED = 42
K_MAX = 20
EVIDENCE_LABELS = [f'E{i}' for i in range(1, K_MAX + 1)]
def _schema(fields):
    props = {'title_vi': {'type': 'string'}, 'problem_vi': {'type': 'string'}, 'user_story': {'type': 'string'},
             'acceptance_criteria': {'type': 'array', 'items': {'type': 'string'}},
             'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'enum': EVIDENCE_LABELS}}}
    item = {'type': 'object', 'additionalProperties': False, 'required': fields, 'properties': {f: props[f] for f in fields}}
    return {'type': 'json_schema',
            'json_schema': {'name': 'candidate_requirements', 'strict': True,
                            'schema': {'type': 'object', 'additionalProperties': False, 'required': ['requirements'],
                                       'properties': {'requirements': {'type': 'array', 'items': item}}}}}


# v2: evidence first, then the shared problem, so that the story is written from the cited reviews (field order = decoding order)
SCHEMAS = {'v1': _schema(['title_vi', 'user_story', 'acceptance_criteria', 'evidence_ids']),
           'v2': _schema(['evidence_ids', 'problem_vi', 'title_vi', 'user_story', 'acceptance_criteria']),
           'v3': _schema(['evidence_ids', 'problem_vi', 'title_vi', 'user_story', 'acceptance_criteria'])}
US_RE = re.compile(r'^As an? [^,]+, I want .+, so that .+\.$')
AC_RE = re.compile(r'^Given .+, when .+, then .+\.$', re.IGNORECASE)


def load_config(repo_dir, stage='dev'):
    path = os.path.join(repo_dir, '03_configs', stage, 'rq3_generation_config.yaml' if stage == 'frozen'
                        else 'rq3_generation_dev_config.yaml')
    return yaml.safe_load(open(path, encoding='utf-8'))['rq3_generation']


def load_prompt(version='v3'):
    """(system text, prompt_hash, schema); the hash covers the system text and the output schema."""
    system = open(os.path.join(PROMPT_DIR, f'requirement_generation_system_{version}.txt'), encoding='utf-8').read().strip()
    schema = SCHEMAS[version]
    payload = json.dumps({'system': system, 'schema': schema}, ensure_ascii=False, sort_keys=True)
    return system, hashlib.sha256(payload.encode('utf-8')).hexdigest(), schema


# ---------------------------------------------------------------- 1. units and evidence pools
def _dominant(types, d):
    """Dominant type (≥ threshold) or None, and the Q14 types kept when mixed."""
    c = pd.Series(types).value_counts().reindex(TYPES, fill_value=0)
    n = int(c.sum())
    top = c.idxmax()
    if n and c[top] / n >= d['threshold']:
        return top, [top]
    keep = [t for t in TYPES if c[t] >= d['mixed_origin_min_reviews'] or c[t] / n >= d['mixed_origin_min_share']]
    return None, keep


def build_units(cfg, topics, rmap, screened):
    """(units, pool). pool: one row per clustered actionable review — unit_id or exclusion_reason.
    units: unit_id, topic_id, type, step, n_pool, mixed_origin_flag, scope_note (temporal added later)."""
    sel, d = cfg['selection'], cfg['dominant_type']
    t = topics.set_index('topic_id')
    lab = screened.set_index('review_id')['label']
    m = rmap[['review_id', 'topic_id', 'tier2_label']].copy()
    m['unit_id'], m['exclusion_reason'] = None, None
    units = []

    def add(tid, typ, mask, step, flag, note, split):
        uid = f'{tid}-{typ}' if split else str(tid)
        m.loc[mask, 'unit_id'] = uid
        units.append({'unit_id': uid, 'topic_id': tid, 'type': typ, 'step': step, 'n_pool': int(mask.sum()),
                      'topic_size': int(t.loc[tid, 'size']), 'mixed_origin_flag': flag, 'scope_note': note})

    for tid in sorted(x for x in m['topic_id'].unique() if x != -1):
        in_t = m['topic_id'] == tid
        size = int(in_t.sum())
        dom = t.loc[tid, 'dominant_type']
        if tid in sel['ba_manual']:
            m.loc[in_t, 'exclusion_reason'] = 'ba_manual'
        elif tid in sel['scope_filter']:
            app = in_t & (m['review_id'].map(lab) == 'app_actionable')
            m.loc[in_t & ~app, 'exclusion_reason'] = 'out_of_scope_app'
            base = (f'Cụm lọc theo nội dung chức năng app (v2.6 bước 2): {int(app.sum())}/{size} review app_actionable làm '
                    f'evidence; {size - int(app.sum())} review ngoài phạm vi app không dùng.')
            top, keep = _dominant(m.loc[app, 'tier2_label'], d)
            if top:
                add(tid, top, app, 'scope_filter', True, base, split=False)
            else:
                for typ in TYPES:
                    sub = app & (m['tier2_label'] == typ)
                    if typ in keep:
                        add(tid, typ, sub, 'scope_filter', True,
                            base + f' Tập đã lọc là mixed, tách theo loại (Q14): {typ} {int(sub.sum())} review.', split=True)
                    else:
                        m.loc[sub, 'exclusion_reason'] = 'mixed_origin_below_q14'
        elif tid in sel['type_filter']:
            keep = in_t & (m['tier2_label'] == dom)
            m.loc[in_t & ~keep, 'exclusion_reason'] = 'type_mismatch'
            add(tid, dom, keep, 'type_filter', True,
                f'Evidence chỉ gồm review {dom} ({int(keep.sum())}/{size}) (v2.6 bước 3).', split=False)
        elif tid in sel['q14_split']:
            origin = str(t.loc[tid, 'mixed_origin_types']).split(';')
            for typ in TYPES:
                sub = in_t & (m['tier2_label'] == typ)
                if typ in origin:
                    add(tid, typ, sub, 'q14_split', True,
                        f'Cụm mixed, tách theo loại (Q14): {typ} {int(sub.sum())}/{size} review.', split=True)
                else:
                    m.loc[sub, 'exclusion_reason'] = 'mixed_origin_below_q14'
        elif tid in sel['normal']:
            add(tid, dom, in_t, 'normal', False, '', split=False)
        else:
            raise ValueError(f'cụm {tid} không thuộc nhóm nào của amendment v2.6')
    m.loc[m['topic_id'] == -1, 'exclusion_reason'] = 'outlier'
    units = pd.DataFrame(units)
    ads = cfg['duplicate_theme']['ads']
    gen_ads = [x for x in ads if x in set(units['topic_id'])]
    for i, r in units.iterrows():
        if r['topic_id'] in ads:
            others = ', '.join(str(x) for x in gen_ads if x != r['topic_id'])
            ba = ', '.join(str(x) for x in ads if x in sel['ba_manual'])
            note = (f'Cùng chủ đề quảng cáo với cụm {others} (và cụm {ba} chuyển BA): requirement có thể trùng, BA gộp nếu cần.')
            units.at[i, 'scope_note'] = (r['scope_note'] + ' ' + note).strip()
    assert m['unit_id'].notna().ne(m['exclusion_reason'].notna()).all(), 'mỗi review phải có đúng một unit hoặc một lý do loại'
    return units, m


def unit_temporal(units, pool, actionable, tcfg, TC):
    """Frozen temporal rule (v2.5) on each unit's evidence set; adds N, c30, R90, n_win, status, eligible."""
    a = pool.merge(actionable[['review_id', 'review_date', 'app_version']], on='review_id', how='left', validate='one_to_one')
    dates = TC.local_dates(a['review_date'], tcfg)
    rows = []
    for uid in units['unit_id']:
        mask = (a['unit_id'] == uid).to_numpy()
        st = TC.group_stats(dates[mask], a.loc[mask, 'app_version'].to_numpy(), tcfg, len(actionable))
        s = TC.status(st, tcfg['rule'])
        rows.append({'unit_id': uid, 'N': st['N'], 'c30': st['c30'], 'c90': st['c90'], 'R90': st['R90'], 'n_win': st['n_win'],
                     'temporal_status': s, 'eligible': s in tcfg['statuses_for_generation']})
    out = units.merge(pd.DataFrame(rows), on='unit_id')
    out['issue_frequency'] = out['n_pool'] / len(actionable)
    return out


# ---------------------------------------------------------------- 2. evidence shown to the LLM
def shown_evidence(pool, unit_id, embeddings, act_index, k):
    """review_ids of the k evidence reviews closest to the centroid of the unit's evidence set, best first."""
    ids = pool.loc[pool['unit_id'] == unit_id, 'review_id'].to_numpy()
    e = embeddings[[act_index[x] for x in ids]]
    c = e.mean(0)
    c = c / np.linalg.norm(c)
    return ids[np.argsort(-(e @ c), kind='stable')[:k]].tolist()


def user_message(unit, topic, shown_ids, text, max_chars):
    lines = [f"Issue group: {unit['unit_id']}", f"Requirement type: {unit['type']}",
             f"Cluster name: {topic['name_en']}", f"Cluster keywords: {topic['top_keywords']}",
             f"Reviews in the group: {unit['n_pool']}", '', 'Reviews (Vietnamese, closest to the centre of the group):']
    for i, rid in enumerate(shown_ids, 1):
        lines.append(f"E{i}: {' '.join(str(text[rid]).split())[:max_chars]}")
    return '\n'.join(lines)


# ---------------------------------------------------------------- 3. generation + validation
def check(out, unit_type, n_shown, cfg):
    """Rule violations of one parsed answer (empty list = valid)."""
    o, e = cfg['output'], cfg['evidence']
    issues = []
    reqs = out.get('requirements', [])
    if not o['requirements_per_unit'][0] <= len(reqs) <= o['requirements_per_unit'][1]:
        issues.append(f'số requirement {len(reqs)} ngoài khoảng {o["requirements_per_unit"]}')
    for j, r in enumerate(reqs, 1):
        if not any(r['title_vi'].startswith(p) for p in o['title_prefix'][unit_type]):
            issues.append(f'R{j}: title_vi phải bắt đầu bằng {" / ".join(o["title_prefix"][unit_type])}')
        if not US_RE.match(r['user_story'].strip()):
            issues.append(f'R{j}: user_story sai dạng "As a <role>, I want <goal>, so that <benefit>."')
        ac = r['acceptance_criteria']
        if not o['acceptance_criteria'][0] <= len(ac) <= o['acceptance_criteria'][1]:
            issues.append(f'R{j}: số acceptance criteria {len(ac)} ngoài khoảng {o["acceptance_criteria"]}')
        for a in ac:
            if not AC_RE.match(a.strip()):
                issues.append(f'R{j}: acceptance criterion sai dạng "Given ..., when ..., then ...."')
        ev = r['evidence_ids']
        if len(set(ev)) != len(ev):
            issues.append(f'R{j}: evidence_ids lặp')
        if not e['cite_min'] <= len(set(ev)) <= e['cite_max']:
            issues.append(f'R{j}: số evidence {len(set(ev))} ngoài khoảng [{e["cite_min"]}, {e["cite_max"]}]')
        if 'problem_vi' in r and not r['problem_vi'].strip():
            issues.append(f'R{j}: problem_vi rỗng')
        bad = [x for x in ev if x not in EVIDENCE_LABELS[:n_shown]]
        if bad:
            issues.append(f'R{j}: evidence ngoài danh sách đưa vào prompt: {bad}')
    return issues


_RETRY = ('Your answer breaks these rules:\n{issues}\nReturn the corrected JSON, following every rule of the instructions.')


def generate(units, system, schema, client, settings, cfg, payloads, log_path, log=print):
    """One call per unit (payloads: {unit_id: (user message, shown review_ids)}); one corrective retry when the answer
    does not parse or breaks a rule. Returns one row per unit with the parsed answer and its remaining violations."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    rows = []
    with open(log_path, 'a', encoding='utf-8') as lf:
        for _, u in units.iterrows():
            msg, shown = payloads[u['unit_id']]
            msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': msg}]
            out, issues = None, ['chưa gọi']
            tok_in = tok_out = 0
            t0 = time.time()
            for attempt in (1, 2):
                resp, errors = None, []
                for k in range(4):
                    try:
                        resp = client.chat.completions.create(model=settings.snapshot, temperature=settings.temperature,
                                                              seed=API_SEED, max_tokens=int(cfg['max_tokens']),
                                                              messages=msgs, response_format=schema)
                        break
                    except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                        errors.append(f'{type(e).__name__}: {str(e)[:200]}')
                        time.sleep(2 ** k)
                if resp is None:
                    raise RuntimeError(f'{u["unit_id"]}: API lỗi 4 lần liên tiếp: {errors[-1]}')
                tok_in, tok_out = tok_in + resp.usage.prompt_tokens, tok_out + resp.usage.completion_tokens
                content = resp.choices[0].message.content
                try:
                    out = json.loads(content)
                    issues = check(out, u['type'], len(shown), cfg)
                except (ValueError, KeyError, TypeError) as e:
                    out, issues = None, [f'không parse được JSON: {type(e).__name__}']
                lf.write(json.dumps({'unit_id': u['unit_id'], 'attempt': attempt, 'shown_review_ids': shown, 'raw': content,
                                     'issues': issues, 'prompt_tokens': resp.usage.prompt_tokens,
                                     'completion_tokens': resp.usage.completion_tokens, 'model': resp.model,
                                     'system_fingerprint': resp.system_fingerprint, 'api_errors': errors},
                                    ensure_ascii=False) + '\n')
                if not issues:
                    break
                msgs = msgs + [{'role': 'assistant', 'content': content},
                               {'role': 'user', 'content': _RETRY.format(issues='\n'.join(f'- {x}' for x in issues))}]
            rows.append({'unit_id': u['unit_id'], 'answer': out, 'issues': issues, 'attempts': attempt,
                         'prompt_tokens': tok_in, 'completion_tokens': tok_out, 'seconds': time.time() - t0,
                         'model': resp.model})
            log(f"  {u['unit_id']}: {len(out['requirements']) if out else 0} requirement, lần {attempt}"
                + (f', còn vi phạm: {issues}' if issues else ''))
    return pd.DataFrame(rows)


def cost(gen, prices):
    usd = (gen['prompt_tokens'].sum() * prices['input'] + gen['completion_tokens'].sum() * prices['output']) / 1e6
    return {'n_unit': len(gen), 'token_vào': int(gen['prompt_tokens'].sum()), 'token_ra': int(gen['completion_tokens'].sum()),
            'chi_phí_usd': float(usd), 'giây': float(gen['seconds'].sum()), 'thử_lại': int((gen['attempts'] > 1).sum()),
            'unit_còn_vi_phạm': int(gen['issues'].map(bool).sum())}


# ---------------------------------------------------------------- 4. tables + backend evidence check
def requirement_id(unit_id, j):
    tid, _, typ = unit_id.partition('-')
    return f'REQ-{int(tid):02d}' + (f'-{typ}' if typ else '') + f'-{j}'


def requirements_table(gen, units, payloads):
    """requirements.csv (one row per requirement) and requirement_evidence.csv (one row per cited review)."""
    u = units.set_index('unit_id')
    reqs, ev = [], []
    for _, g in gen.iterrows():
        if not g['answer']:
            continue
        shown = payloads[g['unit_id']][1]
        x = u.loc[g['unit_id']]
        for j, r in enumerate(g['answer']['requirements'], 1):
            rid = requirement_id(g['unit_id'], j)
            labels = list(dict.fromkeys(r['evidence_ids']))
            review_ids = [shown[int(e[1:]) - 1] for e in labels if int(e[1:]) <= len(shown)]
            reqs.append({'requirement_id': rid, 'cluster_id': int(x['topic_id']), 'type': x['type'],
                         'user_story': r['user_story'].strip(),
                         'acceptance_criteria': json.dumps([a.strip() for a in r['acceptance_criteria']], ensure_ascii=False),
                         'evidence_ids': ';'.join(review_ids), 'temporal_status': x['temporal_status'],
                         'mixed_origin_flag': bool(x['mixed_origin_flag']), 'scope_note': x['scope_note'],
                         'title_vi': r['title_vi'].strip(), 'problem_vi': r.get('problem_vi', '').strip(), 'unit_id': g['unit_id'], 'issue_frequency': x['issue_frequency'],
                         'n_evidence_pool': int(x['n_pool']), 'n_evidence_cited': len(review_ids),
                         'validation_issues': '; '.join(g['issues'])})
            ev += [{'requirement_id': rid, 'review_id': v, 'evidence_label': e, 'centroid_rank': int(e[1:])}
                   for e, v in zip(labels, review_ids)]
    return pd.DataFrame(reqs), pd.DataFrame(ev)


def verify_evidence(reqs, ev, pool, actionable, rmap, units, payloads):
    """Backend traceability check (protocol RQ3): every evidence id exists in the actionable corpus, belongs to the
    requirement's cluster and unit evidence set, and was in the list given to the prompt. Returns one row per check."""
    act = set(actionable['review_id'])
    topic_of = rmap.set_index('review_id')['topic_id']
    unit_of = pool.set_index('review_id')['unit_id']
    r_unit = reqs.set_index('requirement_id')['unit_id']
    r_topic = reqs.set_index('requirement_id')['cluster_id']
    e = ev.assign(unit_id=ev['requirement_id'].map(r_unit), cluster_id=ev['requirement_id'].map(r_topic))
    checks = {
        'tồn tại trong corpus actionable': e['review_id'].isin(act),
        'thuộc đúng cụm': e['review_id'].map(topic_of).eq(e['cluster_id']),
        'thuộc tập evidence của unit': e['review_id'].map(unit_of).eq(e['unit_id']),
        'nằm trong danh sách đưa vào prompt': [rv in payloads[u][1] for rv, u in zip(e['review_id'], e['unit_id'])],
    }
    out = pd.DataFrame([{'kiểm tra': k, 'đạt': int(np.sum(v)), 'tổng': len(e)} for k, v in checks.items()])
    out['hợp lệ'] = out['đạt'] == out['tổng']
    per_req = reqs['requirement_id'].isin(ev['requirement_id']) & (reqs['n_evidence_cited'] > 0)
    return out, bool(out['hợp lệ'].all() and per_req.all())


def ba_manual_table(pool, topics, purity):
    """Clusters sent to BA (v2.6 step 1): one row per review, no text."""
    b = pool[pool['exclusion_reason'] == 'ba_manual'][['topic_id', 'review_id', 'tier2_label']]
    t = topics.set_index('topic_id')
    p = purity.set_index('topic_id')['purity']
    return b.assign(name_vi=b['topic_id'].map(t['name_vi']), size=b['topic_id'].map(t['size']),
                    manual_purity=b['topic_id'].map(p), note='cần BA xem thủ công')[
        ['topic_id', 'name_vi', 'size', 'manual_purity', 'review_id', 'tier2_label', 'note']].sort_values(['topic_id', 'review_id'])
