"""GĐ14 step 1 — scope screening of topics 0 and 8 (amendment v2.6, step 2).

Each review of the two topics is labelled by the LLM (gpt-4o-mini snapshot, temperature 0, locked prompt) as
`app_actionable` (states a bug or a concrete request about the app's functions) or `out_of_scope` (pure logistics /
shipper / customer service, asking Shopee to police sellers or change policy, product quality or price, general praise
or complaint). Only `app_actionable` reviews may become evidence of a requirement. Every call is logged as one JSON line
(review_id, raw output, tokens, latency) — no review text.
"""
import hashlib
import json
import os
import time

import pandas as pd

from src.utils.kaggle_paths import _repo_root

PROMPT_DIR = os.path.join(_repo_root(), '04_prompts', 'requirement_generation')
REGISTRY = os.path.join(_repo_root(), '04_prompts', 'prompts_registry.md')
LABELS = ['app_actionable', 'out_of_scope']
API_SEED = 42
MAX_TOKENS = 80
SCHEMA = {'type': 'json_schema',
          'json_schema': {'name': 'scope_screening', 'strict': True,
                          'schema': {'type': 'object', 'additionalProperties': False, 'required': ['label', 'reason'],
                                     'properties': {'label': {'type': 'string', 'enum': LABELS},
                                                    'reason': {'type': 'string'}}}}}


def load_prompt(version='v1'):
    """(system text, prompt_hash); the hash covers the system text and the output schema."""
    system = open(os.path.join(PROMPT_DIR, f'scope_screening_system_{version}.txt'), encoding='utf-8').read().strip()
    payload = json.dumps({'system': system, 'schema': SCHEMA}, ensure_ascii=False, sort_keys=True)
    return system, hashlib.sha256(payload.encode('utf-8')).hexdigest()


def registered(prompt_hash):
    return prompt_hash in open(REGISTRY, encoding='utf-8').read()


def screen(df, system, client, settings, log_path, id_col='review_id', text_col='text', progress_every=200, log=print):
    """One row per review: label (None if unparsable after one retry), reason, tokens, seconds, attempts."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    rows = []
    with open(log_path, 'a', encoding='utf-8') as lf:
        for i, (rid, text) in enumerate(zip(df[id_col], df[text_col].astype(str)), 1):
            msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': f'Review: {text}'}]
            label = reason = None
            tok_in = tok_out = cached = 0
            t0 = time.time()
            for attempt in (1, 2):                          # one call + one retry when the answer does not parse
                resp, errors = None, []
                for k in range(4):
                    try:
                        resp = client.chat.completions.create(model=settings.snapshot, temperature=settings.temperature,
                                                              seed=API_SEED, max_tokens=MAX_TOKENS, messages=msgs,
                                                              response_format=SCHEMA)
                        break
                    except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                        errors.append(f'{type(e).__name__}: {str(e)[:200]}')
                        time.sleep(2 ** k)
                if resp is None:
                    raise RuntimeError(f'{rid}: API lỗi 4 lần liên tiếp: {errors[-1]}')
                u = resp.usage
                tok_in, tok_out = tok_in + u.prompt_tokens, tok_out + u.completion_tokens
                details = getattr(u, 'prompt_tokens_details', None)
                cached += getattr(details, 'cached_tokens', 0) or 0
                content = resp.choices[0].message.content
                lf.write(json.dumps({'review_id': rid, 'attempt': attempt, 'raw': content, 'prompt_tokens': u.prompt_tokens,
                                     'completion_tokens': u.completion_tokens, 'model': resp.model,
                                     'system_fingerprint': resp.system_fingerprint, 'api_errors': errors},
                                    ensure_ascii=False) + '\n')
                try:
                    out = json.loads(content)
                    if out['label'] in LABELS:
                        label, reason = out['label'], str(out['reason']).strip()
                        break
                except (ValueError, KeyError, TypeError):
                    pass
            rows.append({id_col: rid, 'label': label, 'reason': reason, 'attempts': attempt, 'prompt_tokens': tok_in,
                         'cached_tokens': cached, 'completion_tokens': tok_out, 'seconds': time.time() - t0,
                         'model': resp.model})
            if progress_every and i % progress_every == 0:
                log(f'  {i}/{len(df)}')
    return pd.DataFrame(rows)


def cost(df, prices):
    usd = (df['prompt_tokens'].sum() * prices['input'] + df['completion_tokens'].sum() * prices['output']) / 1e6
    return {'n_review': len(df), 'token_vào': int(df['prompt_tokens'].sum()), 'token_vào_cached': int(df['cached_tokens'].sum()),
            'token_ra': int(df['completion_tokens'].sum()), 'chi_phí_usd': float(usd), 'giây': float(df['seconds'].sum()),
            'không_parse_được': int(df['label'].isna().sum()), 'thử_lại': int((df['attempts'] > 1).sum())}


def qa_sheet(screened, actionable, topics, per_label=10, seed=42):
    """QA sample per topic: per_label random reviews of each label (20 per topic by default), with text and an empty
    `qa_label` column for the researcher; order shuffled inside a topic so the label is not guessed from position."""
    text = actionable.set_index('review_id')['text']
    rows = []
    for tid, g in screened.groupby('topic_id'):
        parts = [x.sample(min(per_label, len(x)), random_state=seed) for _, x in g.groupby('label')]
        s = pd.concat(parts).sample(frac=1, random_state=seed)
        for _, r in s.iterrows():
            rows.append({'topic_id': int(tid), 'review_id': r['review_id'], 'text': text[r['review_id']],
                         'llm_label': r['label'], 'llm_reason': r['reason'], 'qa_label': '', 'qa_note': ''})
    out = pd.DataFrame(rows)
    out.insert(0, 'item_no', range(1, len(out) + 1))
    return out


def qa_score(qa, screened):
    """Agreement of the LLM screening with the researcher's `qa_label`, per topic and overall.

    The QA sample is stratified (same number of reviews per LLM label), so the per-label agreement estimates the
    precision of each label, and the corpus-level accuracy / recall of `app_actionable` are re-weighted by the
    number of reviews the LLM put in each label (`N_app`, `N_oos`)."""
    from sklearn.metrics import cohen_kappa_score
    q = qa.dropna(subset=['qa_label'])
    assert set(q['qa_label']) <= set(LABELS), set(q['qa_label']) - set(LABELS)
    rows = []
    for tid, g in list(q.groupby('topic_id')) + [('all', q)]:
        s = screened if tid == 'all' else screened[screened['topic_id'] == tid]
        n_app, n_oos = int((s['label'] == 'app_actionable').sum()), int((s['label'] == 'out_of_scope').sum())
        a, o = g[g['llm_label'] == 'app_actionable'], g[g['llm_label'] == 'out_of_scope']
        prec_app = float((a['qa_label'] == 'app_actionable').mean())
        prec_oos = float((o['qa_label'] == 'out_of_scope').mean())
        true_app = n_app * prec_app + n_oos * (1 - prec_oos)
        rows.append({'topic_id': tid, 'n_qa': len(g), 'agree': int((g['llm_label'] == g['qa_label']).sum()),
                     'agreement': float((g['llm_label'] == g['qa_label']).mean()),
                     'cohen_kappa': float(cohen_kappa_score(g['llm_label'], g['qa_label'])),
                     'precision_app_actionable': prec_app, 'precision_out_of_scope': prec_oos,
                     'N_app': n_app, 'N_oos': n_oos,
                     'est_accuracy': (n_app * prec_app + n_oos * prec_oos) / (n_app + n_oos),
                     'est_recall_app_actionable': n_app * prec_app / true_app if true_app else float('nan'),
                     'est_true_app_actionable': true_app})
    return pd.DataFrame(rows)
