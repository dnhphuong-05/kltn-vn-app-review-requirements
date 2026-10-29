"""RQ2 (GĐ12) — topic names with the LLM (gpt-4o-mini snapshot, temperature 0; research_protocol: the same model as
classification). A name is a label for the BA only: it never changes topic membership.

Input per topic: the c-TF-IDF keywords, the reviews closest to the topic centroid and (v2) a few random members. v2 also
returns two flags for the BA — `specific` (one concrete issue vs. a grab-bag) and `scope` (app / outside the app / mixed);
they never remove a topic. Every call is logged as one JSON line (topic_id, raw output, tokens, latency) — no review text.
"""
import hashlib
import json
import os
import time

import pandas as pd

from src.utils.kaggle_paths import _repo_root

PROMPT_DIR = os.path.join(_repo_root(), '04_prompts', 'topic_naming')
REGISTRY = os.path.join(_repo_root(), '04_prompts', 'prompts_registry.md')
API_SEED = 42
MAX_TOKENS = 80
SCOPES = ['app', 'ngoài_app', 'lẫn_lộn']


def _schema(fields):
    return {'type': 'json_schema',
            'json_schema': {'name': 'topic_name', 'strict': True,
                            'schema': {'type': 'object', 'additionalProperties': False, 'required': list(fields),
                                       'properties': fields}}}


SCHEMAS = {'v1': _schema({'name_vi': {'type': 'string'}, 'name_en': {'type': 'string'}}),
           'v2': _schema({'name_vi': {'type': 'string'}, 'name_en': {'type': 'string'}, 'specific': {'type': 'boolean'},
                          'scope': {'type': 'string', 'enum': SCOPES}}),
           # v3: the yes/no `specific` of v2 came back false for every topic (even one where 10/10 reviews named the same
           # issue); the LLM now counts reviews sharing the main issue and `specific` is derived here (≥ SPECIFIC_MIN).
           'v3': _schema({'n_same_issue': {'type': 'integer', 'enum': list(range(11))}, 'name_vi': {'type': 'string'},
                          'name_en': {'type': 'string'}, 'scope': {'type': 'string', 'enum': SCOPES}})}
# v4: one call for ALL topics so that topics sharing a theme get names that say how they differ (v1–v3 named each topic
# alone and gave six advertising topics the same name). The v2/v3 specificity flag is dropped — v2 was too strict, v3 too
# lenient; topic specificity is judged by people in the protocol's manual topic purity check.
_TOPIC_V4 = {'type': 'object', 'additionalProperties': False, 'required': ['topic_id', 'name_vi', 'name_en', 'scope'],
             'properties': {'topic_id': {'type': 'integer'}, 'name_vi': {'type': 'string'}, 'name_en': {'type': 'string'},
                            'scope': {'type': 'string', 'enum': SCOPES}}}
SCHEMAS['v4'] = _schema({'topics': {'type': 'array', 'items': _TOPIC_V4}})
JOINT = {'v4'}
SCHEMA = SCHEMAS['v1']
SPECIFIC_MIN = 6                  # v3 only: of the 10 reviews shown (5 nearest the centroid + 5 random)


def load_prompt(version='v2'):
    """(system text, prompt_hash, schema). The hash covers the system text and the output schema."""
    system = open(os.path.join(PROMPT_DIR, f'topic_naming_system_{version}.txt'), encoding='utf-8').read().strip()
    payload = json.dumps({'system': system, 'schema': SCHEMAS[version]}, ensure_ascii=False, sort_keys=True)
    return system, hashlib.sha256(payload.encode('utf-8')).hexdigest(), SCHEMAS[version]


def registered(prompt_hash):
    return prompt_hash in open(REGISTRY, encoding='utf-8').read()


def user_message(keywords, examples, random_examples=()):
    lines = ['Từ khóa: ' + ', '.join(keywords), '', 'Review gần tâm cụm nhất:']
    lines += [f'{i}. {t}' for i, t in enumerate(examples, 1)]
    if random_examples:
        lines += ['', 'Review lấy ngẫu nhiên trong cụm:']
        lines += [f'{i}. {t}' for i, t in enumerate(random_examples, 1)]
    return '\n'.join(lines)


def name_topics(items, system, client, settings, log_path, schema=SCHEMA, log=print):
    """items: [{'topic_id', 'keywords', 'examples', 'random_examples'?}]. One row per topic: names, flags (v2), tokens,
    seconds."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    rows = []
    with open(log_path, 'a', encoding='utf-8') as lf:
        for it in items:
            msgs = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user_message(it['keywords'], it['examples'], it.get('random_examples', ()))}]
            t0, resp, errors = time.time(), None, []
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(model=settings.snapshot, temperature=settings.temperature,
                                                          seed=API_SEED, max_tokens=MAX_TOKENS, messages=msgs,
                                                          response_format=schema)
                    break
                except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                    errors.append(f'{type(e).__name__}: {str(e)[:200]}')
                    time.sleep(2 ** attempt)
            if resp is None:
                raise RuntimeError(f"Topic {it['topic_id']}: API lỗi 4 lần liên tiếp: {errors[-1]}")
            content = resp.choices[0].message.content
            try:
                out = json.loads(content)
                name_vi, name_en = out['name_vi'].strip(), out['name_en'].strip()
            except (ValueError, KeyError, AttributeError):
                out, name_vi, name_en = {}, '', ''
            u = resp.usage
            n_same = out.get('n_same_issue')
            specific = (n_same >= SPECIFIC_MIN) if n_same is not None else out.get('specific')
            row = {'topic_id': int(it['topic_id']), 'name_vi': name_vi, 'name_en': name_en, 'n_same_issue': n_same,
                   'specific': specific, 'scope': out.get('scope'), 'parsed': bool(name_vi),
                   'prompt_tokens': u.prompt_tokens, 'completion_tokens': u.completion_tokens,
                   'seconds': time.time() - t0, 'api_errors': len(errors), 'model': resp.model,
                   'system_fingerprint': resp.system_fingerprint}
            lf.write(json.dumps({**row, 'raw': content}, ensure_ascii=False) + '\n')
            rows.append(row)
    return pd.DataFrame(rows)


def joint_message(items):
    blocks = []
    for it in items:
        lines = [f"### Cụm {int(it['topic_id'])} ({it['size']} review)", 'Từ khóa: ' + ', '.join(it['keywords']),
                 'Review gần tâm cụm nhất:'] + [f'{i}. {t}' for i, t in enumerate(it['examples'], 1)]
        lines += ['Review lấy ngẫu nhiên trong cụm:'] + [f'{i}. {t}' for i, t in enumerate(it.get('random_examples', ()), 1)]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def name_topics_joint(items, system, client, settings, log_path, schema, max_tokens=4000):
    """All topics in one call (v4). Raises if the answer does not cover every topic exactly once (after one retry)."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    want = sorted(int(it['topic_id']) for it in items)
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': joint_message(items)}]
    with open(log_path, 'a', encoding='utf-8') as lf:
        for attempt in (1, 2):
            t0, resp, errors = time.time(), None, []
            for k in range(4):
                try:
                    resp = client.chat.completions.create(model=settings.snapshot, temperature=settings.temperature,
                                                          seed=API_SEED, max_tokens=max_tokens, messages=msgs,
                                                          response_format=schema)
                    break
                except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                    errors.append(f'{type(e).__name__}: {str(e)[:200]}')
                    time.sleep(2 ** k)
            if resp is None:
                raise RuntimeError(f'API lỗi 4 lần liên tiếp: {errors[-1]}')
            content, u, secs = resp.choices[0].message.content, resp.usage, time.time() - t0
            lf.write(json.dumps({'call': 'joint', 'attempt': attempt, 'prompt_tokens': u.prompt_tokens,
                                 'completion_tokens': u.completion_tokens, 'seconds': secs, 'model': resp.model,
                                 'system_fingerprint': resp.system_fingerprint, 'raw': content}, ensure_ascii=False) + '\n')
            try:
                got = json.loads(content)['topics']
            except (ValueError, KeyError, TypeError):
                got = []
            if sorted(int(g['topic_id']) for g in got) == want:
                df = pd.DataFrame(got)
                df['topic_id'] = df['topic_id'].astype(int)
                share = 1 / len(df)                    # one call: tokens and time are spread evenly for cost()
                df['prompt_tokens'], df['completion_tokens'] = u.prompt_tokens * share, u.completion_tokens * share
                df['seconds'], df['parsed'], df['attempts'] = secs * share, True, attempt
                df['model'], df['system_fingerprint'] = resp.model, resp.system_fingerprint
                return df.sort_values('topic_id').reset_index(drop=True)
    raise RuntimeError('LLM không trả đủ đúng một tên cho mỗi cụm sau 2 lần gọi')


def cost(df, prices):
    usd = (df['prompt_tokens'].sum() * prices['input'] + df['completion_tokens'].sum() * prices['output']) / 1e6
    return {'n_topic': len(df), 'token_vào': int(df['prompt_tokens'].sum()), 'token_ra': int(df['completion_tokens'].sum()),
            'chi_phí_usd': float(usd), 'giây': float(df['seconds'].sum()), 'không_parse_được': int((~df['parsed']).sum())}
