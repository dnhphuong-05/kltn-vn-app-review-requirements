"""RQ2 (GĐ12) — topic names with the LLM (gpt-4o-mini snapshot, temperature 0; research_protocol: the same model as
classification). A name is a label for the BA only: it never changes topic membership.

Input per topic: the c-TF-IDF keywords and the reviews closest to the topic centroid. Every call is logged as one JSON
line (topic_id, raw output, tokens, latency) — no review text.
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
SCHEMA = {'type': 'json_schema',
          'json_schema': {'name': 'topic_name', 'strict': True,
                          'schema': {'type': 'object', 'additionalProperties': False, 'required': ['name_vi', 'name_en'],
                                     'properties': {'name_vi': {'type': 'string'}, 'name_en': {'type': 'string'}}}}}


def load_prompt(version='v1'):
    """(system text, prompt_hash). The hash covers the system text and the output schema."""
    system = open(os.path.join(PROMPT_DIR, f'topic_naming_system_{version}.txt'), encoding='utf-8').read().strip()
    payload = json.dumps({'system': system, 'schema': SCHEMA}, ensure_ascii=False, sort_keys=True)
    return system, hashlib.sha256(payload.encode('utf-8')).hexdigest()


def registered(prompt_hash):
    return prompt_hash in open(REGISTRY, encoding='utf-8').read()


def user_message(keywords, examples):
    lines = ['Từ khóa: ' + ', '.join(keywords), '', 'Review tiêu biểu:']
    lines += [f'{i}. {t}' for i, t in enumerate(examples, 1)]
    return '\n'.join(lines)


def name_topics(items, system, client, settings, log_path, log=print):
    """items: [{'topic_id', 'keywords', 'examples'}]. Returns one row per topic with names, tokens and seconds."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    rows = []
    with open(log_path, 'a', encoding='utf-8') as lf:
        for it in items:
            msgs = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user_message(it['keywords'], it['examples'])}]
            t0, resp, errors = time.time(), None, []
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(model=settings.snapshot, temperature=settings.temperature,
                                                          seed=API_SEED, max_tokens=MAX_TOKENS, messages=msgs,
                                                          response_format=SCHEMA)
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
                name_vi = name_en = ''
            u = resp.usage
            row = {'topic_id': it['topic_id'], 'name_vi': name_vi, 'name_en': name_en, 'parsed': bool(name_vi),
                   'prompt_tokens': u.prompt_tokens, 'completion_tokens': u.completion_tokens,
                   'seconds': time.time() - t0, 'api_errors': len(errors), 'model': resp.model,
                   'system_fingerprint': resp.system_fingerprint}
            lf.write(json.dumps({**row, 'raw': content}, ensure_ascii=False) + '\n')
            rows.append(row)
    return pd.DataFrame(rows)


def cost(df, prices):
    usd = (df['prompt_tokens'].sum() * prices['input'] + df['completion_tokens'].sum() * prices['output']) / 1e6
    return {'n_topic': len(df), 'token_vào': int(df['prompt_tokens'].sum()), 'token_ra': int(df['completion_tokens'].sum()),
            'chi_phí_usd': float(usd), 'giây': float(df['seconds'].sum()), 'không_parse_được': int((~df['parsed']).sum())}
