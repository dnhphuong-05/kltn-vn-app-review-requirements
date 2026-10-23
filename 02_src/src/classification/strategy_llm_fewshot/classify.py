"""Strategy C — classify reviews with the locked few-shot prompt (gpt-4o-mini snapshot, temperature 0).

Every API call is logged as one JSON line (review_id, tier, attempt, raw output, tokens, latency, errors) — no
review text. Parsing accepts only the tier's labels (plus obvious aliases: BUG/FEATURE/UX → ACTIONABLE and
OTHER/NOT → NOT_ACTIONABLE in Tier 1). An unparseable answer is retried once; if it still fails, Tier 1 falls
back to NOT_ACTIONABLE (= OTHER) and Tier 2 to an invalid label that counts as an error. Retries, API errors
and fallbacks are counted in runtime and cost (protocol: never relabel by hand).
"""
import json
import time

import pandas as pd

from src.classification.strategy_llm_fewshot.prompts import LABELS, schema

API_SEED = 42                     # OpenAI best-effort determinism; LLM is not trained, this is not a model seed
MAX_TOKENS = 20
ALIASES = {1: {'NOT': 'NOT_ACTIONABLE', 'NOT ACTIONABLE': 'NOT_ACTIONABLE', 'NON_ACTIONABLE': 'NOT_ACTIONABLE',
               'OTHER': 'NOT_ACTIONABLE', 'BUG': 'ACTIONABLE', 'FEATURE': 'ACTIONABLE', 'UX': 'ACTIONABLE'},
           2: {}}
FALLBACK = {1: 'NOT_ACTIONABLE', 2: 'UNPARSED'}


def parse(content, tier):
    if content is None:
        return None
    try:
        lab = json.loads(content)['label']
    except (ValueError, TypeError, KeyError):
        lab = content
    lab = str(lab).strip().strip('"\'.` ').upper().replace('-', '_')
    lab = ALIASES[tier].get(lab, lab)
    return lab if lab in LABELS[tier] else None


def _create(client, settings, messages, tier, max_api_retries=4):
    """One completion with backoff on transient API errors. Returns (response, transient_error_messages)."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    errors = []
    for attempt in range(max_api_retries):
        try:
            return client.chat.completions.create(
                model=settings.snapshot, temperature=settings.temperature, seed=API_SEED, max_tokens=MAX_TOKENS,
                messages=messages, response_format=schema(tier)), errors
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            errors.append(f'{type(e).__name__}: {str(e)[:200]}')
            time.sleep(2 ** attempt)
    raise RuntimeError(f'API lỗi {max_api_retries} lần liên tiếp: {errors[-1]}')


def classify(df, prompt, client, settings, log_path, id_col='review_id', text_col='text', progress_every=50, log=print):
    """Returns one row per review: pred_label, pred (class id, -1 if invalid), attempts, fallback, tokens, seconds."""
    tier, names = prompt.tier, LABELS[prompt.tier]
    rows = []
    with open(log_path, 'a', encoding='utf-8') as lf:
        for i, (rid, text) in enumerate(zip(df[id_col], df[text_col].astype(str)), 1):
            msgs = prompt.messages(text)
            label, tok_in, tok_out, cached, n_api_err, fp = None, 0, 0, 0, 0, None
            t0 = time.time()
            for attempt in (1, 2):                          # 1 lần gọi + tối đa 1 lần thử lại khi parse lỗi
                resp, api_errors = _create(client, settings, msgs, tier)
                n_api_err += len(api_errors)
                msg, u = resp.choices[0].message, resp.usage
                c = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
                tok_in, tok_out, cached, fp = tok_in + u.prompt_tokens, tok_out + u.completion_tokens, cached + c, resp.system_fingerprint
                label = parse(msg.content, tier)
                lf.write(json.dumps({'review_id': rid, 'tier': tier, 'attempt': attempt, 'raw': msg.content,
                                     'refusal': getattr(msg, 'refusal', None), 'parsed': label,
                                     'finish_reason': resp.choices[0].finish_reason, 'model': resp.model,
                                     'system_fingerprint': fp, 'prompt_tokens': u.prompt_tokens,
                                     'completion_tokens': u.completion_tokens, 'cached_tokens': c,
                                     'api_errors': api_errors}, ensure_ascii=False) + '\n')
                if label is not None:
                    break
            fallback = label is None
            if fallback:
                label = FALLBACK[tier]
                log(f'[tier{tier}] {rid}: không parse được sau 2 lần → {label}')
            rows.append({'review_id': rid, 'tier': tier, 'pred_label': label,
                         'pred': names.index(label) if label in names else -1, 'attempts': attempt,
                         'fallback': fallback, 'api_errors': n_api_err, 'prompt_tokens': tok_in,
                         'completion_tokens': tok_out, 'cached_tokens': cached, 'seconds': time.time() - t0,
                         'system_fingerprint': fp})
            if progress_every and i % progress_every == 0:
                log(f'[tier{tier}] {i}/{len(df)}')
    return pd.DataFrame(rows)


def results_from_log(log_path, tier):
    """Rebuild classify()'s per-review rows from the call log (no API call). `seconds` is not logged → NaN."""
    calls = pd.DataFrame([json.loads(l) for l in open(log_path, encoding='utf-8')])
    calls = calls[calls['tier'] == tier]
    names, rows = LABELS[tier], []
    for rid, g in calls.groupby('review_id', sort=False):
        label = g['parsed'].dropna().iloc[-1] if g['parsed'].notna().any() else FALLBACK[tier]
        rows.append({'review_id': rid, 'tier': tier, 'pred_label': label,
                     'pred': names.index(label) if label in names else -1, 'attempts': int(g['attempt'].max()),
                     'fallback': not g['parsed'].notna().any(), 'api_errors': int(g['api_errors'].map(len).sum()),
                     'prompt_tokens': int(g['prompt_tokens'].sum()), 'completion_tokens': int(g['completion_tokens'].sum()),
                     'cached_tokens': int(g['cached_tokens'].sum()), 'seconds': float('nan'),
                     'system_fingerprint': g['system_fingerprint'].iloc[-1]})
    return pd.DataFrame(rows)


def cost_summary(res, prices, name):
    """Real token usage and $ cost at the configured prices. Cached input tokens are billed lower by the
    provider; cost here charges them at the full input price (upper bound) and reports the cached count."""
    tin, tout = int(res['prompt_tokens'].sum()), int(res['completion_tokens'].sum())
    usd = tin / 1e6 * prices['input'] + tout / 1e6 * prices['output']
    return {'bước': name, 'n_review': len(res), 'n_lần_gọi': int(res['attempts'].sum()),
            'thử_lại_do_parse': int((res['attempts'] > 1).sum()), 'fallback': int(res['fallback'].sum()),
            'lỗi_API_tạm_thời': int(res['api_errors'].sum()), 'token_vào': tin, 'token_vào_cached': int(res['cached_tokens'].sum()),
            'token_ra': tout, 'chi_phí_usd': usd, 'giây': float(res['seconds'].sum(min_count=1)),   # NaN khi dựng từ log
            'giây_/1000_review': float(res['seconds'].sum(min_count=1)) / max(len(res), 1) * 1000,
            'usd_/1000_review': usd / max(len(res), 1) * 1000}
