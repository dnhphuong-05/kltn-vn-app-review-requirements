"""OpenAI client + frozen LLM settings, working both locally and on Kaggle.

Local: reads `.env` at repo root (python-dotenv). Kaggle: reads the secret
OPENAI_API_KEY via UserSecretsClient. Model/snapshot/temperature come from env
(see `.env.example`); defaults mirror research_protocol v1.1.
"""
import os
from dataclasses import dataclass

from .kaggle_paths import is_kaggle, _repo_root

_DEFAULTS = {
    'LLM_MODEL': 'gpt-4o-mini',
    'LLM_SNAPSHOT': 'gpt-4o-mini-2024-07-18',
    'LLM_TEMPERATURE': '0',
}


@dataclass(frozen=True)
class LLMSettings:
    model: str          # model family, for reporting
    snapshot: str       # pinned version — pass THIS to the API
    temperature: float


def _load_env() -> None:
    if is_kaggle():
        return
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo_root(), '.env'))


def get_api_key() -> str:
    _load_env()
    if is_kaggle():
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret('OPENAI_API_KEY')
    else:
        key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError(
            'Thiếu OPENAI_API_KEY. Local: điền vào .env (copy từ .env.example). '
            'Kaggle: Add-ons → Secrets → OPENAI_API_KEY.')
    return key


def get_settings() -> LLMSettings:
    _load_env()
    env = lambda k: os.environ.get(k) or _DEFAULTS[k]
    return LLMSettings(
        model=env('LLM_MODEL'),
        snapshot=env('LLM_SNAPSHOT'),
        temperature=float(env('LLM_TEMPERATURE')),
    )


def get_client():
    """Return (OpenAI client, LLMSettings)."""
    from openai import OpenAI
    return OpenAI(api_key=get_api_key()), get_settings()


# ---------------------------------------------------------------- one-off API check (before GĐ8)
# Reference prices (USD per 1M tokens) published for gpt-4o-mini in 2024. Override with
# LLM_PRICE_INPUT_PER_1M / LLM_PRICE_OUTPUT_PER_1M in .env and confirm on the provider's pricing
# page on the run date (protocol: record pricing at run time).
_REF_PRICE = {'input': 0.15, 'output': 0.60}


def mask(text, secret):
    """Replace a secret (and its tail) in any string before logging."""
    text = str(text)
    if secret:
        text = text.replace(secret, '***')
        if len(secret) > 8:
            text = text.replace(secret[-8:], '***')
    return text


def prices():
    _load_env()
    return {'input': float(os.environ.get('LLM_PRICE_INPUT_PER_1M') or _REF_PRICE['input']),
            'output': float(os.environ.get('LLM_PRICE_OUTPUT_PER_1M') or _REF_PRICE['output']),
            'source': 'env' if os.environ.get('LLM_PRICE_INPUT_PER_1M') else 'giá tham chiếu 2024 — cần xác nhận'}


def check_api(review: str = 'app hay bị văng khi thanh toán') -> dict:
    """One chat completion with the pinned snapshot. Returns a result dict; never raises on API
    errors — they are classified (model_not_found / auth / rate_limit / ...) with the key masked."""
    from openai import (APIConnectionError, AuthenticationError, BadRequestError, NotFoundError,
                        OpenAI, PermissionDeniedError, RateLimitError)
    settings = get_settings()
    key = get_api_key()
    prompt = (f"Phân loại review sau thành đúng một nhãn BUG/FEATURE/UX/OTHER. "
              f"Chỉ trả lời nhãn.\nReview: '{review}'")
    out = {'requested_model': settings.snapshot, 'temperature': settings.temperature, 'ok': False}
    try:
        resp = OpenAI(api_key=key).chat.completions.create(
            model=settings.snapshot, temperature=settings.temperature, max_tokens=5,
            messages=[{'role': 'user', 'content': prompt}])
    except NotFoundError as e:
        out.update(error='model_not_found', detail=mask(e, key),
                   hint=f'Snapshot {settings.snapshot} không còn (đã bị gỡ hoặc tài khoản không có quyền) — cần quyết model khác.')
        return out
    except AuthenticationError as e:
        out.update(error='auth', detail=mask(e, key), hint='API key sai hoặc đã bị thu hồi.')
        return out
    except PermissionDeniedError as e:
        out.update(error='permission', detail=mask(e, key), hint='Key không có quyền dùng model này / dự án.')
        return out
    except RateLimitError as e:
        out.update(error='rate_limit_or_quota', detail=mask(e, key),
                   hint='Bị giới hạn tốc độ hoặc hết hạn mức (quota/billing) — kiểm tra tài khoản.')
        return out
    except BadRequestError as e:
        out.update(error='bad_request', detail=mask(e, key))
        return out
    except APIConnectionError as e:
        out.update(error='connection', detail=mask(e, key), hint='Không kết nối được API (mạng/proxy).')
        return out
    p = prices()
    u = resp.usage
    cost = u.prompt_tokens / 1e6 * p['input'] + u.completion_tokens / 1e6 * p['output']
    out.update(ok=True, response_model=resp.model, snapshot_match=resp.model == settings.snapshot,
               answer=(resp.choices[0].message.content or '').strip(), prompt_tokens=u.prompt_tokens,
               completion_tokens=u.completion_tokens, cost_usd=cost, prices=p,
               system_fingerprint=getattr(resp, 'system_fingerprint', None))
    return out


def estimate_cost(n_reviews: int, input_tokens_per_call: int, output_tokens_per_call: int,
                  calls_per_review: float = 1.0) -> float:
    """USD estimate for classifying n_reviews with a given prompt size."""
    p = prices()
    per_call = input_tokens_per_call / 1e6 * p['input'] + output_tokens_per_call / 1e6 * p['output']
    return n_reviews * calls_per_review * per_call
