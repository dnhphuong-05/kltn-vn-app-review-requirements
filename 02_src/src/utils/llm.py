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
