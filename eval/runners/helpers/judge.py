"""Reusable provider-backed judge helpers for eval runners."""

from __future__ import annotations

import os
from typing import Literal

from config import load_runtime_env
from llm.base import BaseLLMClient
from llm.factory import LLMProvider, create_llm_client

ProviderName = Literal["openai"]


def clear_empty_provider_env_vars() -> None:
    """Treat empty provider API-key env vars as unset before loading dotenv files."""

    for key in ("OPENAI_API_KEY",):
        if os.getenv(key) == "":
            os.environ.pop(key, None)


def provider_as_literal(provider: str) -> LLMProvider:
    """Return the provider name as the factory's typed literal."""

    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}")
    return "openai"


def make_judge_client(
    *,
    provider: ProviderName,
    model: str | None,
) -> BaseLLMClient:
    """Build a judge LLM client using the backend runtime environment loading."""

    clear_empty_provider_env_vars()
    load_runtime_env()
    return create_llm_client(provider=provider_as_literal(provider), model=model)


__all__ = [
    "ProviderName",
    "clear_empty_provider_env_vars",
    "make_judge_client",
    "provider_as_literal",
]
