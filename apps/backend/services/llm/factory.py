"""Factory helpers for constructing configured LLM provider clients."""

from __future__ import annotations

from typing import Literal

from services.llm.base import BaseLLMClient
from services.llm.google_genai import DEFAULT_GEMINI_MODEL, GeminiLLMClient
from services.llm.openai_client import DEFAULT_OPENAI_MODEL, OpenAILLMClient

LLMProvider = Literal["gemini", "openai"]


def create_llm_client(
    *,
    provider: LLMProvider,
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLLMClient:
    """Create a provider-specific LLM client.

    Args:
        provider: Normalized provider name.
        model: Optional explicit model override. When ``None``, uses the
            provider's ``DEFAULT_*_MODEL`` constant as the single source
            of truth for model defaults.
        api_key: Optional explicit API key override.

    Returns:
        A configured provider client.

    Raises:
        ValueError: If the provider name is unsupported.
    """

    if provider == "gemini":
        return GeminiLLMClient(
            api_key=api_key,
            model=model or DEFAULT_GEMINI_MODEL,
        )

    if provider == "openai":
        return OpenAILLMClient(
            api_key=api_key,
            model=model or DEFAULT_OPENAI_MODEL,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
