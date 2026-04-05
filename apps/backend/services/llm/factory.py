from __future__ import annotations

from typing import Literal

from services.llm.base import BaseLLMClient
from services.llm.google_genai import GeminiLLMClient
from services.llm.openai_client import OpenAILLMClient

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
        model: Optional explicit model override.
        api_key: Optional explicit API key override.

    Returns:
        A configured provider client.

    Raises:
        ValueError: If the provider name is unsupported.
    """

    if provider == "gemini":
        return GeminiLLMClient(
            api_key=api_key,
            model=model or "gemini-3-flash-preview",
        )

    if provider == "openai":
        return OpenAILLMClient(
            api_key=api_key,
            model=model or "gpt-5.4-nano",
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
