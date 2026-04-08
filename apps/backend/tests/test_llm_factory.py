import pytest

from core.config import create_configured_llm_client, get_settings
from services.llm.factory import create_llm_client
from services.llm.google_genai import GeminiLLMClient
from services.llm.openai_client import OpenAILLMClient


def test_create_llm_client_returns_gemini_client() -> None:
    """Factory should build a Gemini client for the Gemini provider."""

    client = create_llm_client(provider="gemini", api_key="test-key")
    assert isinstance(client, GeminiLLMClient)


def test_create_llm_client_returns_openai_client() -> None:
    """Factory should build an OpenAI client for the OpenAI provider."""

    client = create_llm_client(provider="openai", api_key="test-key")
    assert isinstance(client, OpenAILLMClient)


def test_get_settings_uses_env_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should honor the configured environment provider.

    Args:
        monkeypatch: Pytest monkeypatch fixture for environment overrides.
    """

    monkeypatch.setenv("LLM_PROVIDER", "openai")

    settings = get_settings()
    assert settings.llm_provider == "openai"


def test_get_settings_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should default to OpenAI when no provider is configured.

    Args:
        monkeypatch: Pytest monkeypatch fixture for environment overrides.
    """

    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    settings = get_settings()
    assert settings.llm_provider == "openai"


def test_create_configured_llm_client_uses_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured client creation should use the resolved settings.

    Args:
        monkeypatch: Pytest monkeypatch fixture for environment overrides.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    client = create_configured_llm_client()
    assert isinstance(client, OpenAILLMClient)


def test_create_llm_client_rejects_unknown_provider() -> None:
    """Factory should reject unsupported provider names."""

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(provider="unknown")  # type: ignore[arg-type]
