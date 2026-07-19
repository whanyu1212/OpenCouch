"""Tests for provider adapter request wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm.factory import create_llm_client
from llm.openai_client import DEFAULT_OPENAI_MODEL
from llm.openai_client import OpenAILLMClient


class _FakeResponses:
    """Capture Responses API calls without network access."""

    def __init__(self) -> None:
        """Initialize the fake response endpoint.

        Returns:
            None.
        """

        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Capture a create call and return text.

        Args:
            **kwargs: Responses API request payload.

        Returns:
            Response-like object exposing ``output_text``.
        """

        self.calls.append(kwargs)
        return SimpleNamespace(output_text="grounded response")


class _FakeAsyncOpenAI:
    """Minimal AsyncOpenAI replacement for adapter tests."""

    instances: list["_FakeAsyncOpenAI"] = []

    def __init__(self, *, api_key: str) -> None:
        """Initialize the fake client.

        Args:
            api_key: API key passed by the adapter.

        Returns:
            None.
        """

        self.api_key = api_key
        self.responses = _FakeResponses()
        self.instances.append(self)


class _FakeProviderClient:
    """Capture provider factory construction arguments."""

    calls: list[dict[str, str | None]] = []

    def __init__(self, *, api_key: str | None = None, model: str) -> None:
        """Initialize the fake provider client.

        Args:
            api_key: API key passed by the factory.
            model: Model selected by the factory.

        Returns:
            None.
        """

        self.api_key = api_key
        self.model = model
        self.calls.append({"api_key": api_key, "model": model})


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    """Reset fake provider state before each test.

    Returns:
        None.
    """

    _FakeAsyncOpenAI.instances.clear()
    _FakeProviderClient.calls.clear()


@pytest.mark.asyncio
async def test_openai_generate_text_omits_tools_without_search(monkeypatch) -> None:
    """Plain text generation should not send an empty tools list."""

    monkeypatch.setattr("llm.openai_client.AsyncOpenAI", _FakeAsyncOpenAI)

    client = OpenAILLMClient(api_key="test-key", model="test-model")
    text = await client.generate_text(prompt="hello", use_search=False)

    assert text == "grounded response"
    call = _FakeAsyncOpenAI.instances[0].responses.calls[0]
    assert call["model"] == "test-model"
    assert "tools" not in call


@pytest.mark.asyncio
async def test_openai_generate_text_uses_responses_web_search_tool(monkeypatch) -> None:
    """Search-enabled text generation should use the documented tool type."""

    monkeypatch.setattr("llm.openai_client.AsyncOpenAI", _FakeAsyncOpenAI)

    client = OpenAILLMClient(api_key="test-key", model="test-model")
    await client.generate_text(prompt="find current info", use_search=True)

    call = _FakeAsyncOpenAI.instances[0].responses.calls[0]
    assert call["tools"] == [{"type": "web_search_preview"}]


def test_create_llm_client_routes_openai_with_default_model(monkeypatch) -> None:
    """Factory should route OpenAI clients with the OpenAI default model."""

    monkeypatch.setattr("llm.factory.OpenAILLMClient", _FakeProviderClient)

    client = create_llm_client(provider="openai", api_key="openai-key")

    assert isinstance(client, _FakeProviderClient)
    assert _FakeProviderClient.calls == [
        {"api_key": "openai-key", "model": DEFAULT_OPENAI_MODEL}
    ]


def test_create_llm_client_rejects_unknown_provider() -> None:
    """Factory should fail closed for unsupported providers."""

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(provider="anthropic")  # type: ignore[arg-type]
