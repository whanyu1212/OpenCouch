"""Tests for embedding provider runtime selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.memory.providers.embeddings import (
    NullEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_configured_embedding_provider,
)


def test_missing_openai_key_selects_null_embedding_provider(monkeypatch) -> None:
    """Embedding retrieval falls back to token-only mode without OpenAI keys."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = create_configured_embedding_provider()

    assert isinstance(provider, NullEmbeddingProvider)


@pytest.mark.asyncio
async def test_openai_provider_requests_configured_dimensions() -> None:
    """Configured dimensions are sent to OpenAI instead of only validated later."""

    calls: list[dict[str, object]] = []

    class _Embeddings:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0])],
            )

    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._client = SimpleNamespace(embeddings=_Embeddings())  # noqa: SLF001
    provider._model = "text-embedding-3-large"  # noqa: SLF001
    provider._dimension = 2  # noqa: SLF001

    result = await provider.aembed(["hello"])

    assert result == [[1.0, 0.0]]
    assert calls == [
        {
            "model": "text-embedding-3-large",
            "input": ["hello"],
            "dimensions": 2,
        }
    ]


@pytest.mark.asyncio
async def test_openai_provider_omits_dimensions_for_legacy_model() -> None:
    """Legacy OpenAI embedding models reject the dimensions parameter."""

    calls: list[dict[str, object]] = []

    class _Embeddings:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0])],
            )

    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._client = SimpleNamespace(embeddings=_Embeddings())  # noqa: SLF001
    provider._model = "text-embedding-ada-002"  # noqa: SLF001
    provider._dimension = 2  # noqa: SLF001

    result = await provider.aembed(["hello"])

    assert result == [[1.0, 0.0]]
    assert calls == [
        {
            "model": "text-embedding-ada-002",
            "input": ["hello"],
        }
    ]
