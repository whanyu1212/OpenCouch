"""Tests for embedding provider runtime selection."""

from __future__ import annotations

from agent.memory.providers.embeddings import (
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)


def test_missing_openai_key_selects_null_embedding_provider(monkeypatch) -> None:
    """Embedding retrieval falls back to token-only mode without OpenAI keys."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = create_configured_embedding_provider()

    assert isinstance(provider, NullEmbeddingProvider)
