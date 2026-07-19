"""Embedding providers for hybrid memory retrieval.

Lexical token recall works well for exact overlap and simple
paraphrases, but it misses stemming, synonyms, tense changes, and
semantic paraphrases without shared tokens. Embedding providers add a
dense retrieval path that complements lexical recall when credentials
are configured.

This module declares the :class:`EmbeddingProvider` protocol and
two concrete implementations:

- :class:`OpenAIEmbeddingProvider` — calls OpenAI's
  ``text-embedding-3-large`` via the existing ``openai`` client.
  This is the default when ``OPENAI_API_KEY`` is configured.

- :class:`NullEmbeddingProvider` — a no-op that always returns
  ``None`` for embedding calls. Used when no API key is available,
  when memory mode is INCOGNITO, or in tests that don't want to
  mock a real embedding provider. Nodes that try to compute an
  embedding via a ``NullEmbeddingProvider`` get back ``None``,
  which is the signal to fall back to token-recall.

Design decisions:

1. **Protocol, not inheritance.** The :class:`EmbeddingProvider`
   protocol has exactly one method (``aembed``) plus two metadata
   properties (``dimension``, ``model_name``). Concrete classes
   implement the protocol structurally rather than inheriting from
   an abstract base class. This matches the
   :class:`agent.memory.store.MemoryStore` pattern and keeps the
   provider surface minimal.

2. **Metadata is part of the protocol.** ``dimension`` and
   ``model_name`` are required because the store writes the model
   name alongside each record's embedding in the ``embedding_model``
   column. Model migrations can compare the stored model name against
   the current provider's model name and skip retrieval on mismatched
   cohorts until a re-embed sweep runs.

3. **Batch support.** ``aembed`` takes a list of texts and returns
   a list of embeddings (one per input). This matches the OpenAI
   batch embedding API and lets callers efficiently
   embed multiple records in a single network round-trip when
   they need to.

4. **Embeddings are plain ``list[float]``, not numpy arrays.** Each
   concrete store owns its representation. Keeping plain lists at the protocol
   boundary avoids a numpy dependency.
   Shared cosine scoring lives in :mod:`agent.memory.retrieval.ranking`.

5. **Failures degrade to None, not exceptions.** ``aembed`` catches
   provider errors internally and returns a list of ``None`` values
   matching the input length. The caller then decides whether to
   skip the embedding write (if the failure is in a write path)
   or fall back to token-recall (if the failure is in a query
   path). This matches the "never propagate" contract used by the
   semantic and procedural extractor nodes.

See :mod:`agent.memory.retrieval.ranking` for the RRF hybrid fusion helper
that combines embedding-similarity and token-recall rankings.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# OpenAI's text-embedding-3-large is the default embedding model.
# It produces 3072-dimensional embeddings and is the current
# high-capability OpenAI embedding option for retrieval use cases.
# Switching models later requires (a) updating this constant or
# injecting the model name at provider construction time, and (b)
# running a re-embed sweep for records that still have the old
# model stored in ``embedding_model``.
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 3072


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The async embedding interface for memory retrieval.

    Concrete implementations:
    - :class:`OpenAIEmbeddingProvider` — OpenAI text-embedding-3-large
    - :class:`NullEmbeddingProvider` — the no-embedding fallback

    All nodes that need embeddings should type their dependencies
    against this protocol so the runtime can swap implementations
    without touching node code. Same pattern as
    :class:`agent.memory.store.MemoryStore`.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model this provider uses.

        Stored alongside each record's embedding in the
        ``embedding_model`` column so the store can detect and
        handle model-migration cohorts. Should be stable across
        calls for a given provider instance — don't return random
        strings or timestamps.

        Returns:
            str: Stable embedding model identifier.
        """
        ...

    @property
    def dimension(self) -> int:
        """The fixed dimensionality of embeddings this provider returns.

        Used by the store to validate that stored embeddings match
        the current provider's output dimensionality before running
        cosine similarity. Mismatched dimensions are a signal to
        skip the record in the embedding path (the token-recall
        fallback still applies).

        Returns:
            int: Expected embedding vector dimension.
        """
        ...

    async def awarmup(self) -> None:
        """Initialize provider resources ahead of first use.

        Returns:
            None: Warms the provider for later embed calls.
        """
        ...

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float] | None]:
        """Embed a batch of texts.

        Args:
            texts (list[str]): Input texts to embed.
            task_type (str): Embedding task hint such as query vs document mode.

        Returns:
            list[list[float] | None]: One embedding result per input text.
        """
        ...


class NullEmbeddingProvider:
    """A no-op embedding provider used when no real provider is configured.

    Returns ``None`` for every embedding request. This is the
    default runtime wiring when:

    - No OpenAI API key is configured (deterministic mode, CI, some test setups)
    - Memory mode is INCOGNITO (no long-term writes means no
      embeddings to store either)
    - A test wants to exercise the token-recall fallback path
      without spinning up a real provider

    Downstream callers that see a ``None`` embedding fall back to
    token-recall for that specific operation, so the system
    degrades gracefully rather than breaking.
    """

    model_name: str = "null"
    dimension: int = 0

    async def awarmup(self) -> None:
        """Warm the null provider.

        Returns:
            None: No-op for the null provider.
        """
        return

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",  # noqa: ARG002 — contract
    ) -> list[list[float] | None]:
        """Return null embeddings for each input text.

        Args:
            texts (list[str]): Input texts to embed.
            task_type (str): Unused task hint kept for protocol compatibility.

        Returns:
            list[list[float] | None]: ``None`` for each input text.
        """

        return [None] * len(texts)


class OpenAIEmbeddingProvider:
    """OpenAI implementation of :class:`EmbeddingProvider`.

    Uses the existing ``openai`` SDK already present in the backend for
    text generation. Calls ``client.embeddings.create(...)`` and returns
    vectors in input order. This is the default embedding provider when
    ``OPENAI_API_KEY`` is configured.

    Construction requires an OpenAI API key, either passed explicitly or
    resolved from ``OPENAI_API_KEY``. If no key is available, prefer
    :class:`NullEmbeddingProvider` at the runtime wiring layer — don't
    catch the ValueError from construction and silently fall back,
    because that would hide configuration mistakes. The runtime's
    :func:`agent.memory.providers.embeddings.create_configured_embedding_provider`
    helper handles the key-missing case explicitly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        dimension: int = DEFAULT_OPENAI_EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize an OpenAI-backed embedding provider.

        Args:
            api_key: Optional explicit OpenAI API key. Falls back to
                ``OPENAI_API_KEY``.
            model: Embedding model identifier. Defaults to
                ``text-embedding-3-large``.
            dimension: Expected output dimensionality for the configured
                model. Used by the store to validate stored-vs-current
                matches.

        Raises:
            ValueError: If no OpenAI API key can be resolved.
        """

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI embedding provider: no API key. Set OPENAI_API_KEY."
            )

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=resolved_key)
        self._model = model
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI embedding model name.

        Returns:
            str: Embedding model identifier.
        """

        return self._model

    @property
    def dimension(self) -> int:
        """Return the configured OpenAI embedding dimensionality.

        Returns:
            int: Expected embedding vector dimension.
        """

        return self._dimension

    async def awarmup(self) -> None:
        """Warm the OpenAI provider.

        Returns:
            None: Issues a lightweight embed call.
        """
        await self.aembed([" "], task_type="RETRIEVAL_QUERY")

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",  # noqa: ARG002 — OpenAI ignores task hints
    ) -> list[list[float] | None]:
        """Embed a batch of texts with OpenAI.

        Args:
            texts (list[str]): Input texts to embed.
            task_type (str): Unused task hint kept for protocol compatibility.

        Returns:
            list[list[float] | None]: One embedding result per input text.
        """

        if not texts:
            return []

        sanitized = [t if t else " " for t in texts]

        try:
            request: dict[str, Any] = dict(
                model=self._model,
                input=sanitized,
            )
            if self._model.startswith("text-embedding-3"):
                request["dimensions"] = self._dimension
            response = await self._client.embeddings.create(**request)
        except Exception:
            logger.warning(
                "OpenAIEmbeddingProvider: embeddings.create failed for batch of %d; "
                "returning all-None. Caller should fall back to token-recall.",
                len(sanitized),
                exc_info=True,
            )
            return [None] * len(texts)

        response_data = getattr(response, "data", None) or []
        if len(response_data) != len(sanitized):
            logger.warning(
                "OpenAIEmbeddingProvider: response length %d != input length %d. "
                "Returning all-None for the batch.",
                len(response_data),
                len(sanitized),
            )
            return [None] * len(texts)

        embeddings_out: list[list[float] | None] = []
        for item in response_data:
            values = getattr(item, "embedding", None)
            if values is None:
                embeddings_out.append(None)
                continue
            if len(values) != self._dimension:
                logger.warning(
                    "OpenAIEmbeddingProvider: got embedding of dim %d, "
                    "expected %d. Dropping this entry.",
                    len(values),
                    self._dimension,
                )
                embeddings_out.append(None)
                continue
            embeddings_out.append([float(v) for v in values])

        return embeddings_out


def create_configured_embedding_provider() -> EmbeddingProvider:
    """Build the configured embedding provider for the current environment.

    Returns:
        EmbeddingProvider: OpenAI or null provider based on env config.
    """

    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingProvider(
                model=os.getenv(
                    "OPENAI_EMBEDDING_MODEL",
                    DEFAULT_OPENAI_EMBEDDING_MODEL,
                ),
                dimension=int(
                    os.getenv(
                        "OPENAI_EMBEDDING_DIMENSION",
                        str(DEFAULT_OPENAI_EMBEDDING_DIMENSION),
                    )
                ),
            )
        except Exception:
            logger.warning(
                "create_configured_embedding_provider: OpenAIEmbeddingProvider "
                "construction failed; falling back to NullEmbeddingProvider. "
                "Retrieval may degrade to token-recall only.",
                exc_info=True,
            )

    return NullEmbeddingProvider()
