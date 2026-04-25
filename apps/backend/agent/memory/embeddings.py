"""Embedding providers for hybrid memory retrieval.

Lexical token recall works well for exact overlap and simple
paraphrases, but it misses stemming, synonyms, tense changes, and
semantic paraphrases without shared tokens. Embedding providers add a
dense retrieval path that complements lexical recall when credentials
are configured.

This module declares the :class:`EmbeddingProvider` protocol and
three concrete implementations:

- :class:`OpenAIEmbeddingProvider` — calls OpenAI's
  ``text-embedding-3-large`` via the existing ``openai`` client.
  This is the default when ``OPENAI_API_KEY`` is configured.

- :class:`GeminiEmbeddingProvider` — calls Google's
  ``gemini-embedding-001`` via the existing ``google.genai`` client.
  This is the fallback real provider when only ``GEMINI_API_KEY``
  or ``GOOGLE_API_KEY`` is configured.

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
   a list of embeddings (one per input). This matches the Gemini
   and OpenAI batch embedding APIs and lets callers efficiently
   embed multiple records in a single network round-trip when
   they need to.

4. **Embeddings are plain ``list[float]``, not numpy arrays.** The
   store serializes embeddings to a BLOB via
   ``struct.pack``/``unpack`` (see ``sqlite_store.py``). Keeping
   the protocol type as plain lists means callers don't need
   numpy at the boundary. numpy only enters the picture inside
   the retrieval scoring helper in ``retrieval.py``, and even
   there it's optional (the cosine similarity can run on pure
   Python floats).

5. **Failures degrade to None, not exceptions.** ``aembed`` catches
   provider errors internally and returns a list of ``None`` values
   matching the input length. The caller then decides whether to
   skip the embedding write (if the failure is in a write path)
   or fall back to token-recall (if the failure is in a query
   path). This matches the "never propagate" contract used by the
   semantic and procedural extractor nodes.

See :mod:`agent.memory.retrieval` for the RRF hybrid fusion helper
that combines embedding-similarity and token-recall rankings.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, cast, runtime_checkable

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

# Gemini's gemini-embedding-001 remains supported as a fallback
# provider when OpenAI credentials are unavailable.
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_EMBEDDING_DIMENSION = 3072


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The async embedding interface for memory retrieval.

    Concrete implementations:
    - :class:`OpenAIEmbeddingProvider` — OpenAI text-embedding-3-large
    - :class:`GeminiEmbeddingProvider` — Google gemini-embedding-001
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

    - No Gemini or OpenAI API key is configured (deterministic mode,
      CI, some test setups)
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
    :func:`agent.memory.embeddings.create_configured_embedding_provider`
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
            response = await self._client.embeddings.create(
                model=self._model,
                input=sanitized,
            )
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


class GeminiEmbeddingProvider:
    """Google Gen AI implementation of :class:`EmbeddingProvider`.

    Uses the existing ``google.genai`` SDK that the project already
    depends on for the chat/structured-output clients. Calls
    ``client.aio.models.embed_content(...)`` and pulls the vector
    out of the response's ``embeddings[i].values`` field.

    Construction requires a Gemini API key, either passed explicitly
    or resolved from ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``. If no
    key is available, prefer :class:`NullEmbeddingProvider` at the
    runtime wiring layer — don't catch the ValueError from
    construction and silently fall back, because that would hide
    configuration mistakes. The runtime's
    :func:`agent.memory.embeddings.create_configured_embedding_provider`
    helper handles the key-missing case explicitly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        dimension: int = DEFAULT_GEMINI_EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize a Gemini-backed embedding provider.

        Args:
            api_key: Optional explicit Gemini API key. Falls back to
                ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env vars.
            model: Embedding model identifier. Defaults to
                ``gemini-embedding-001``. Changing this requires a
                re-embed sweep for records already stored with the
                old model, since cosine similarity across model
                cohorts is not meaningful.
            dimension: The model's output dimensionality. Used by
                the store to validate stored-vs-current matches.
                Defaults to 3072 (the current
                ``gemini-embedding-001`` setting used here).

        Raises:
            ValueError: if no Gemini API key can be resolved.
        """

        resolved_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Gemini embedding provider: no API key. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        # Import lazily so this module can load when google-genai is
        # unavailable or unused.
        from google import genai

        self._client = genai.Client(api_key=resolved_key)
        self._model = model
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Return the configured Gemini embedding model name.

        Returns:
            str: Embedding model identifier.
        """

        return self._model

    @property
    def dimension(self) -> int:
        """Return the configured Gemini embedding dimensionality.

        Returns:
            int: Expected embedding vector dimension.
        """

        return self._dimension

    async def awarmup(self) -> None:
        """Warm the Gemini provider.

        Returns:
            None: Issues a lightweight embed call.
        """
        await self.aembed([" "], task_type="RETRIEVAL_QUERY")

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float] | None]:
        """Embed a batch of texts with Gemini.

        Args:
            texts (list[str]): Input texts to embed.
            task_type (str): Gemini embedding task type.

        Returns:
            list[list[float] | None]: One embedding result per input text.
        """

        if not texts:
            return []

        # Guard against empty strings which Gemini rejects with a
        # 400. Replace with a single space which embeds to a
        # near-zero vector — still meaningless, but doesn't fail
        # the batch. Callers should avoid passing empty strings
        # in the first place.
        sanitized = [t if t else " " for t in texts]

        from google.genai import types

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=cast(Any, sanitized),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                ),
            )
        except Exception:
            logger.warning(
                "GeminiEmbeddingProvider: embed_content failed for batch of %d; "
                "returning all-None. Caller should fall back to token-recall.",
                len(sanitized),
                exc_info=True,
            )
            return [None] * len(texts)

        # The response's ``.embeddings`` field is a list of
        # ContentEmbedding objects, each with a ``.values`` field
        # that is a list of floats. Match order is preserved:
        # embeddings[i] corresponds to contents[i].
        embeddings_out: list[list[float] | None] = []
        response_embeddings = getattr(response, "embeddings", None) or []

        if len(response_embeddings) != len(sanitized):
            # Shape mismatch from the provider is a protocol
            # violation; log it and degrade gracefully.
            logger.warning(
                "GeminiEmbeddingProvider: response length %d != input length %d. "
                "Returning all-None for the batch.",
                len(response_embeddings),
                len(sanitized),
            )
            return [None] * len(texts)

        for embedding_obj in response_embeddings:
            values = getattr(embedding_obj, "values", None)
            if values is None:
                embeddings_out.append(None)
                continue
            # Validate dimensionality against the configured
            # provider dimension. Off-dimension results usually
            # mean a model-version mismatch or an API drift, both
            # of which we should surface rather than silently mix.
            if len(values) != self._dimension:
                logger.warning(
                    "GeminiEmbeddingProvider: got embedding of dim %d, "
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
        EmbeddingProvider: OpenAI, Gemini, or null provider based on env config.
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
                "construction failed; falling back to Gemini/Null provider. "
                "Retrieval may degrade to token-recall only.",
                exc_info=True,
            )

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            return GeminiEmbeddingProvider()
        except Exception:
            logger.warning(
                "create_configured_embedding_provider: GeminiEmbeddingProvider "
                "construction failed; falling back to NullEmbeddingProvider. "
                "Retrieval will use token-recall only.",
                exc_info=True,
            )
            return NullEmbeddingProvider()
    return NullEmbeddingProvider()
