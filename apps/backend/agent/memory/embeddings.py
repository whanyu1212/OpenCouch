"""Embedding providers for semantic retrieval (v0.8.1).

Before v0.8.1 the memory store's retrieval path was token-recall only:
the v0.3.1 scorer in ``store.asearch`` computed
``|query_tokens ∩ haystack_tokens| / |query_tokens|`` and returned
matches above threshold 0.33. That works for exact-token overlap
and reasonable paraphrases, but it has documented failure modes:
stemming ("anxiety" ↔ "anxious"), synonyms ("sister" ↔ "sibling"),
tense / number variation, and semantic paraphrase without lexical
overlap. Those gaps are the main dogfood pain points v0.8.1 closes.

This module declares the :class:`EmbeddingProvider` protocol and
two concrete implementations:

- :class:`GeminiEmbeddingProvider` — calls Google's
  ``text-embedding-004`` via the existing ``google.genai`` client.
  This is the default when a Gemini API key is configured (via
  ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env vars).

- :class:`NullEmbeddingProvider` — a no-op that always returns
  ``None`` for embedding calls. Used when no API key is available,
  when memory mode is INCOGNITO, or in tests that don't want to
  mock a real embedding provider. Nodes that try to compute an
  embedding via a ``NullEmbeddingProvider`` get back ``None``,
  which is the signal to fall back to token-recall.

Design decisions locked for v0.8.1:

1. **Protocol, not inheritance.** The :class:`EmbeddingProvider`
   protocol has exactly one method (``aembed``) plus two metadata
   properties (``dimension``, ``model_name``). Concrete classes
   implement the protocol structurally rather than inheriting from
   an abstract base class — this matches the
   :class:`agent.memory.store.MemoryStore` pattern and keeps the
   provider surface minimal.

2. **Metadata is part of the protocol.** ``dimension`` and
   ``model_name`` are required because the store writes them
   alongside each record's embedding (in the ``embedding_dim`` and
   ``embedding_model`` columns added in v0.8.1). Later model
   migrations can compare the stored model name against the current
   provider's model name and skip retrieval on mismatched cohorts
   until a re-embed sweep runs.

3. **Batch support.** ``aembed`` takes a list of texts and returns
   a list of embeddings (one per input). This matches the Gemini
   and OpenAI batch embedding APIs and lets callers efficiently
   embed multiple records in a single network round-trip when
   they need to.

4. **Embeddings are plain ``list[float]``, not numpy arrays.** The
   store serializes embeddings to a BLOB via
   ``struct.pack``/``unpack`` (see ``sqlite_store.py``). Keeping
   the protocol type as plain lists means callers don't need
   numpy at the boundary — numpy only enters the picture inside
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
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Gemini's text-embedding-004 is the default model for v0.8.1. It
# produces 768-dimensional embeddings, supports batch input, and is
# cheap enough (~$0.00001/1k tokens) that dogfood doesn't need
# per-session cost tracking. Switching models later requires (a)
# updating this constant or injecting the model name at provider
# construction time, and (b) running a re-embed sweep for records
# that still have the old model stored in ``embedding_model``.
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GEMINI_EMBEDDING_DIMENSION = 768


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The async embedding interface for v0.8.1 retrieval.

    Concrete implementations:
    - :class:`GeminiEmbeddingProvider` — Google text-embedding-004
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
            texts: List of strings to embed. Empty strings are
                allowed but will produce a zero-vector or a None
                depending on the provider. Callers should skip
                empty texts upstream when possible.
            task_type: The embedding task hint (Gemini-specific).
                ``"RETRIEVAL_DOCUMENT"`` is the default for
                document-side embeddings (facts, arcs) written
                at extraction time. ``"RETRIEVAL_QUERY"`` is used
                for the query-side embeddings computed in
                ``load_memory_node``. Asymmetric embeddings tuned
                for retrieval generally prefer this distinction
                because document-side and query-side embeddings
                have different usage patterns. Providers that
                don't support task types (e.g.,
                :class:`NullEmbeddingProvider`) can ignore this
                argument.

        Returns:
            A list of embeddings, one per input text. Each entry
            is either a list of floats (length equal to
            ``self.dimension``) or ``None`` if embedding computation
            failed for that specific text. The list length always
            matches the input length so callers can zip
            texts with their embeddings by index.

        Never raises: provider errors are caught internally and
        returned as ``None`` entries in the output list. This is a
        deliberate contract — the node code above (extractor, load
        memory) treats an embedding as an optimization, not a
        correctness requirement, so it should never poison a
        turn with an embedding-provider exception.
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

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",  # noqa: ARG002 — contract
    ) -> list[list[float] | None]:
        """Return a list of ``None`` values matching the input length.

        The ``None`` return signals "no embedding available" which
        callers treat as "skip embedding, use token-recall."
        """

        return [None] * len(texts)


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
                ``text-embedding-004``. Changing this requires a
                re-embed sweep for records already stored with the
                old model, since cosine similarity across model
                cohorts is not meaningful.
            dimension: The model's output dimensionality. Used by
                the store to validate stored-vs-current matches.
                Defaults to 768 (text-embedding-004's native
                dimension). Some models support configurable
                output dimensions (OpenAI 3-small can be 512 or
                1536); Gemini 004 is fixed at 768.

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

        # Import inside __init__ so the module can be imported in
        # environments without google-genai installed (though the
        # project always has it as of v0.8). Matches the laziness
        # pattern used elsewhere in services/llm/.
        from google import genai

        self._client = genai.Client(api_key=resolved_key)
        self._model = model
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def aembed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float] | None]:
        """Embed a batch of texts via Gemini's embed_content API.

        See :meth:`EmbeddingProvider.aembed` for the contract.
        Provider errors are caught and returned as all-``None``
        results so a transient Gemini outage degrades to token-recall
        rather than poisoning the turn.

        Empty input strings are replaced with a single space before
        embedding because the Gemini API rejects zero-length
        content. The caller should ideally filter empty texts
        upstream, but the fallback means an accidental empty won't
        crash the whole batch.
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
                contents=sanitized,
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
    """Build the right embedding provider based on environment config.

    Resolution order:
    1. If ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` is set, returns
       :class:`GeminiEmbeddingProvider`.
    2. Otherwise, returns :class:`NullEmbeddingProvider`.

    This helper is the canonical wiring path used by
    :class:`agent.persistence.PersistentAgentRuntime`. Tests and
    one-off scripts can instantiate a concrete provider directly.

    Returns a provider that always satisfies the
    :class:`EmbeddingProvider` protocol — the caller never has to
    handle a ``None`` return here. Graceful degradation happens at
    the ``aembed`` call level, not at construction time.
    """

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
