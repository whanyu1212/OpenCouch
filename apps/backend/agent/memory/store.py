"""Unified memory store for the OpenCouch agent.

The :class:`MemoryStore` protocol is the interface the agent's nodes
use to read and write long-term memory. Two concrete implementations
satisfy it:

- :class:`OpenCouchMemoryStore` (in-memory, dict-backed) — original
  v0.1 scaffolding, fast and ephemeral. Used by unit tests and by
  incognito-mode CLI sessions where nothing should persist.
- :class:`agent.memory.sqlite_store.SqliteMemoryStore` (v0.8) —
  aiosqlite-backed, durable across process restarts. Used by
  persistent-mode CLI sessions so that semantic facts and episodic
  arcs survive ``/exit`` + restart.

Both implementations fan out records by namespace (semantic /
episodic / procedural) via a ``(user_id, kind)`` tuple. The in-memory
version keeps per-namespace dicts; the SQLite version keeps a single
``memory_records`` table with ``namespace_kind`` as a discriminator
column and an index on ``(owner_id, namespace_kind)``.

Phase 1 v0.8 scope (for this file):
- :class:`MemoryStore` protocol ships alongside the existing
  :class:`OpenCouchMemoryStore`. The protocol captures the async
  interface (``aput``, ``aget``, ``asearch``, ``adelete``, ``aclose``)
  plus the debug helpers the CLI uses (``record_count``,
  ``namespaces``). Both concrete classes satisfy it structurally.
- Token-recall search behavior is unchanged. :class:`SqliteMemoryStore`
  runs the same Python-side scoring loop used by
  :class:`OpenCouchMemoryStore` — the only difference is where the
  rows come from (SQL query vs dict iteration). See the
  :data:`SEARCH_MATCH_THRESHOLD` comment below for the scorer rationale.
- Incognito mode still uses :class:`OpenCouchMemoryStore` (the runtime
  picks the implementation based on mode).

Token-recall scoring details (v0.3.1, still current):
- Uses the shared tokenizer from :mod:`agent.memory.text_tokens` and
  computes ``|query_tokens ∩ haystack_tokens| / |query_tokens|`` after
  filtering stopwords from the query side. A record counts as a match
  when that recall ratio meets :data:`SEARCH_MATCH_THRESHOLD`.
- Results are returned in score-descending order with insertion order
  as a deterministic tiebreaker. Callers that relied on pure
  insertion-order behavior should be aware of the change; in practice
  the only caller (``load_memory_node``) wants best-match-first anyway.
- Still a placeholder for the real text-embedding-3-small pathway
  that lands after v0.8. Token-recall is deterministic, cheap, and
  dramatically better than substring match for the paraphrase-heavy
  retrieval path, without introducing any embedding dependency or
  cold-start model load.
- **Semantic** (v0.3) and **episodic** (v0.4) namespaces are both
  wired through the real extraction/retrieval path. Procedural reads
  still return empty (procedural memory lands in v0.7).

Design decisions:
- The store is a **standalone class hierarchy**, not a subclass of
  LangGraph's ``BaseStore``. The full ``BaseStore`` batch-op dispatcher
  is more scaffolding than phase 1 needs; phase 3 will revisit whether
  to inherit or write an adapter when graph memory lands.
- Namespaces are represented as tuples (``(user_id, kind)``) matching
  the LangGraph convention, so a future migration to ``BaseStore`` is
  a straightforward adapter rather than a redesign.
- The store exposes a small async interface: ``aput``, ``aget``,
  ``asearch``, ``adelete``, ``aclose``. No sync variants. Agent nodes
  always run in async context so there's no caller that needs sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent.memory.text_tokens import tokenize, tokenize_meaningful

# A namespace is a tuple of strings, typically ``(user_id, kind)`` where
# kind is one of "semantic", "episodic", "procedural". The tuple shape
# matches LangGraph's BaseStore convention so a future adapter is easy.
Namespace = tuple[str, ...]

# Minimum query-token recall ratio for a record to count as a search hit.
# Recall is ``|query_tokens ∩ haystack_tokens| / |query_tokens|`` after
# filtering stopwords from the query. 0.33 means "at least one in three
# of the query's meaningful tokens must appear in the record."
#
# Why 0.33 and not the more obvious 0.5:
# Natural user queries are wordier than the stored evidence quotes they
# query, so the denominator (query token count) tends to be larger than
# the overlap. A realistic retrieval case like the query "I feel
# overwhelmed at work" against a stored fact "I worry about work stress"
# produces meaningful tokens {feel, overwhelmed, work} vs a haystack that
# contains {work} — overlap 1, recall 1/3 ≈ 0.333. At threshold 0.5 this
# correctly-relevant record misses; at 0.33 it hits. The Stage E dogfood
# pass showed this pattern repeatedly, so the bar is set where natural
# queries with one topical keyword still land.
#
# Why not lower (0.2, 0.25):
# Lower thresholds start to allow spurious matches. The Stage E long-
# paraphrase query "things have been tense with Sarah lately" has four
# meaningful tokens and only {sarah} in the haystack — recall 1/4 = 0.25.
# At threshold 0.33 this correctly misses (low signal, only one keyword
# among four connectives). At threshold 0.25 it would hit, but only
# because Sarah is the one word the query happens to share. That's too
# lenient to trust — it matches on coincidence rather than majority
# signal.
#
# This is deliberately lower than dedup's 0.85 Jaccard threshold. The
# asymmetry is intentional: retrieval false-negatives hurt user experience
# directly (the agent forgets), while dedup false-positives hurt memory
# integrity permanently (distinct facts get merged). Tune retrieval
# loose, tune dedup tight. The v0.8 embedding pathway will replace this
# with proper semantic similarity and render the threshold obsolete.
SEARCH_MATCH_THRESHOLD = 0.33


@dataclass(slots=True)
class StoreRecord:
    """One record in the memory store.

    The store is namespace-aware; each record knows which namespace it
    belongs to. The ``value`` dict holds the serialized pydantic model
    (via ``model.model_dump()``) so the store stays model-agnostic.

    v0.8.1 adds two optional fields — ``embedding`` and
    ``embedding_model`` — to support the hybrid retrieval path via
    :meth:`MemoryStore.asearch_similar`. Records written before
    v0.8.1 or through a NullEmbeddingProvider leave both fields
    as ``None``, and those records participate in hybrid retrieval
    via the token-recall path only. Records with embeddings
    participate in both the token-recall and the embedding-
    similarity paths.
    """

    namespace: Namespace
    key: str
    value: dict[str, Any]
    embedding: list[float] | None = None
    embedding_model: str | None = None


@dataclass(slots=True)
class _NamespaceBucket:
    """Internal per-namespace storage. One bucket per unique namespace."""

    records: dict[str, StoreRecord] = field(default_factory=dict)


@runtime_checkable
class MemoryStore(Protocol):
    """The async memory-store interface both implementations satisfy.

    Added in v0.8 alongside the SQLite-backed implementation. Before
    v0.8, :class:`OpenCouchMemoryStore` was the only implementation and
    callers typed their store parameters as that concrete class. v0.8
    adds :class:`agent.memory.sqlite_store.SqliteMemoryStore` as a
    sibling — same interface, SQLite-backed persistence — so the
    runtime can swap implementations without the callers needing to
    know which one they're holding.

    The protocol mirrors the existing OpenCouchMemoryStore surface
    exactly, because OpenCouchMemoryStore was the reference
    implementation. Any method added to both concrete classes should
    be added here first.

    Design notes:
    - **``@runtime_checkable``** so ``isinstance(store, MemoryStore)``
      works for debug tools and tests. The runtime cost is small and
      it makes the protocol more useful for assertion-style checks.
    - **No class-level state in the protocol** — concrete classes
      manage their own lifecycle (in-memory dict vs SQLite connection).
    - **Names match the concrete method names** rather than being
      renamed for protocol-style brevity (``put`` vs ``aput``), so
      callers can substitute the type annotation without having to
      adjust call sites.

    When this protocol grows, keep two rules in mind:
    1. Any method added here MUST be implemented by both concrete
       classes before the protocol signature ships, or runtime callers
       that duck-type against the protocol will get AttributeError.
    2. The protocol should only capture the methods nodes and runtime
       code actually USE. Internal debug helpers (e.g.,
       ``record_count``) that aren't part of the node interface don't
       need to be in the protocol — they can live on the concrete
       classes only.
    """

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
        *,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Store a record under ``(namespace, key)``.

        v0.8.1: the optional ``embedding`` and ``embedding_model``
        parameters let callers attach a pre-computed embedding
        alongside the record. When present, the store uses them
        for the embedding-similarity path in :meth:`asearch_similar`.
        When absent (the common case for existing callers and for
        tests that don't need embeddings), the record is stored
        without an embedding and only the token-recall path
        applies. Both parameters default to None so existing
        callers are unchanged.

        The ``embedding_model`` string identifies which model
        produced the embedding — stored alongside the vector so
        future model migrations can detect cohort mismatches and
        skip cross-model similarity computations.
        """
        ...

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one record by its ``(namespace, key)``."""
        ...

    async def asearch(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Search for records within ``namespace`` matching ``query``.

        v0.3.1 token-recall path. Retained as of v0.8.1 for (a)
        backward compatibility with existing callers that don't
        compute embeddings, (b) the ``query=None`` enumeration
        path used by dedup, and (c) the fallback path inside
        :meth:`asearch_similar` when no embedding is available.
        """
        ...

    async def asearch_similar(
        self,
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Hybrid retrieval via Reciprocal Rank Fusion (v0.8.1).

        Runs the v0.3.1 token-recall scan and an embedding-similarity
        scan on the same namespace, then combines the two ranked
        lists with RRF (see :mod:`agent.memory.retrieval`). Returns
        the top-``limit`` records from the fused ranking.

        ``query_text`` is always required (it's the input to the
        token-recall path). ``query_embedding`` is optional: when
        ``None``, the method degenerates to pure token-recall
        (which is how guest mode, no-provider mode, and tests
        without embedding mocks continue to work). When provided,
        the embedding must match the dimensionality of the stored
        embeddings for this namespace — mismatches are silently
        skipped.

        ``embedding_model`` (optional) is the identifier of the
        model that produced ``query_embedding``. When present, the
        store skips records whose stored ``embedding_model`` doesn't
        match, preventing cross-model cosine similarity that isn't
        meaningful. When ``None``, the store uses every record's
        stored embedding regardless of model.

        Returns records sorted by hybrid RRF score descending, with
        insertion-order tiebreaking — same determinism contract as
        the token-recall ``asearch``.
        """
        ...

    async def adelete(
        self,
        namespace: Namespace,
        key: str,
    ) -> bool:
        """Delete a record by ``(namespace, key)``."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by the store."""
        ...

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Return the total number of records, optionally filtered by namespace.

        Included in the protocol because the CLI ``/memory status`` and
        ``/memory list`` commands both read it. Async because the
        SQLite-backed implementation needs to share the aiosqlite
        connection — a sync version would either open a second
        connection (breaking ``:memory:`` databases) or block the
        event loop.

        The ``a`` prefix matches the other async methods; the in-memory
        implementation just doesn't need to await anything but
        still has to be declared ``async`` to satisfy the protocol.
        """
        ...

    async def anamespaces(self) -> list[Namespace]:
        """Return every namespace that currently contains at least one record.

        Same async rationale as :meth:`arecord_count`.
        """
        ...


class OpenCouchMemoryStore:
    """In-memory implementation of the :class:`MemoryStore` protocol.

    This is the original v0.1 scaffolding — dict-backed, fast, and
    ephemeral. Records live in a per-instance dict keyed by namespace
    and are discarded when the instance is garbage collected.

    As of v0.8 this is no longer the only implementation. A sibling
    :class:`agent.memory.sqlite_store.SqliteMemoryStore` provides the
    same interface with SQLite-backed persistence. The runtime can
    accept either one through the :class:`MemoryStore` protocol type
    annotation. Tests and unit-test fixtures should prefer this
    in-memory version because it has no connection lifecycle and no
    I/O overhead; production runtime uses the SQLite version so
    session arcs and semantic facts survive CLI restarts.

    The store is **not** thread-safe. Each runtime instance should own
    its own store; do not share a single instance across runtimes or
    across multiple concurrent calls.
    """

    def __init__(self) -> None:
        self._buckets: dict[Namespace, _NamespaceBucket] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OpenCouchMemoryStore is closed.")

    def _bucket(self, namespace: Namespace) -> _NamespaceBucket:
        """Get (or create) the bucket for a namespace."""

        bucket = self._buckets.get(namespace)
        if bucket is None:
            bucket = _NamespaceBucket()
            self._buckets[namespace] = bucket
        return bucket

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
        *,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Store a record under ``(namespace, key)``.

        If a record already exists at that key, it is overwritten. The
        store does NOT perform hot-path deduplication here — that logic
        lives in the node layer (see ``extract_semantic_facts_node``
        in the design sketch) so the store stays a pure key-value layer.

        v0.8.1: ``embedding`` and ``embedding_model`` are optional
        keyword args. When the caller has a pre-computed embedding
        (via an :class:`agent.memory.embeddings.EmbeddingProvider`),
        it passes both here and the store keeps them alongside the
        record for use in :meth:`asearch_similar`. When the caller
        doesn't have an embedding (guest mode, no provider, tests),
        both default to None and the record participates in
        hybrid retrieval via the token-recall path only.
        """

        self._ensure_open()
        bucket = self._bucket(namespace)
        bucket.records[key] = StoreRecord(
            namespace=namespace,
            key=key,
            value=dict(value),  # defensive copy so callers can mutate the input
            embedding=list(embedding) if embedding is not None else None,
            embedding_model=embedding_model,
        )

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one record by its ``(namespace, key)``.

        Returns ``None`` when the record does not exist.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return None
        return bucket.records.get(key)

    async def asearch(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Search for records within ``namespace`` matching ``query``.

        v0.3.1 uses **query-token recall scoring** against the serialized
        value of each record. For each record:

        1. Concatenate its non-null string-coercible field values into a
           haystack string.
        2. Tokenize the haystack with the full (non-stopword-filtered)
           tokenizer so every query token has a chance to land.
        3. Tokenize the query with the stopword-and-length-filtered
           tokenizer so connective words don't inflate overlap.
        4. Compute ``recall = |query ∩ haystack| / |query|``.
        5. Keep the record if ``recall >= SEARCH_MATCH_THRESHOLD``.

        Results are returned sorted by recall descending, with insertion
        order as the tiebreaker. This makes ``load_memory_node``'s
        top-k retrieval genuinely pick the "best" matches instead of
        the first ``limit`` substring hits.

        Degenerate-query handling: if the query has no meaningful tokens
        after stopword filtering (e.g. the user typed "I am so" or "!!!"),
        returns an empty list. Returning the full namespace would be
        surprising behavior for a one-word noise query; returning
        nothing fails fast and lets the caller degrade however makes
        sense for its context.

        When ``query`` is ``None``, returns all records in the namespace
        (up to ``limit``) in insertion order. This path is used by
        ``extract_facts`` to enumerate existing records for dedup.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        if query is None:
            return list(bucket.records.values())[:limit]

        query_tokens = tokenize_meaningful(query)
        if not query_tokens:
            # No meaningful query tokens → nothing to score against.
            # Better to return empty than to flood the caller with
            # everything in the namespace.
            return []

        query_token_count = len(query_tokens)
        scored: list[tuple[float, int, StoreRecord]] = []
        for insertion_index, record in enumerate(bucket.records.values()):
            haystack = " ".join(str(v) for v in record.value.values() if v is not None)
            haystack_tokens = tokenize(haystack)
            if not haystack_tokens:
                continue
            overlap = len(query_tokens & haystack_tokens)
            recall = overlap / query_token_count
            if recall >= SEARCH_MATCH_THRESHOLD:
                # ``insertion_index`` is the tiebreaker that preserves
                # insertion order when two records have the same recall
                # score — keeps the test_memory_store_search_respects_limit
                # assertion deterministic.
                scored.append((recall, insertion_index, record))

        # Sort by (recall desc, insertion_index asc) then slice to limit.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:limit]]

    async def asearch_similar(
        self,
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Hybrid retrieval via Reciprocal Rank Fusion (v0.8.1).

        Combines the v0.3.1 token-recall scan with an embedding-
        similarity scan. See :mod:`agent.memory.retrieval` for the
        RRF fusion rationale and :data:`SEARCH_MATCH_THRESHOLD` /
        :data:`agent.memory.retrieval.EMBEDDING_MATCH_THRESHOLD`
        for the per-scorer cutoffs.

        Degenerate cases:

        - ``query_text`` has no meaningful tokens AND
          ``query_embedding`` is None → empty list. Neither scorer
          can contribute.
        - ``query_text`` has no meaningful tokens but
          ``query_embedding`` is provided → embedding-only result,
          RRF with one list degenerates to the dense ranking.
        - ``query_embedding`` is None but ``query_text`` has
          meaningful tokens → token-recall-only result, RRF with
          one list degenerates to the lexical ranking and this
          method behaves identically to :meth:`asearch` for this
          query.
        - Record has no stored embedding → it participates in
          the token-recall side only. Its rank position on the
          dense side is effectively infinite (i.e., zero contribution).
        - Stored ``embedding_model`` doesn't match the query's
          ``embedding_model`` → the record is skipped in the dense
          scan (still participates in the lexical scan). This
          prevents cross-model cosine similarity which isn't
          meaningful.
        """

        from agent.memory.retrieval import (
            EMBEDDING_MATCH_THRESHOLD,
            ScoredRecord,
            cosine_similarity,
            rrf_fuse,
        )

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        # ── Token-recall side (lexical) ───────────────────────────────
        lexical_scored: list[ScoredRecord] = []
        query_tokens = tokenize_meaningful(query_text)
        if query_tokens:
            query_token_count = len(query_tokens)
            for insertion_index, record in enumerate(bucket.records.values()):
                haystack = " ".join(
                    str(v) for v in record.value.values() if v is not None
                )
                haystack_tokens = tokenize(haystack)
                if not haystack_tokens:
                    continue
                overlap = len(query_tokens & haystack_tokens)
                recall = overlap / query_token_count
                if recall >= SEARCH_MATCH_THRESHOLD:
                    lexical_scored.append(
                        ScoredRecord(
                            record=record,
                            score=recall,
                            insertion_index=insertion_index,
                        )
                    )
            lexical_scored.sort(key=lambda sr: (-sr.score, sr.insertion_index))

        # ── Embedding-similarity side (dense) ─────────────────────────
        dense_scored: list[ScoredRecord] = []
        if query_embedding is not None:
            for insertion_index, record in enumerate(bucket.records.values()):
                if record.embedding is None:
                    continue
                # Skip cross-model similarity — comparing embeddings
                # from different models with cosine is noise.
                if (
                    embedding_model is not None
                    and record.embedding_model is not None
                    and record.embedding_model != embedding_model
                ):
                    continue
                # Dimensionality mismatch is also a skip; caller is
                # expected to have passed the right model, but we
                # guard against schema drift.
                if len(record.embedding) != len(query_embedding):
                    continue
                sim = cosine_similarity(query_embedding, record.embedding)
                if sim >= EMBEDDING_MATCH_THRESHOLD:
                    dense_scored.append(
                        ScoredRecord(
                            record=record,
                            score=sim,
                            insertion_index=insertion_index,
                        )
                    )
            dense_scored.sort(key=lambda sr: (-sr.score, sr.insertion_index))

        if not lexical_scored and not dense_scored:
            return []

        return rrf_fuse(
            lexical_ranked=lexical_scored,
            dense_ranked=dense_scored,
            limit=limit,
        )

    async def adelete(
        self,
        namespace: Namespace,
        key: str,
    ) -> bool:
        """Delete a record by ``(namespace, key)``.

        Returns ``True`` if a record was deleted, ``False`` if no record
        existed at that key. The distinction lets callers (e.g. the CLI
        ``/memory forget`` command) report success accurately.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None or key not in bucket.records:
            return False
        del bucket.records[key]
        return True

    async def aclose(self) -> None:
        """Mark the store as closed and clear its contents.

        Closed stores raise ``RuntimeError`` on any further access.
        Calling ``aclose`` on an already-closed store is a no-op.
        """

        if self._closed:
            return
        self._closed = True
        self._buckets.clear()

    # ── Debug / observability helpers ────────────────────────────────────

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Return the total number of records, optionally filtered by namespace.

        Used by ``/memory status`` and ``/memory list`` CLI commands
        and by tests. NOT part of the graph-node interface — nodes
        should use ``asearch`` with ``query=None`` if they need to
        enumerate. Async to match the SQLite implementation's
        connection-sharing contract (see :class:`MemoryStore` protocol).
        """

        if self._closed:
            return 0
        if namespace is not None:
            bucket = self._buckets.get(namespace)
            return len(bucket.records) if bucket is not None else 0
        return sum(len(b.records) for b in self._buckets.values())

    async def anamespaces(self) -> list[Namespace]:
        """Return every namespace that currently contains at least one record.

        Async to match the SQLite implementation's connection-sharing
        contract (see :class:`MemoryStore` protocol).
        """

        if self._closed:
            return []
        return [ns for ns, bucket in self._buckets.items() if bucket.records]
