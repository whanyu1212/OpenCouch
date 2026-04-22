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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent.memory.retrieval import IndexedRecord, dense_rank, lexical_rank, rrf_fuse

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
        """Store one record under a namespace/key pair.

        Args:
            namespace (Namespace): Record namespace tuple.
            key (str): Record key within the namespace.
            value (dict[str, Any]): Serialized record payload.
            embedding (list[float] | None): Optional precomputed embedding vector.
            embedding_model (str | None): Optional embedding model identifier.

        Returns:
            None: Stores or overwrites the record.
        """
        ...

    async def aput_batch(
        self,
        items: list[
            tuple[
                Namespace,
                str,
                dict[str, Any],
                list[float] | None,
                str | None,
            ]
        ],
    ) -> None:
        """Write multiple records in one batch.

        Args:
            items (list[tuple[Namespace, str, dict[str, Any], list[float] | None, str | None]]):
                Items shaped as ``(namespace, key, value, embedding, embedding_model)``.

        Returns:
            None: Persists the batch atomically for the backend.
        """
        ...

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one record by namespace and key.

        Args:
            namespace (Namespace): Namespace tuple to search.
            key (str): Record key within the namespace.

        Returns:
            StoreRecord | None: Matching record, or ``None``.
        """
        ...

    async def asearch(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Search records in a namespace with lexical scoring.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query (str | None): Optional lexical query. ``None`` enumerates records.
            limit (int): Maximum number of records to return.

        Returns:
            list[StoreRecord]: Matching records in backend-defined order.
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
        max_age_days: int | None = None,
        record_filter: Callable[[StoreRecord], bool] | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid lexical-plus-dense retrieval in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for the lexical scorer.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (Callable[[StoreRecord], bool] | None): Optional predicate
                applied to candidate records before ranking and truncation.

        Returns:
            list[StoreRecord]: Top fused retrieval results.
        """
        ...

    async def adelete(
        self,
        namespace: Namespace,
        key: str,
    ) -> bool:
        """Delete one record by namespace and key.

        Args:
            namespace (Namespace): Namespace tuple containing the record.
            key (str): Record key within the namespace.

        Returns:
            bool: ``True`` when a record was deleted.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Returns:
            None: Closes the store.
        """
        ...

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Count stored records.

        Args:
            namespace (Namespace | None): Optional namespace filter.

        Returns:
            int: Total record count for the store or namespace.
        """
        ...

    async def anamespaces(self) -> list[Namespace]:
        """List namespaces that currently contain records.

        Returns:
            list[Namespace]: Non-empty namespaces.
        """
        ...

    async def alatest(self, namespace: Namespace) -> StoreRecord | None:
        """Fetch the most recently inserted record in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to inspect.

        Returns:
            StoreRecord | None: Most recent record, or ``None``.
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
        """Get or create the bucket for a namespace.

        Args:
            namespace (Namespace): Namespace tuple to resolve.

        Returns:
            _NamespaceBucket: Bucket backing the namespace.
        """

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
        """Store or overwrite one in-memory record.

        Args:
            namespace (Namespace): Record namespace tuple.
            key (str): Record key within the namespace.
            value (dict[str, Any]): Serialized record payload.
            embedding (list[float] | None): Optional precomputed embedding vector.
            embedding_model (str | None): Optional embedding model identifier.

        Returns:
            None: Updates the in-memory store.
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

    async def aput_batch(
        self,
        items: list[
            tuple[
                Namespace,
                str,
                dict[str, Any],
                list[float] | None,
                str | None,
            ]
        ],
    ) -> None:
        """Write multiple in-memory records.

        Args:
            items (list[tuple[Namespace, str, dict[str, Any], list[float] | None, str | None]]):
                Items shaped as ``(namespace, key, value, embedding, embedding_model)``.

        Returns:
            None: Updates the in-memory store.
        """

        self._ensure_open()
        for namespace, key, value, embedding, embedding_model in items:
            bucket = self._bucket(namespace)
            bucket.records[key] = StoreRecord(
                namespace=namespace,
                key=key,
                value=dict(value),
                embedding=list(embedding) if embedding is not None else None,
                embedding_model=embedding_model,
            )

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one in-memory record.

        Args:
            namespace (Namespace): Namespace tuple to search.
            key (str): Record key within the namespace.

        Returns:
            StoreRecord | None: Matching record, or ``None``.
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
        """Search in-memory records with lexical scoring.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query (str | None): Optional lexical query. ``None`` enumerates records.
            limit (int): Maximum number of records to return.

        Returns:
            list[StoreRecord]: Matching records in recall-score order.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        if query is None:
            return list(bucket.records.values())[:limit]

        candidates = [
            IndexedRecord(record=record, insertion_index=insertion_index)
            for insertion_index, record in enumerate(bucket.records.values())
        ]
        scored = lexical_rank(
            candidates,
            query_text=query,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )
        return [scored_record.record for scored_record in scored[:limit]]

    async def asearch_similar(
        self,
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
        max_age_days: int | None = None,
        record_filter: Callable[[StoreRecord], bool] | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval over in-memory records.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for lexical scoring.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (Callable[[StoreRecord], bool] | None): Optional predicate
                applied to candidate records before ranking and truncation.

        Returns:
            list[StoreRecord]: Top fused retrieval results.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        # Pre-filter by age when requested
        if max_age_days is not None:
            from datetime import UTC, datetime, timedelta

            cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

            def _is_recent(record: StoreRecord) -> bool:
                created_raw = record.value.get("created_at", "")
                if not created_raw:
                    return False  # no timestamp → exclude from time-filtered results
                try:
                    created_str = str(created_raw).replace("Z", "+00:00")
                    created = datetime.fromisoformat(created_str)
                    # Treat naive timestamps as UTC
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    return created >= cutoff
                except (ValueError, TypeError):
                    return False  # unparseable → exclude

            candidates = {k: v for k, v in bucket.records.items() if _is_recent(v)}
        else:
            candidates = bucket.records

        indexed_candidates = [
            IndexedRecord(record=record, insertion_index=insertion_index)
            for insertion_index, record in enumerate(candidates.values())
        ]
        if record_filter is not None:
            indexed_candidates = [
                candidate
                for candidate in indexed_candidates
                if record_filter(candidate.record)
            ]
        lexical_scored = lexical_rank(
            indexed_candidates,
            query_text=query_text,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )
        dense_scored = dense_rank(
            indexed_candidates,
            query_embedding=query_embedding,
            embedding_model=embedding_model,
        )

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
        """Delete one in-memory record.

        Args:
            namespace (Namespace): Namespace tuple containing the record.
            key (str): Record key within the namespace.

        Returns:
            bool: ``True`` when a record was deleted.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None or key not in bucket.records:
            return False
        del bucket.records[key]
        return True

    async def aclose(self) -> None:
        """Close the in-memory store.

        Returns:
            None: Marks the store closed and clears in-memory data.
        """

        if self._closed:
            return
        self._closed = True
        self._buckets.clear()

    # ── Debug / observability helpers ────────────────────────────────────

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Count in-memory records.

        Args:
            namespace (Namespace | None): Optional namespace filter.

        Returns:
            int: Total record count for the store or namespace.
        """

        if self._closed:
            return 0
        if namespace is not None:
            bucket = self._buckets.get(namespace)
            return len(bucket.records) if bucket is not None else 0
        return sum(len(b.records) for b in self._buckets.values())

    async def anamespaces(self) -> list[Namespace]:
        """List non-empty in-memory namespaces.

        Returns:
            list[Namespace]: Namespaces that currently contain records.
        """

        if self._closed:
            return []
        return [ns for ns, bucket in self._buckets.items() if bucket.records]

    async def alatest(self, namespace: Namespace) -> StoreRecord | None:
        """Fetch the latest in-memory record in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to inspect.

        Returns:
            StoreRecord | None: Most recent record, or ``None``.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None or not bucket.records:
            return None
        return next(reversed(bucket.records.values()))
