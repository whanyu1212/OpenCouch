"""In-memory implementation of the memory-store protocol.

The :class:`OpenCouchMemoryStore` keeps records in per-instance dicts and
is the default backend for tests and incognito-mode sessions, where no
connection lifecycle or disk persistence is wanted. Durable memory uses
:mod:`agent.memory.store.postgres`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.memory.retrieval.ranking import (
    IndexedRecord,
    dense_candidate_limit,
    dense_rank,
    lexical_rank,
    rrf_fuse,
)
from agent.memory.store.base import (
    SEARCH_MATCH_THRESHOLD,
    MemoryRecordFilter,
    Namespace,
    StoreRecord,
    memory_record_matches_filter,
    unpack_memory_namespace,
)


@dataclass(slots=True)
class _NamespaceBucket:
    """Internal per-namespace storage. One bucket per unique namespace."""

    records: dict[str, StoreRecord] = field(default_factory=dict)


class OpenCouchMemoryStore:
    """In-memory implementation of the :class:`MemoryStore` protocol.

    Records live in a per-instance dict keyed by namespace and are
    discarded when the instance is garbage collected. Tests and
    incognito-mode sessions prefer this implementation because it has
    no connection lifecycle and no disk writes. Supported persistent runtimes
    use :class:`agent.memory.store.postgres.PostgresMemoryStore`.

    The store is **not** thread-safe. Each runtime instance should own
    its own store; do not share a single instance across runtimes or
    across multiple concurrent calls.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store.

        Returns:
            None: Creates empty namespace buckets.
        """

        self._buckets: dict[Namespace, _NamespaceBucket] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        """Raise when the store has already been closed.

        Raises:
            RuntimeError: If the store is closed.

        Returns:
            None: The store is open.
        """

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
        """Write multiple in-memory records atomically.

        Records are materialized before any bucket is touched so that a
        malformed item cannot leave the batch half-applied. This mirrors the
        transactional guarantee the PostgreSQL backend gets from wrapping its
        writes in one transaction.

        Args:
            items (list[tuple[Namespace, str, dict[str, Any], list[float] | None, str | None]]):
                Items shaped as ``(namespace, key, value, embedding, embedding_model)``.

        Returns:
            None: Updates the in-memory store.
        """

        self._ensure_open()
        staged: list[tuple[Namespace, StoreRecord]] = []
        for namespace, key, value, embedding, embedding_model in items:
            # Validate eagerly so a malformed namespace rejects the whole batch,
            # matching the PostgreSQL backend instead of writing the good items
            # and raising partway through.
            unpack_memory_namespace(namespace)
            staged.append(
                (
                    namespace,
                    StoreRecord(
                        namespace=namespace,
                        key=key,
                        value=dict(value),
                        embedding=list(embedding) if embedding is not None else None,
                        embedding_model=embedding_model,
                    ),
                )
            )
        for namespace, record in staged:
            self._bucket(namespace).records[record.key] = record

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
        record_filter: MemoryRecordFilter | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval over in-memory records.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for lexical scoring.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (MemoryRecordFilter | None): Optional declarative filter
                applied before ranking and truncation.

        Returns:
            list[StoreRecord]: Top fused retrieval results.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        if max_age_days is not None:
            from datetime import UTC, datetime, timedelta

            cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

            def _is_recent(record: StoreRecord) -> bool:
                """Return whether a record was created within the age window.

                Args:
                    record (StoreRecord): Candidate record to inspect.

                Returns:
                    bool: ``True`` when the record is recent enough.
                """

                created_raw = record.value.get("created_at", "")
                if not created_raw:
                    return False
                try:
                    created_str = str(created_raw).replace("Z", "+00:00")
                    created = datetime.fromisoformat(created_str)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    return created >= cutoff
                except (ValueError, TypeError):
                    return False

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
                if memory_record_matches_filter(candidate.record, record_filter)
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
        )[: dense_candidate_limit(limit)]

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

    async def ensure_schema(self) -> None:
        """Prepare the in-memory store.

        Records live in per-instance dicts, so there is no schema to create
        and no connection to open. Kept to satisfy the store protocol and to
        keep incognito startup credential-free.

        Returns:
            None: No preparation is required.
        """

        self._ensure_open()

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

        if self._closed:
            return None
        bucket = self._buckets.get(namespace)
        if bucket is None or not bucket.records:
            return None
        return next(reversed(bucket.records.values()))
