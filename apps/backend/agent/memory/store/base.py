"""Unified memory-store interface and shared record types.

The :class:`MemoryStore` protocol is the async interface the agent's
nodes use to read and write long-term memory. Supported implementations are
:class:`agent.memory.store.memory.OpenCouchMemoryStore` for ephemeral use and
:class:`agent.memory.store.postgres.PostgresMemoryStore` for durable use.

Records are grouped by namespace, usually ``(user_id, kind)`` where
``kind`` is ``"semantic"``, ``"episodic"``, or ``"procedural"``. The
in-memory store keeps one dict-backed bucket per namespace. The Postgres store
persists the same logical records in a single table.

Search combines lexical token recall with optional embedding
similarity. Lexical recall remains deterministic and cheap, while
embedding matches improve paraphrase-heavy retrieval when an embedding
provider is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

# A namespace is typically ``(user_id, kind)`` where kind is one of
# "semantic", "episodic", or "procedural". The tuple shape mirrors common
# key-value memory store conventions.
Namespace = tuple[str, ...]
MemoryRecordFilter = Literal["active_semantic"]

# Minimum query-token recall ratio for a record to count as a search hit.
# Recall is ``|query_tokens ∩ haystack_tokens| / |query_tokens|`` after
# filtering stopwords from the query. 0.33 means "at least one in three
# of the query's meaningful tokens must appear in the record."
#
# Why 0.33 and not the more obvious 0.5:
# Natural user queries are often wordier than stored evidence quotes,
# so the denominator tends to be larger than the overlap. A query like
# "I feel overwhelmed at work" against "I worry about work stress" has
# one topical overlap across three meaningful query tokens. At 0.5 this
# relevant record misses; at 0.33 it lands.
#
# Why not lower (0.2, 0.25):
# Lower thresholds start to allow spurious matches. The query "things
# have been tense with Sarah lately" has four meaningful tokens and
# only {sarah} in a matching haystack, so recall is 1/4 = 0.25.
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
# loose, tune dedup tight. Embedding retrieval complements this lexical
# threshold but does not replace it; both paths feed hybrid ranking.
SEARCH_MATCH_THRESHOLD = 0.33


@dataclass(frozen=True, slots=True)
class PreparedMemoryRecordFields:
    """Backend-neutral fields derived from a memory payload before persistence."""

    category: Any
    serialized_value: str
    created_at: str
    last_referenced_at: str
    dormant_at: Any
    user_visible: bool
    embedding_dim: int | None


@dataclass(slots=True)
class StoreRecord:
    """One record in the memory store.

    The store is namespace-aware; each record knows which namespace it
    belongs to. The ``value`` dict holds the serialized pydantic model
    (via ``model.model_dump()``) so the store stays model-agnostic.

    Optional ``embedding`` and ``embedding_model`` fields support
    hybrid retrieval. Records without embeddings still participate
    through the lexical recall path.
    """

    namespace: Namespace
    key: str
    value: dict[str, Any]
    embedding: list[float] | None = None
    embedding_model: str | None = None


def memory_record_matches_filter(
    record: StoreRecord,
    record_filter: MemoryRecordFilter | None,
) -> bool:
    """Return whether a record satisfies a supported declarative filter."""

    if record_filter is None:
        return True
    if record_filter == "active_semantic":
        value = record.value
        return bool(
            value.get("user_visible", True)
            and not value.get("dormant_at")
            and not value.get("superseded_by")
        )
    raise ValueError(f"Unsupported memory record filter: {record_filter}")


def unpack_memory_namespace(namespace: Namespace) -> tuple[str, str]:
    """Extract normalized owner and kind fields from a memory namespace."""

    if len(namespace) != 2:
        raise ValueError(
            f"MemoryStore namespace must be (owner_id, kind) tuple; got {namespace!r}"
        )
    owner_id, namespace_kind = namespace
    return str(owner_id), str(namespace_kind)


def prepare_memory_record_fields(
    value: dict[str, Any],
    *,
    embedding: list[float] | None,
) -> PreparedMemoryRecordFields:
    """Derive backend-neutral memory row fields from a serialized payload."""

    created_at = str(value.get("created_at") or "")
    return PreparedMemoryRecordFields(
        category=value.get("category"),
        serialized_value=json.dumps(value, default=str),
        created_at=created_at,
        last_referenced_at=str(value.get("last_referenced_at") or created_at or ""),
        dormant_at=value.get("dormant_at"),
        user_visible=bool(value.get("user_visible", True)),
        embedding_dim=len(embedding) if embedding is not None else None,
    )


def parse_store_record_value(value: Any) -> dict[str, Any]:
    """Return a StoreRecord value from a backend-extracted payload."""

    if isinstance(value, str):
        return cast(dict[str, Any], json.loads(value))
    return cast(dict[str, Any], value)


def build_store_record(
    *,
    namespace: Namespace,
    key: Any,
    value: Any,
    embedding: list[float] | None,
    embedding_model: str | None,
) -> StoreRecord:
    """Build the shared StoreRecord shape from backend-extracted fields."""

    return StoreRecord(
        namespace=namespace,
        key=str(key),
        value=parse_store_record_value(value),
        embedding=embedding,
        embedding_model=embedding_model,
    )


@runtime_checkable
class MemoryStore(Protocol):
    """The async interface implemented by supported memory stores.

    The runtime can swap ephemeral in-memory and durable Postgres
    implementations without node code knowing which one it is holding.

    Design notes:
    - **``@runtime_checkable``** so ``isinstance(store, MemoryStore)``
      works for debug tools and tests. The runtime cost is small and
      it makes the protocol more useful for assertion-style checks.
    - **No class-level state in the protocol** — concrete classes
      manage their own lifecycle (in-memory dict vs database connection).
    - **Names match the concrete method names** rather than being
      renamed for protocol-style brevity (``put`` vs ``aput``), so
      callers can substitute the type annotation without having to
      adjust call sites.

    Keep this protocol limited to methods used by nodes, runtime code,
    or CLI diagnostics. Any method added here must be implemented by
    all supported concrete stores before callers depend on it.
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
        record_filter: MemoryRecordFilter | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid lexical-plus-dense retrieval in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for the lexical scorer.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (MemoryRecordFilter | None): Optional declarative filter
                applied before ranking and truncation.

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
