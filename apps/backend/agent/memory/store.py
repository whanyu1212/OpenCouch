"""Unified memory store for the OpenCouch agent.

The ``OpenCouchMemoryStore`` is the single entry point the agent's nodes
use to read and write long-term memory. It fans out to the right backend
based on the record namespace (semantic / episodic / procedural) and the
active ``MemoryMode`` (incognito / local / synced).

Phase 1 v0.1 scope:
- In-memory backing only. No SQLite, no Postgres, no Graphiti. All data
  lives in a per-instance dict and is discarded when the instance is
  garbage collected.
- Substring search instead of embedding similarity. This is a placeholder
  for the real text-embedding-3-small pathway that lands alongside the
  SQLite backend in v0.8. Substring search is enough to prove the wiring
  and to unit-test the store's put/get/search round-trip.
- Only the semantic namespace is wired. Episodic and procedural reads
  return empty; writes to those namespaces are accepted but stored in
  the same underlying dict without special handling.

Design decisions:
- The store is a **standalone class**, not a subclass of LangGraph's
  ``BaseStore``. The full ``BaseStore`` batch-op dispatcher is more
  scaffolding than v0.1 needs; phase 3 will revisit whether to inherit
  or write an adapter when graph memory lands.
- Namespaces are represented as tuples (``(user_id, kind)``) matching the
  LangGraph convention, so a future migration to ``BaseStore`` is a
  straightforward adapter rather than a redesign.
- The store exposes a small async interface: ``aput``, ``aget``,
  ``asearch``, ``adelete``, ``aclose``. No sync variants. Agent nodes
  always run in async context so there's no caller that needs sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A namespace is a tuple of strings, typically ``(user_id, kind)`` where
# kind is one of "semantic", "episodic", "procedural". The tuple shape
# matches LangGraph's BaseStore convention so a future adapter is easy.
Namespace = tuple[str, ...]


@dataclass(slots=True)
class StoreRecord:
    """One record in the memory store.

    The store is namespace-aware; each record knows which namespace it
    belongs to. The ``value`` dict holds the serialized pydantic model
    (via ``model.model_dump()``) so the store stays model-agnostic.
    """

    namespace: Namespace
    key: str
    value: dict[str, Any]


@dataclass(slots=True)
class _NamespaceBucket:
    """Internal per-namespace storage. One bucket per unique namespace."""

    records: dict[str, StoreRecord] = field(default_factory=dict)


class OpenCouchMemoryStore:
    """In-memory store with a namespace-aware put/get/search interface.

    This is the v0.1 scaffolding — it proves the wiring and the node
    interface without committing to any persistence backend. The shape
    of the public methods is designed so that swapping in a SQLite-backed
    or Postgres-backed implementation later is a drop-in replacement.

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
    ) -> None:
        """Store a record under ``(namespace, key)``.

        If a record already exists at that key, it is overwritten. The
        store does NOT perform hot-path deduplication here — that logic
        lives in the node layer (see ``extract_semantic_facts_node``
        in the design sketch) so the store stays a pure key-value layer.
        """

        self._ensure_open()
        bucket = self._bucket(namespace)
        bucket.records[key] = StoreRecord(
            namespace=namespace,
            key=key,
            value=dict(value),  # defensive copy so callers can mutate the input
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

        Phase 1 v0.1 uses **case-insensitive substring matching** against
        the serialized value of each record. This is a placeholder for
        the real embedding-based similarity search that lands with the
        SQLite backend in v0.8 — substring matching works against the
        same ``evidence_quote`` field the embeddings would use, which
        makes the v0.1 test suite forward-compatible.

        When ``query`` is ``None``, returns all records in the namespace
        (up to ``limit``) in insertion order.
        """

        self._ensure_open()
        bucket = self._buckets.get(namespace)
        if bucket is None:
            return []

        if query is None:
            return list(bucket.records.values())[:limit]

        needle = query.casefold()
        matches: list[StoreRecord] = []
        for record in bucket.records.values():
            # Search across every string field of the record's value.
            # This is deliberately lenient — any substring hit counts.
            haystack = " ".join(
                str(v) for v in record.value.values() if v is not None
            ).casefold()
            if needle in haystack:
                matches.append(record)
                if len(matches) >= limit:
                    break
        return matches

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

    # ── Debug / observability helpers (not part of the node interface) ───

    def record_count(self, namespace: Namespace | None = None) -> int:
        """Return the total number of records, optionally filtered by namespace.

        Used by ``/memory status`` CLI command and by tests. NOT part of
        the public node interface — nodes should use ``asearch`` with
        ``query=None`` if they need to enumerate.
        """

        if self._closed:
            return 0
        if namespace is not None:
            bucket = self._buckets.get(namespace)
            return len(bucket.records) if bucket is not None else 0
        return sum(len(b.records) for b in self._buckets.values())

    def namespaces(self) -> list[Namespace]:
        """Return every namespace that currently contains at least one record."""

        if self._closed:
            return []
        return [ns for ns, bucket in self._buckets.items() if bucket.records]
