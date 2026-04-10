"""Unified memory store for the OpenCouch agent.

The ``OpenCouchMemoryStore`` is the single entry point the agent's nodes
use to read and write long-term memory. It fans out to the right backend
based on the record namespace (semantic / episodic / procedural) and the
active ``MemoryMode`` (incognito / local / synced).

Phase 1 v0.3 scope:
- In-memory backing only. No SQLite, no Postgres, no Graphiti. All data
  lives in a per-instance dict and is discarded when the instance is
  garbage collected.
- **Token-recall search** against the serialized value of each record.
  This replaces v0.1's one-directional substring match, which failed on
  paraphrased queries (e.g. "tense with Sarah lately" wouldn't find a
  stored "I have a sister named Sarah" because the query isn't a
  substring of the haystack). The v0.3.1 scorer uses the shared
  tokenizer from :mod:`agent.memory.text_tokens` and computes
  ``|query_tokens ∩ haystack_tokens| / |query_tokens|`` after filtering
  stopwords from the query side. A record counts as a match when that
  recall ratio meets :data:`SEARCH_MATCH_THRESHOLD`.
- Results are returned in **score-descending order** (highest recall
  first), with insertion order as a deterministic tiebreaker. Callers
  that relied on pure insertion-order behavior should be aware of the
  change; in practice the only caller (``load_memory_node``) wants
  best-match-first anyway.
- This is still a placeholder for the real text-embedding-3-small
  pathway that lands alongside the SQLite backend in v0.8. Token-recall
  is deterministic, cheap, and dramatically better than substring match
  for the paraphrase-heavy retrieval path, without introducing any
  embedding dependency or cold-start model load.
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
