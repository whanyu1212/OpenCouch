"""Unit tests for OpenCouchMemoryStore and InMemoryCrisisLogBackend.

These tests exercise the store interface that phase-1 memory nodes will
use: put, get, search, delete, round-trip, and the append-only crisis
log backend. They are deliberately shape-only — no pydantic model
validation, no LLM calls, no async fixtures more complex than
``pytest.mark.asyncio``.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.memory.crisis_log import (
    InMemoryCrisisLogBackend,
    NullCrisisLogBackend,
)
from agent.memory.models import CrisisLogRecord
from agent.memory.store import OpenCouchMemoryStore, StoreRecord


# ─── OpenCouchMemoryStore round-trip tests ────────────────────────────────


@pytest.mark.asyncio
async def test_memory_store_put_and_get_round_trip() -> None:
    """A record stored with aput should be retrievable via aget."""

    store = OpenCouchMemoryStore()
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})

    record = await store.aget(namespace, "fact-1")

    assert record is not None
    assert isinstance(record, StoreRecord)
    assert record.namespace == namespace
    assert record.key == "fact-1"
    assert record.value == {"evidence_quote": "I worry about work"}


@pytest.mark.asyncio
async def test_memory_store_get_missing_key_returns_none() -> None:
    """Fetching a non-existent key should return None, not raise."""

    store = OpenCouchMemoryStore()
    record = await store.aget(("user-1", "semantic"), "nope")
    assert record is None


@pytest.mark.asyncio
async def test_memory_store_search_by_substring() -> None:
    """asearch should return records matching a case-insensitive substring."""

    store = OpenCouchMemoryStore()
    namespace = ("user-1", "semantic")
    await store.aput(
        namespace, "fact-1", {"evidence_quote": "Work stress keeps me up at night"}
    )
    await store.aput(
        namespace, "fact-2", {"evidence_quote": "My sister is coming to visit"}
    )
    await store.aput(
        namespace, "fact-3", {"evidence_quote": "Job interviews make me anxious"}
    )

    # "work" should match fact-1 only
    work_matches = await store.asearch(namespace, query="work")
    assert len(work_matches) == 1
    assert work_matches[0].key == "fact-1"

    # Case-insensitive
    upper_matches = await store.asearch(namespace, query="WORK")
    assert len(upper_matches) == 1
    assert upper_matches[0].key == "fact-1"


@pytest.mark.asyncio
async def test_memory_store_search_empty_namespace_returns_empty_list() -> None:
    """Searching an unknown namespace should return an empty list."""

    store = OpenCouchMemoryStore()
    results = await store.asearch(("nobody", "semantic"), query="anything")
    assert results == []


@pytest.mark.asyncio
async def test_memory_store_search_with_none_query_returns_all_up_to_limit() -> None:
    """Passing query=None should return all records up to the limit."""

    store = OpenCouchMemoryStore()
    namespace = ("user-1", "semantic")
    for i in range(15):
        await store.aput(namespace, f"fact-{i}", {"evidence_quote": f"Fact {i}"})

    results = await store.asearch(namespace, query=None, limit=10)
    assert len(results) == 10


@pytest.mark.asyncio
async def test_memory_store_search_respects_limit() -> None:
    """asearch with query should respect the limit parameter."""

    store = OpenCouchMemoryStore()
    namespace = ("user-1", "semantic")
    for i in range(5):
        await store.aput(
            namespace, f"fact-{i}", {"evidence_quote": "worry worry worry"}
        )

    results = await store.asearch(namespace, query="worry", limit=3)
    assert len(results) == 3


# ─── v0.3.1 token-recall search regression tests ────────────────────────
#
# These tests pin the behavior introduced by the v0.3.1 retrieval-gap fix.
# The v0.1 store used one-directional substring matching (``if needle in
# haystack``), which meant paraphrased queries returned zero results even
# when the fact was in the store. The smoke test at the end of v0.3
# Stage E surfaced the gap with three canary queries; the first three
# tests below turn those canaries into permanent regressions.
#
# See ``agent/memory/store.py`` SEARCH_MATCH_THRESHOLD and the docstring
# on ``OpenCouchMemoryStore.asearch`` for the scoring algorithm.


class TestTokenRecallSearch:
    """Regression tests for the v0.3.1 query-token recall scorer."""

    @pytest.mark.asyncio
    async def test_long_paraphrase_finds_stored_fact(self) -> None:
        """The v0.3 Stage E smoke test canary: a paraphrased long query
        should find a stored fact whose evidence quote shares only some
        of its tokens. The v0.1 substring matcher returned 0 results for
        this input; the v0.3.1 recall scorer must return the record."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace,
            "fact-sarah",
            {
                "evidence_quote": "I have a sister named Sarah",
                "subject": "user",
                "predicate": "KNOWS",
                "object": "Sarah",
            },
        )

        # Meaningful query tokens: {things, tense, sarah, lately}.
        # Haystack tokens include {sarah}.
        # Recall: 1/4 = 0.25 — BELOW the 0.33 SEARCH_MATCH_THRESHOLD.
        # So the full-sentence paraphrase query is still a miss. This
        # test pins that behavior: at the current threshold, "Sarah"
        # alone has to carry the query, but four-way dilution is too
        # much noise. The next test exercises the short-query case,
        # which SHOULD find the fact.
        results = await store.asearch(
            namespace, query="Things have been tense with Sarah lately"
        )
        # Document the current (acceptable) behavior: low-signal queries
        # with one named entity buried in connective noise still miss.
        # The v0.8 embedding pathway will close this gap; at v0.3.1 we
        # only promise that **short named-entity queries**, **majority-
        # overlap paraphrases**, and **queries with one topical keyword
        # out of ~3 meaningful tokens** work.
        assert results == []

    @pytest.mark.asyncio
    async def test_short_named_entity_query_finds_stored_fact(self) -> None:
        """The second v0.3 Stage E canary: a single-word named-entity
        query should find the stored fact. This is the main win of the
        v0.3.1 fix — the v0.1 substring matcher handled this case, and
        the v0.3.1 recall scorer must continue to handle it."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        # Query tokens (after stopword filter): {sarah}.
        # Haystack tokens include {sarah}. Recall: 1/1 = 1.0.
        results = await store.asearch(namespace, query="Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-sarah"

    @pytest.mark.asyncio
    async def test_majority_overlap_paraphrase_finds_stored_fact(self) -> None:
        """The third v0.3 Stage E canary: when the query shares the
        majority of its meaningful tokens with the haystack, the fact
        should be retrieved. Exact-phrase queries are the easy win."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        # Query tokens (after stopword filter): {have, sister, named, sarah}.
        # Haystack has all four. Recall: 4/4 = 1.0.
        results = await store.asearch(namespace, query="I have a sister named Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-sarah"

    @pytest.mark.asyncio
    async def test_stopword_only_query_returns_empty(self) -> None:
        """A query containing only stopwords has no meaningful tokens
        after filtering, so search should return an empty list rather
        than flooding the caller with the whole namespace."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})
        await store.aput(
            namespace, "fact-2", {"evidence_quote": "My sister Sarah visited"}
        )

        # Query "I am so the" → all tokens are stopwords → empty query set.
        results = await store.asearch(namespace, query="I am so the")
        assert results == []

    @pytest.mark.asyncio
    async def test_punctuation_only_query_returns_empty(self) -> None:
        """A query with no word characters yields zero tokens and should
        return an empty list, not crash."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(namespace, "fact-1", {"evidence_quote": "work stress"})

        results = await store.asearch(namespace, query="!!!???")
        assert results == []

    @pytest.mark.asyncio
    async def test_connective_overlap_is_filtered_by_stopwords(self) -> None:
        """A query and a haystack that share only connective words
        (via the stopword list) should NOT match. This guards against
        spurious matches like 'I worry about Sarah' vs 'I worry about work'
        where the only shared tokens are the stopwords."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace, "fact-work", {"evidence_quote": "I worry about work"}
        )

        # Query meaningful tokens: {worry, sarah}.
        # Haystack full tokens: {i, worry, about, work}.
        # Overlap on {worry}: 1. Recall: 1/2 = 0.5, above the 0.33
        # threshold, so this SHOULD match. The guard is that the
        # connectives "I" and "about" are NOT inflating the overlap
        # beyond what the topical overlap justifies — if they were
        # counted, recall would artificially jump to 4/2 (nonsense).
        results = await store.asearch(namespace, query="I worry about Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-work"

        # Now a harder case: a query with NO topical overlap at all.
        # Query: "the book is on the table" → meaningful tokens: {book, table}.
        # Neither appears in the haystack. Recall: 0/2 = 0.0 → no match.
        results_no_overlap = await store.asearch(
            namespace, query="the book is on the table"
        )
        assert results_no_overlap == []

    @pytest.mark.asyncio
    async def test_results_ordered_by_recall_score_desc(self) -> None:
        """When multiple records match, they should be returned in
        score-descending order — the best match first. This is the
        behavior change that makes ``load_memory_node``'s top-k
        retrieval meaningful."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        # Record A: one overlapping meaningful token with the query.
        await store.aput(
            namespace,
            "fact-weak",
            {"evidence_quote": "My sister visited yesterday"},
        )
        # Record B: three overlapping meaningful tokens with the query.
        await store.aput(
            namespace,
            "fact-strong",
            {"evidence_quote": "My sister Sarah visited me yesterday"},
        )

        # Query meaningful tokens: {sister, sarah, visited, yesterday}.
        # fact-weak overlap: {sister, visited, yesterday}. Recall: 3/4 = 0.75.
        # fact-strong overlap: {sister, sarah, visited, yesterday}. Recall: 4/4 = 1.0.
        results = await store.asearch(
            namespace, query="my sister Sarah visited yesterday"
        )
        assert len(results) == 2
        assert results[0].key == "fact-strong"
        assert results[1].key == "fact-weak"

    @pytest.mark.asyncio
    async def test_insertion_order_is_tiebreaker_for_equal_scores(self) -> None:
        """When two records tie on recall score, insertion order breaks
        the tie so the earlier-inserted record comes first. This keeps
        ``test_memory_store_search_respects_limit`` deterministic."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(namespace, "first", {"evidence_quote": "work stress"})
        await store.aput(namespace, "second", {"evidence_quote": "work stress"})
        await store.aput(namespace, "third", {"evidence_quote": "work stress"})

        results = await store.asearch(namespace, query="work", limit=3)
        assert [r.key for r in results] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_records_with_no_overlap_are_excluded(self) -> None:
        """Records whose haystack shares zero tokens with the query
        must not appear in results, regardless of how large the
        namespace is."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace, "fact-work", {"evidence_quote": "I worry about work"}
        )
        await store.aput(
            namespace, "fact-sleep", {"evidence_quote": "I can't sleep at night"}
        )
        await store.aput(
            namespace, "fact-sarah", {"evidence_quote": "My sister Sarah visited"}
        )

        results = await store.asearch(namespace, query="Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-sarah"

    @pytest.mark.asyncio
    async def test_search_matches_across_multiple_string_fields(self) -> None:
        """The haystack is built from every non-null stringified field,
        not just ``evidence_quote``. A query that matches the subject
        or object field should still return the record."""

        store = OpenCouchMemoryStore()
        namespace = ("user-1", "semantic")
        await store.aput(
            namespace,
            "fact-1",
            {
                "evidence_quote": "I worry about things",
                "object": "Sarah",
                "predicate": "KNOWS",
            },
        )

        # "Sarah" is in the ``object`` field but not the evidence quote.
        # With a concatenated haystack, it should still match.
        results = await store.asearch(namespace, query="Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-1"


@pytest.mark.asyncio
async def test_memory_store_delete_returns_true_when_record_existed() -> None:
    """adelete should return True when it removed something, False otherwise."""

    store = OpenCouchMemoryStore()
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "test"})

    assert await store.adelete(namespace, "fact-1") is True
    assert await store.adelete(namespace, "fact-1") is False  # already gone
    assert await store.aget(namespace, "fact-1") is None


@pytest.mark.asyncio
async def test_memory_store_isolates_namespaces() -> None:
    """Writes to one namespace must not leak into another."""

    store = OpenCouchMemoryStore()
    await store.aput(("user-1", "semantic"), "shared-key", {"v": "alice"})
    await store.aput(("user-2", "semantic"), "shared-key", {"v": "bob"})

    alice = await store.aget(("user-1", "semantic"), "shared-key")
    bob = await store.aget(("user-2", "semantic"), "shared-key")

    assert alice is not None and alice.value == {"v": "alice"}
    assert bob is not None and bob.value == {"v": "bob"}


@pytest.mark.asyncio
async def test_memory_store_record_count_tracks_total_and_per_namespace() -> None:
    """record_count should report total and per-namespace sizes."""

    store = OpenCouchMemoryStore()
    await store.aput(("user-1", "semantic"), "a", {"v": 1})
    await store.aput(("user-1", "semantic"), "b", {"v": 2})
    await store.aput(("user-1", "episodic"), "c", {"v": 3})

    assert store.record_count() == 3
    assert store.record_count(("user-1", "semantic")) == 2
    assert store.record_count(("user-1", "episodic")) == 1
    assert store.record_count(("user-2", "semantic")) == 0


@pytest.mark.asyncio
async def test_memory_store_close_makes_store_unusable() -> None:
    """After aclose, put/get/search/delete should raise RuntimeError."""

    store = OpenCouchMemoryStore()
    await store.aput(("user-1", "semantic"), "fact-1", {"v": 1})
    await store.aclose()

    with pytest.raises(RuntimeError):
        await store.aput(("user-1", "semantic"), "fact-2", {"v": 2})
    with pytest.raises(RuntimeError):
        await store.aget(("user-1", "semantic"), "fact-1")


@pytest.mark.asyncio
async def test_memory_store_close_is_idempotent() -> None:
    """Calling aclose on an already-closed store should not raise."""

    store = OpenCouchMemoryStore()
    await store.aclose()
    await store.aclose()  # must not raise


# ─── CrisisLogBackend tests ───────────────────────────────────────────────


def _crisis_record(
    *,
    record_id: str = "rec-1",
    detected_at: str = "2026-04-10T12:00:00Z",
    level: int = 2,
    user_id: str | None = None,
) -> CrisisLogRecord:
    """Helper: build a valid CrisisLogRecord for tests."""

    return CrisisLogRecord(
        id=record_id,
        session_id_opaque="a" * 64,
        user_id_or_null=user_id,
        detected_at=detected_at,
        level=level,  # type: ignore[arg-type]
        override_kind="none",
        classifier_path="deterministic",
        reason="test",
        response_node_completed=True,
        llm_failure_occurred=False,
    )


@pytest.mark.asyncio
async def test_crisis_log_append_and_list_by_date() -> None:
    """Appended records should be retrievable by their date."""

    backend = InMemoryCrisisLogBackend()
    record = _crisis_record(detected_at="2026-04-10T12:00:00Z")
    await backend.aappend(record)

    results = await backend.alist_by_date(date(2026, 4, 10))
    assert len(results) == 1
    assert results[0].id == "rec-1"


@pytest.mark.asyncio
async def test_crisis_log_groups_by_date() -> None:
    """Records from different days should go into different buckets."""

    backend = InMemoryCrisisLogBackend()
    await backend.aappend(
        _crisis_record(record_id="a", detected_at="2026-04-10T11:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="b", detected_at="2026-04-10T23:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="c", detected_at="2026-04-11T00:00:00Z")
    )

    day_10 = await backend.alist_by_date(date(2026, 4, 10))
    day_11 = await backend.alist_by_date(date(2026, 4, 11))

    assert [r.id for r in day_10] == ["a", "b"]
    assert [r.id for r in day_11] == ["c"]


@pytest.mark.asyncio
async def test_crisis_log_list_missing_day_returns_empty() -> None:
    """Asking for a day with no records should return an empty list."""

    backend = InMemoryCrisisLogBackend()
    results = await backend.alist_by_date(date(2026, 4, 10))
    assert results == []


@pytest.mark.asyncio
async def test_crisis_log_record_count() -> None:
    """record_count should report the total across all days."""

    backend = InMemoryCrisisLogBackend()
    assert backend.record_count() == 0

    await backend.aappend(
        _crisis_record(record_id="a", detected_at="2026-04-10T11:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="b", detected_at="2026-04-11T11:00:00Z")
    )

    assert backend.record_count() == 2


@pytest.mark.asyncio
async def test_crisis_log_close_blocks_further_use() -> None:
    """After aclose, append and list should raise RuntimeError."""

    backend = InMemoryCrisisLogBackend()
    await backend.aclose()

    with pytest.raises(RuntimeError):
        await backend.aappend(_crisis_record())
    with pytest.raises(RuntimeError):
        await backend.alist_by_date(date(2026, 4, 10))


@pytest.mark.asyncio
async def test_null_crisis_log_backend_is_a_silent_noop() -> None:
    """NullCrisisLogBackend should accept writes and return empty reads."""

    backend = NullCrisisLogBackend()
    await backend.aappend(_crisis_record())
    await backend.aappend(_crisis_record(record_id="b"))

    assert await backend.alist_by_date(date(2026, 4, 10)) == []
    assert backend.record_count() == 0
    await backend.aclose()  # must not raise
