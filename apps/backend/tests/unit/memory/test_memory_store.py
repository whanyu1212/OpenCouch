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

from agent.audit.crisis_log import (
    InMemoryCrisisLogBackend,
    NullCrisisLogBackend,
)
from agent.memory.models import (
    CrisisLogRecord,
    MoodArc,
    SessionArc,
    StoredSessionArc,
    SummarizationResult,
)
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
        behavior change that makes turn memory context top-k
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

    assert await store.arecord_count() == 3
    assert await store.arecord_count(("user-1", "semantic")) == 2
    assert await store.arecord_count(("user-1", "episodic")) == 1
    assert await store.arecord_count(("user-2", "semantic")) == 0


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


# ─── v0.4 episodic namespace model + store tests ────────────────────────
#
# These tests cover the Stage A scope of v0.4: the pydantic models for
# session summaries (SessionArc, StoredSessionArc, MoodArc,
# SummarizationResult) and their round-trip through the existing store
# interface. The store itself is namespace-agnostic — it treats the
# episodic namespace identically to semantic — so these tests verify
# that the SHAPES land cleanly rather than any new store code.


def _make_session_arc(
    *,
    session_id: str = "session-test",
    started_at: str = "2026-04-10T12:00:00Z",
    ended_at: str = "2026-04-10T12:30:00Z",
    duration_seconds: int = 1800,
    turn_count: int = 8,
    primary_themes: list[str] | None = None,
    summary: str = "User talked about work anxiety around an upcoming meeting.",
    opened: str = "anxious",
    closed: str = "calmer",
    open_loops: list[str] | None = None,
    resolved_threads: list[str] | None = None,
) -> SessionArc:
    """Build a SessionArc with sensible defaults for testing.

    Note the ``primary_themes is None`` check rather than a falsy-default
    — an explicit empty list ``[]`` should be respected (the test for
    empty themes depends on this).

    ``crisis_level_max`` is NOT a SessionArc field (v0.4 refactor).
    Callers that need a non-zero crisis level on the STORED shape
    should pass it directly to ``_make_stored_session_arc`` instead.
    """

    return SessionArc(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        turn_count=turn_count,
        primary_themes=(
            primary_themes if primary_themes is not None else ["work stress"]
        ),
        summary=summary,
        mood_arc=MoodArc(opened=opened, closed=closed),
        open_loops=open_loops if open_loops is not None else [],
        resolved_threads=resolved_threads if resolved_threads is not None else [],
    )


def _make_stored_session_arc(
    *,
    owner_id: str = "user-1",
    record_id: str = "arc-1",
    crisis_level_max: int = 0,
    **kwargs,
) -> StoredSessionArc:
    """Build a StoredSessionArc with store-layer metadata populated.

    ``crisis_level_max`` is a StoredSessionArc-only field (v0.4 refactor)
    — it's runtime-computed from per-turn crisis gate verdicts, not
    LLM-produced. Tests that want a non-zero level pass it here
    directly; it's applied during the stored-shape promotion rather
    than coming through the underlying SessionArc.
    """

    arc = _make_session_arc(**kwargs)
    return StoredSessionArc(
        **arc.model_dump(),
        id=record_id,
        owner_id=owner_id,
        created_at="2026-04-10T12:30:00Z",
        last_referenced_at="2026-04-10T12:30:00Z",
        user_visible=True,
        crisis_level_max=crisis_level_max,  # type: ignore[arg-type]
    )


class TestEpisodicModels:
    """Unit tests for the v0.4 episodic pydantic models."""

    def test_mood_arc_requires_opened_and_closed(self) -> None:
        """MoodArc is a dead-simple pair of mood descriptor strings."""

        arc = MoodArc(opened="anxious", closed="calmer")
        assert arc.opened == "anxious"
        assert arc.closed == "calmer"

    def test_session_arc_round_trips_through_json(self) -> None:
        """The SessionArc shape must be JSON-serializable for the store."""

        arc = _make_session_arc()
        dumped = arc.model_dump(mode="json")
        restored = SessionArc.model_validate(dumped)
        assert restored.session_id == arc.session_id
        assert restored.summary == arc.summary
        assert restored.mood_arc.opened == arc.mood_arc.opened
        assert restored.primary_themes == arc.primary_themes

    def test_stored_session_arc_adds_store_metadata(self) -> None:
        """StoredSessionArc is SessionArc + id + owner_id + timestamps."""

        stored = _make_stored_session_arc()
        assert stored.id == "arc-1"
        assert stored.owner_id == "user-1"
        assert stored.created_at == "2026-04-10T12:30:00Z"
        assert stored.last_referenced_at == "2026-04-10T12:30:00Z"
        assert stored.user_visible is True
        # Inherited fields from SessionArc still work:
        assert stored.session_id == "session-test"
        assert stored.turn_count == 8

    def test_session_arc_rejects_too_many_primary_themes(self) -> None:
        """primary_themes is capped at 3 entries to keep downstream
        filtering and rendering predictable."""

        with pytest.raises(ValueError):
            _make_session_arc(
                primary_themes=["a", "b", "c", "d"],  # 4 — should fail
            )

    def test_session_arc_allows_empty_primary_themes(self) -> None:
        """Some sessions genuinely have no dominant theme (e.g., small
        talk that produced no meaningful arc). Empty list is allowed."""

        arc = _make_session_arc(primary_themes=[])
        assert arc.primary_themes == []

    def test_summarization_result_with_arc(self) -> None:
        """The happy path: LLM returned a valid arc + a reason string."""

        arc = _make_session_arc()
        result = SummarizationResult(
            arc=arc,
            reason="summarized 8 turns into a work-anxiety arc",
        )
        assert result.arc is not None
        assert result.arc.session_id == "session-test"
        assert "work-anxiety" in result.reason

    def test_summarization_result_with_no_arc(self) -> None:
        """A session with nothing worth summarizing returns arc=None
        with a reason explaining why. This is the analog of
        ExtractionResult.facts == []."""

        result = SummarizationResult(
            arc=None,
            reason="session too short (2 turns, small talk only)",
        )
        assert result.arc is None
        assert "too short" in result.reason

    def test_summarization_result_requires_reason(self) -> None:
        """The reason field is always required, even when arc is None.
        This is an observability contract: the LLM must explain itself."""

        with pytest.raises(ValueError):
            SummarizationResult(arc=None, reason="")  # empty reason rejected


# ─── v0.4 episodic store round-trip tests ──────────────────────────────


@pytest.mark.asyncio
async def test_episodic_round_trips_through_store() -> None:
    """A StoredSessionArc written via aput should come back via aget
    with all its fields intact. The store is namespace-agnostic, so
    this tests the serialize/deserialize round-trip for the new type."""

    store = OpenCouchMemoryStore()
    arc = _make_stored_session_arc(
        owner_id="user-42",
        record_id="arc-2026-04-10",
        summary="Discussed work deadlines and upcoming supervisor meeting.",
        primary_themes=["work stress", "performance anxiety"],
        open_loops=["still need to prep slides"],
        crisis_level_max=1,
    )
    namespace = ("user-42", "episodic")

    await store.aput(namespace, arc.id, arc.model_dump(mode="json"))
    record = await store.aget(namespace, arc.id)

    assert record is not None
    # Reconstruct the full pydantic model from the stored dict.
    restored = StoredSessionArc.model_validate(record.value)
    assert restored.session_id == arc.session_id
    assert restored.summary == arc.summary
    assert restored.primary_themes == ["work stress", "performance anxiety"]
    assert restored.open_loops == ["still need to prep slides"]
    assert restored.crisis_level_max == 1
    assert restored.owner_id == "user-42"


@pytest.mark.asyncio
async def test_episodic_namespace_isolated_from_semantic() -> None:
    """The episodic and semantic namespaces are separate buckets even
    for the same user. A write to one must not appear in searches of
    the other."""

    store = OpenCouchMemoryStore()
    # Semantic fact and episodic arc, same user
    await store.aput(
        ("user-1", "semantic"),
        "fact-1",
        {"evidence_quote": "I have a sister named Sarah"},
    )
    arc = _make_stored_session_arc(owner_id="user-1", record_id="arc-1")
    await store.aput(("user-1", "episodic"), arc.id, arc.model_dump(mode="json"))

    # Semantic namespace has 1 record, episodic has 1
    assert await store.arecord_count(("user-1", "semantic")) == 1
    assert await store.arecord_count(("user-1", "episodic")) == 1
    assert await store.arecord_count() == 2  # total across all namespaces


@pytest.mark.asyncio
async def test_episodic_asearch_uses_same_token_recall_scorer() -> None:
    """The store's asearch method is namespace-agnostic, so token-recall
    scoring works on episodic summaries identically to semantic facts.
    A query that hits tokens in the summary text should surface the arc."""

    store = OpenCouchMemoryStore()
    arc = _make_stored_session_arc(
        owner_id="user-1",
        record_id="arc-1",
        summary="User talked about work anxiety around an upcoming meeting.",
    )
    await store.aput(("user-1", "episodic"), arc.id, arc.model_dump(mode="json"))

    # Query with tokens that overlap the summary.
    # Meaningful query tokens: {work, anxiety, meeting}
    # Haystack contains all three; recall = 3/3 = 1.0 → hit.
    results = await store.asearch(
        ("user-1", "episodic"),
        query="tell me about work anxiety and the meeting",
    )
    assert len(results) == 1
    assert results[0].key == "arc-1"


@pytest.mark.asyncio
async def test_episodic_asearch_respects_threshold() -> None:
    """An off-topic query against episodic should miss, same as semantic.
    This guards against "catch-up" logic accidentally leaking episodic
    records into unrelated retrieval contexts."""

    store = OpenCouchMemoryStore()
    arc = _make_stored_session_arc(
        owner_id="user-1",
        record_id="arc-1",
        summary="User discussed grief about a recent loss.",
    )
    await store.aput(("user-1", "episodic"), arc.id, arc.model_dump(mode="json"))

    # Totally off-topic query, no meaningful overlap with "grief" or "loss".
    results = await store.asearch(
        ("user-1", "episodic"),
        query="what movies should I watch this weekend",
    )
    assert results == []


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
        classifier_path="llm_primary",
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
    assert await backend.arecord_count() == 0

    await backend.aappend(
        _crisis_record(record_id="a", detected_at="2026-04-10T11:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="b", detected_at="2026-04-11T11:00:00Z")
    )

    assert await backend.arecord_count() == 2


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
    assert await backend.arecord_count() == 0
    await backend.aclose()  # must not raise
