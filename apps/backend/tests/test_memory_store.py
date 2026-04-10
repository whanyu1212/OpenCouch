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
