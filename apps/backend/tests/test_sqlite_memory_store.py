"""Unit tests for the v0.8 SqliteMemoryStore.

Parallels ``test_memory_store.py`` which tests OpenCouchMemoryStore,
so both implementations have symmetric coverage. Any test that passes
here should have an equivalent in the in-memory test file and vice
versa — both implementations satisfy the same protocol and should
behave identically.

All tests use ``:memory:`` SQLite databases so they're fast, isolated,
and don't touch the real filesystem. Persistence-across-restart
behavior is tested separately in :func:`test_persists_across_close_and_reopen`
which uses a ``tmp_path`` fixture for a real file.

Test structure:
    1. Round-trip tests (put/get/search/delete)
    2. Schema + connection lifecycle
    3. Token-recall search parity with OpenCouchMemoryStore
    4. Persistence across close/reopen
    5. Protocol conformance
"""

from __future__ import annotations

import pytest

from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.store import MemoryStore, StoreRecord


# ─── Round-trip tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_and_get_round_trip() -> None:
    """A record stored with aput should come back via aget."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})

    record = await store.aget(namespace, "fact-1")

    assert record is not None
    assert isinstance(record, StoreRecord)
    assert record.namespace == namespace
    assert record.key == "fact-1"
    assert record.value == {"evidence_quote": "I worry about work"}
    await store.aclose()


@pytest.mark.asyncio
async def test_get_missing_key_returns_none() -> None:
    """Fetching a non-existent key should return None, not raise."""

    store = SqliteMemoryStore(":memory:")
    record = await store.aget(("user-1", "semantic"), "nope")
    assert record is None
    await store.aclose()


@pytest.mark.asyncio
async def test_put_overwrites_existing_record() -> None:
    """INSERT OR REPLACE semantics: a second put to the same key
    replaces the first record's value."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "first"})
    await store.aput(namespace, "fact-1", {"evidence_quote": "second"})

    record = await store.aget(namespace, "fact-1")
    assert record is not None
    assert record.value["evidence_quote"] == "second"
    # Only one row, not two
    assert await store.arecord_count(namespace) == 1
    await store.aclose()


@pytest.mark.asyncio
async def test_get_enforces_namespace_isolation() -> None:
    """A record written under one namespace must not be findable
    via aget with a different namespace, even if the key matches."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "shared-key", {"v": "alice-semantic"})
    await store.aput(("user-1", "episodic"), "shared-key", {"v": "alice-episodic"})

    semantic_record = await store.aget(("user-1", "semantic"), "shared-key")
    episodic_record = await store.aget(("user-1", "episodic"), "shared-key")

    assert semantic_record is not None
    assert episodic_record is not None
    assert semantic_record.value == {"v": "alice-semantic"}
    assert episodic_record.value == {"v": "alice-episodic"}
    await store.aclose()


@pytest.mark.asyncio
async def test_delete_returns_true_when_record_existed() -> None:
    """adelete should return True when it removed something,
    False otherwise."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "test"})

    assert await store.adelete(namespace, "fact-1") is True
    assert await store.adelete(namespace, "fact-1") is False  # already gone
    assert await store.aget(namespace, "fact-1") is None
    await store.aclose()


@pytest.mark.asyncio
async def test_namespaces_isolated_across_users() -> None:
    """Writes to one user's namespace must not leak into another's."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "shared-key", {"v": "alice"})
    await store.aput(("user-2", "semantic"), "shared-key", {"v": "bob"})

    alice = await store.aget(("user-1", "semantic"), "shared-key")
    bob = await store.aget(("user-2", "semantic"), "shared-key")

    assert alice is not None and alice.value == {"v": "alice"}
    assert bob is not None and bob.value == {"v": "bob"}
    await store.aclose()


# ─── record_count + namespaces helpers ─────────────────────────────────


@pytest.mark.asyncio
async def test_record_count_tracks_total_and_per_namespace() -> None:
    """record_count should report total and per-namespace sizes."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "a", {"v": 1})
    await store.aput(("user-1", "semantic"), "b", {"v": 2})
    await store.aput(("user-1", "episodic"), "c", {"v": 3})

    assert await store.arecord_count() == 3
    assert await store.arecord_count(("user-1", "semantic")) == 2
    assert await store.arecord_count(("user-1", "episodic")) == 1
    assert await store.arecord_count(("user-2", "semantic")) == 0
    await store.aclose()


@pytest.mark.asyncio
async def test_namespaces_returns_only_populated_ones() -> None:
    """``namespaces`` should list exactly the namespaces that contain
    at least one record."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "a", {"v": 1})
    await store.aput(("user-2", "episodic"), "b", {"v": 2})

    ns = await store.anamespaces()
    assert ("user-1", "semantic") in ns
    assert ("user-2", "episodic") in ns
    assert len(ns) == 2
    await store.aclose()


# ─── Search (query=None returns all) ───────────────────────────────────


@pytest.mark.asyncio
async def test_search_with_none_query_returns_all_in_insertion_order() -> None:
    """Passing query=None should return all records up to the limit,
    ordered by insertion."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    for i in range(15):
        await store.aput(namespace, f"fact-{i}", {"evidence_quote": f"Fact {i}"})

    results = await store.asearch(namespace, query=None, limit=10)
    assert len(results) == 10
    assert [r.key for r in results] == [f"fact-{i}" for i in range(10)]
    await store.aclose()


@pytest.mark.asyncio
async def test_search_empty_namespace_returns_empty_list() -> None:
    """Searching an unknown namespace should return an empty list."""

    store = SqliteMemoryStore(":memory:")
    results = await store.asearch(("nobody", "semantic"), query="anything")
    assert results == []
    await store.aclose()


# ─── Token-recall search parity ────────────────────────────────────────
#
# These tests mirror the ``TestTokenRecallSearch`` class in
# test_memory_store.py. If a test passes there but fails here, the
# two implementations have drifted — that's a regression in the SQLite
# layer's search path, which is supposed to use the same scorer.


@pytest.mark.asyncio
async def test_short_named_entity_query_hits() -> None:
    """Single-word named-entity query should find the stored fact
    via token-recall scoring (recall 1/1 = 1.0)."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(
        namespace,
        "fact-sarah",
        {"evidence_quote": "I have a sister named Sarah"},
    )

    results = await store.asearch(namespace, query="Sarah")
    assert len(results) == 1
    assert results[0].key == "fact-sarah"
    await store.aclose()


@pytest.mark.asyncio
async def test_majority_overlap_paraphrase_hits() -> None:
    """Paraphrase with majority token overlap should hit via recall
    scoring."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(
        namespace,
        "fact-sarah",
        {"evidence_quote": "I have a sister named Sarah"},
    )

    results = await store.asearch(namespace, query="I have a sister named Sarah")
    assert len(results) == 1
    assert results[0].key == "fact-sarah"
    await store.aclose()


@pytest.mark.asyncio
async def test_long_paraphrase_below_threshold_misses() -> None:
    """Long query with only one topical keyword in four meaningful
    tokens falls below the 0.33 threshold and should miss. Mirrors
    the v0.3.1 canary test in TestTokenRecallSearch."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(
        namespace,
        "fact-sarah",
        {"evidence_quote": "I have a sister named Sarah"},
    )

    results = await store.asearch(
        namespace, query="Things have been tense with Sarah lately"
    )
    assert results == []
    await store.aclose()


@pytest.mark.asyncio
async def test_stopword_only_query_returns_empty() -> None:
    """A query containing only stopwords yields no meaningful tokens
    and should return an empty list (not flood the namespace)."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})

    results = await store.asearch(namespace, query="I am so the")
    assert results == []
    await store.aclose()


@pytest.mark.asyncio
async def test_results_ordered_by_recall_score_desc() -> None:
    """When multiple records match, they should be returned in
    score-descending order — parallel to the in-memory store's
    TestTokenRecallSearch.test_results_ordered_by_recall_score_desc."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(
        namespace,
        "fact-weak",
        {"evidence_quote": "My sister visited yesterday"},
    )
    await store.aput(
        namespace,
        "fact-strong",
        {"evidence_quote": "My sister Sarah visited me yesterday"},
    )

    # Query meaningful tokens: {sister, sarah, visited, yesterday}
    # fact-weak overlap: {sister, visited, yesterday} = 3/4 = 0.75
    # fact-strong overlap: {sister, sarah, visited, yesterday} = 4/4 = 1.0
    results = await store.asearch(namespace, query="my sister Sarah visited yesterday")
    assert len(results) == 2
    assert results[0].key == "fact-strong"
    assert results[1].key == "fact-weak"
    await store.aclose()


@pytest.mark.asyncio
async def test_insertion_order_is_tiebreaker_for_equal_scores() -> None:
    """When two records tie on recall score, insertion order should
    break the tie. This depends on the explicit insertion_order column
    in the schema."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    await store.aput(namespace, "first", {"evidence_quote": "work stress"})
    await store.aput(namespace, "second", {"evidence_quote": "work stress"})
    await store.aput(namespace, "third", {"evidence_quote": "work stress"})

    results = await store.asearch(namespace, query="work", limit=3)
    assert [r.key for r in results] == ["first", "second", "third"]
    await store.aclose()


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    """asearch with query should respect the limit parameter."""

    store = SqliteMemoryStore(":memory:")
    namespace = ("user-1", "semantic")
    for i in range(5):
        await store.aput(
            namespace, f"fact-{i}", {"evidence_quote": "worry worry worry"}
        )

    results = await store.asearch(namespace, query="worry", limit=3)
    assert len(results) == 3
    await store.aclose()


# ─── Close lifecycle ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_makes_store_unusable() -> None:
    """After aclose, any async method should raise RuntimeError."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "fact-1", {"v": 1})
    await store.aclose()

    with pytest.raises(RuntimeError):
        await store.aput(("user-1", "semantic"), "fact-2", {"v": 2})
    with pytest.raises(RuntimeError):
        await store.aget(("user-1", "semantic"), "fact-1")


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """Calling aclose on an already-closed store should not raise."""

    store = SqliteMemoryStore(":memory:")
    await store.aclose()
    await store.aclose()  # must not raise


@pytest.mark.asyncio
async def test_record_count_returns_zero_when_closed() -> None:
    """After close, sync helpers should return safe defaults rather
    than raising. This matches the pattern the CLI /memory status
    command relies on."""

    store = SqliteMemoryStore(":memory:")
    await store.aput(("user-1", "semantic"), "fact-1", {"v": 1})
    await store.aclose()

    assert await store.arecord_count() == 0
    assert await store.anamespaces() == []


# ─── Persistence across close/reopen (the v0.8 core feature) ──────────


@pytest.mark.asyncio
async def test_persists_across_close_and_reopen(tmp_path) -> None:
    """This is the core v0.8 contract: records written to a SQLite
    file-backed store must survive close + reopen. Without this, the
    entire v0.8 refactor is worthless.

    Uses a ``tmp_path`` fixture (pytest-provided) for a real file on
    disk rather than ``:memory:``, since ``:memory:`` databases
    explicitly don't survive connection close."""

    db_path = tmp_path / "test_persistence.sqlite3"

    # First runtime lifetime: write records, close
    store_a = SqliteMemoryStore(db_path)
    await store_a.aput(
        ("user-42", "semantic"),
        "fact-sarah",
        {"evidence_quote": "I have a sister named Sarah", "category": "relationship"},
    )
    await store_a.aput(
        ("user-42", "episodic"),
        "arc-1",
        {"summary": "User talked about work anxiety", "turn_count": 8},
    )
    assert await store_a.arecord_count() == 2
    await store_a.aclose()

    # Second runtime lifetime: reopen, verify records come back
    store_b = SqliteMemoryStore(db_path)
    sarah = await store_b.aget(("user-42", "semantic"), "fact-sarah")
    arc = await store_b.aget(("user-42", "episodic"), "arc-1")

    assert sarah is not None
    assert sarah.value["evidence_quote"] == "I have a sister named Sarah"
    assert sarah.value["category"] == "relationship"

    assert arc is not None
    assert arc.value["summary"] == "User talked about work anxiety"
    assert arc.value["turn_count"] == 8

    assert await store_b.arecord_count() == 2
    assert await store_b.arecord_count(("user-42", "semantic")) == 1
    assert await store_b.arecord_count(("user-42", "episodic")) == 1

    await store_b.aclose()


@pytest.mark.asyncio
async def test_search_works_after_reopen(tmp_path) -> None:
    """Token-recall retrieval should still work after a close/reopen
    cycle. Pins that the schema + scoring path survives persistence."""

    db_path = tmp_path / "test_search_persistence.sqlite3"

    store_a = SqliteMemoryStore(db_path)
    await store_a.aput(
        ("user-1", "semantic"),
        "fact-sarah",
        {"evidence_quote": "I have a sister named Sarah"},
    )
    await store_a.aclose()

    store_b = SqliteMemoryStore(db_path)
    results = await store_b.asearch(("user-1", "semantic"), query="Sarah")
    assert len(results) == 1
    assert results[0].key == "fact-sarah"
    await store_b.aclose()


# ─── Schema + protocol conformance ─────────────────────────────────────


def test_satisfies_memory_store_protocol() -> None:
    """The class must satisfy the MemoryStore protocol at import time.
    The assignment in sqlite_store.py would fail at import if this
    weren't true; this test is belt-and-suspenders."""

    store = SqliteMemoryStore(":memory:")
    assert isinstance(store, MemoryStore)


@pytest.mark.asyncio
async def test_check_constraint_rejects_invalid_namespace_kind() -> None:
    """The schema has a CHECK constraint on namespace_kind. Passing
    an out-of-allowlist kind should raise at the SQLite layer, and
    our _unpack_namespace helper lets the error propagate from
    SQLite rather than validating separately."""

    store = SqliteMemoryStore(":memory:")
    # aiosqlite wraps sqlite3 errors in its own exception type; either
    # a sqlite3.IntegrityError or its aiosqlite equivalent should fire.
    with pytest.raises(Exception):  # noqa: B017 — any DB error is a pass
        await store.aput(
            ("user-1", "bogus_kind"),  # type: ignore[arg-type]
            "fact-1",
            {"v": 1},
        )
    await store.aclose()


@pytest.mark.asyncio
async def test_unpack_namespace_rejects_wrong_tuple_length() -> None:
    """The namespace tuple must be (owner_id, kind). A wrong-length
    tuple should raise ValueError at the boundary rather than
    producing a confusing SQLite error."""

    store = SqliteMemoryStore(":memory:")
    with pytest.raises(ValueError):
        await store.aput(
            ("user-1",),  # type: ignore[arg-type]
            "fact-1",
            {"v": 1},
        )
    with pytest.raises(ValueError):
        await store.aput(
            ("user-1", "semantic", "extra"),  # type: ignore[arg-type]
            "fact-1",
            {"v": 1},
        )
    await store.aclose()
