"""Integration tests for the PostgreSQL-backed memory store."""

from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest

from agent.memory.store import MemoryStore, StoreRecord
from agent.memory.store.postgres import PostgresMemoryStore
from tests.support.persistence import postgres_database_url


def _require_postgres_database_url() -> str:
    """Return the enabled Postgres DSN or skip the test."""

    dsn = postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration tests are disabled; set "
            "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS=1 and "
            "OPENCOUCH_TEST_POSTGRES_URL"
        )
    return dsn


async def _delete_records_for_owners(dsn: str, owner_ids: list[str]) -> None:
    """Delete test-owned memory rows from the shared Postgres table."""

    if not owner_ids:
        return
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM memory_records WHERE owner_id = ANY(%s)",
                (owner_ids,),
            )


def _owner_id() -> str:
    """Return a unique owner id for one test namespace cohort."""

    return f"memory-store-test-{uuid4()}"


def _embedding_at(index: int, dimension: int = 3072) -> list[float]:
    """Return a unit-like embedding vector with one activated coordinate."""

    embedding = [0.0] * dimension
    embedding[index] = 1.0
    return embedding


@pytest.mark.asyncio
async def test_put_and_get_round_trip() -> None:
    """A record stored with aput should come back via aget."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})

        record = await store.aget(namespace, "fact-1")

        assert record is not None
        assert isinstance(record, StoreRecord)
        assert record.namespace == namespace
        assert record.key == "fact-1"
        assert record.value == {"evidence_quote": "I worry about work"}
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_get_missing_key_returns_none() -> None:
    """Fetching a non-existent key should return None, not raise."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        record = await store.aget((owner_id, "semantic"), "nope")
        assert record is None
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_put_overwrites_existing_record() -> None:
    """A second put to the same key replaces the first record value."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(namespace, "fact-1", {"evidence_quote": "first"})
        await store.aput(namespace, "fact-1", {"evidence_quote": "second"})

        record = await store.aget(namespace, "fact-1")
        assert record is not None
        assert record.value["evidence_quote"] == "second"
        assert await store.arecord_count(namespace) == 1
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_get_enforces_namespace_isolation() -> None:
    """A shared key in different namespace kinds must remain isolated."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_id, "semantic"), "shared-key", {"v": "alice-semantic"})
        await store.aput((owner_id, "episodic"), "shared-key", {"v": "alice-episodic"})

        semantic_record = await store.aget((owner_id, "semantic"), "shared-key")
        episodic_record = await store.aget((owner_id, "episodic"), "shared-key")

        assert semantic_record is not None
        assert episodic_record is not None
        assert semantic_record.value == {"v": "alice-semantic"}
        assert episodic_record.value == {"v": "alice-episodic"}
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_delete_returns_true_when_record_existed() -> None:
    """adelete should return True when it removed something, False otherwise."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(namespace, "fact-1", {"evidence_quote": "test"})

        assert await store.adelete(namespace, "fact-1") is True
        assert await store.adelete(namespace, "fact-1") is False
        assert await store.aget(namespace, "fact-1") is None
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_namespaces_isolated_across_users() -> None:
    """Writes to one user's namespace must not leak into another's."""

    dsn = _require_postgres_database_url()
    owner_a = _owner_id()
    owner_b = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_a, "semantic"), "shared-key", {"v": "alice"})
        await store.aput((owner_b, "semantic"), "shared-key", {"v": "bob"})

        alice = await store.aget((owner_a, "semantic"), "shared-key")
        bob = await store.aget((owner_b, "semantic"), "shared-key")

        assert alice is not None and alice.value == {"v": "alice"}
        assert bob is not None and bob.value == {"v": "bob"}
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_a, owner_b])


@pytest.mark.asyncio
async def test_record_count_tracks_total_and_per_namespace() -> None:
    """record_count should report total and per-namespace sizes."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_id, "semantic"), "a", {"v": 1})
        await store.aput((owner_id, "semantic"), "b", {"v": 2})
        await store.aput((owner_id, "episodic"), "c", {"v": 3})

        assert await store.arecord_count() >= 3
        assert await store.arecord_count((owner_id, "semantic")) == 2
        assert await store.arecord_count((owner_id, "episodic")) == 1
        assert await store.arecord_count((_owner_id(), "semantic")) == 0
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_namespaces_returns_only_populated_ones() -> None:
    """namespaces should list exactly the namespaces populated by this test."""

    dsn = _require_postgres_database_url()
    owner_a = _owner_id()
    owner_b = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_a, "semantic"), "a", {"v": 1})
        await store.aput((owner_b, "episodic"), "b", {"v": 2})

        namespaces = await store.anamespaces()
        owned_namespaces = {
            namespace for namespace in namespaces if namespace[0] in {owner_a, owner_b}
        }

        assert owned_namespaces == {
            (owner_a, "semantic"),
            (owner_b, "episodic"),
        }
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_a, owner_b])


@pytest.mark.asyncio
async def test_search_with_none_query_returns_all_in_insertion_order() -> None:
    """Passing query=None should return all records up to the limit in insertion order."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        for i in range(15):
            await store.aput(namespace, f"fact-{i}", {"evidence_quote": f"Fact {i}"})

        results = await store.asearch(namespace, query=None, limit=10)
        assert len(results) == 10
        assert [record.key for record in results] == [f"fact-{i}" for i in range(10)]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_search_empty_namespace_returns_empty_list() -> None:
    """Searching an unknown namespace should return an empty list."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        results = await store.asearch((owner_id, "semantic"), query="anything")
        assert results == []
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_short_named_entity_query_hits() -> None:
    """Single-word named-entity query should find the stored fact."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        results = await store.asearch(namespace, query="Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-sarah"
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_majority_overlap_paraphrase_hits() -> None:
    """Paraphrase with majority token overlap should hit via recall scoring."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        results = await store.asearch(namespace, query="I have a sister named Sarah")
        assert len(results) == 1
        assert results[0].key == "fact-sarah"
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_long_paraphrase_below_threshold_misses() -> None:
    """A long query below the lexical threshold should miss."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        results = await store.asearch(
            namespace, query="Things have been tense with Sarah lately"
        )
        assert results == []
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_stopword_only_query_returns_empty() -> None:
    """A stopword-only query should not flood the namespace."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(namespace, "fact-1", {"evidence_quote": "I worry about work"})

        results = await store.asearch(namespace, query="I am so the")
        assert results == []
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_results_ordered_by_recall_score_desc() -> None:
    """Multiple matches should be returned in descending recall-score order."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
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

        results = await store.asearch(
            namespace, query="my sister Sarah visited yesterday"
        )
        assert len(results) == 2
        assert results[0].key == "fact-strong"
        assert results[1].key == "fact-weak"
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_insertion_order_is_tiebreaker_for_equal_scores() -> None:
    """When records tie on recall score, insertion order should break the tie."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(namespace, "first", {"evidence_quote": "work stress"})
        await store.aput(namespace, "second", {"evidence_quote": "work stress"})
        await store.aput(namespace, "third", {"evidence_quote": "work stress"})

        results = await store.asearch(namespace, query="work", limit=3)
        assert [record.key for record in results] == ["first", "second", "third"]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    """asearch with query should respect the limit parameter."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        for i in range(5):
            await store.aput(
                namespace,
                f"fact-{i}",
                {"evidence_quote": "worry worry worry"},
            )

        results = await store.asearch(namespace, query="worry", limit=3)
        assert len(results) == 3
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_hybrid_search_returns_dense_match_when_lexical_query_is_weak() -> None:
    """Dense retrieval should recover a strong embedding match even without lexical overlap."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(
            namespace,
            "fact-dense-hit",
            {"evidence_quote": "I love hiking in the mountains"},
            embedding=_embedding_at(0),
            embedding_model="text-embedding-3-large",
        )
        await store.aput(
            namespace,
            "fact-dense-miss",
            {"evidence_quote": "I prefer tea in the morning"},
            embedding=_embedding_at(1),
            embedding_model="text-embedding-3-large",
        )

        results = await store.asearch_similar(
            namespace,
            query_text="completely unrelated words",
            query_embedding=_embedding_at(0),
            embedding_model="text-embedding-3-large",
            limit=2,
        )

        assert results
        assert results[0].key == "fact-dense-hit"
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_close_makes_store_unusable() -> None:
    """After aclose, mutating and fetch methods should raise RuntimeError."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_id, "semantic"), "fact-1", {"v": 1})
        await store.aclose()

        with pytest.raises(RuntimeError):
            await store.aput((owner_id, "semantic"), "fact-2", {"v": 2})
        with pytest.raises(RuntimeError):
            await store.aget((owner_id, "semantic"), "fact-1")
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """Calling aclose on an already-closed store should not raise."""

    dsn = _require_postgres_database_url()
    store = PostgresMemoryStore(dsn)

    await store.aclose()
    await store.aclose()


@pytest.mark.asyncio
async def test_record_count_returns_zero_when_closed() -> None:
    """After close, helper methods should return safe defaults rather than raising."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput((owner_id, "semantic"), "fact-1", {"v": 1})
        await store.aclose()

        assert await store.arecord_count() == 0
        assert await store.anamespaces() == []
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_persists_across_close_and_reopen() -> None:
    """Records written through one store instance should survive reopen."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    backend_a = PostgresMemoryStore(dsn)

    try:
        await backend_a.aput(
            (owner_id, "semantic"),
            "fact-sarah",
            {
                "evidence_quote": "I have a sister named Sarah",
                "category": "relationship",
            },
        )
        await backend_a.aput(
            (owner_id, "episodic"),
            "arc-1",
            {
                "summary": "User talked about work anxiety",
                "turn_count": 8,
            },
        )
        assert await backend_a.arecord_count((owner_id, "semantic")) == 1
        assert await backend_a.arecord_count((owner_id, "episodic")) == 1
        await backend_a.aclose()

        backend_b = PostgresMemoryStore(dsn)
        try:
            sarah = await backend_b.aget((owner_id, "semantic"), "fact-sarah")
            arc = await backend_b.aget((owner_id, "episodic"), "arc-1")

            assert sarah is not None
            assert sarah.value["evidence_quote"] == "I have a sister named Sarah"
            assert sarah.value["category"] == "relationship"

            assert arc is not None
            assert arc.value["summary"] == "User talked about work anxiety"
            assert arc.value["turn_count"] == 8

            assert await backend_b.arecord_count((owner_id, "semantic")) == 1
            assert await backend_b.arecord_count((owner_id, "episodic")) == 1
        finally:
            await backend_b.aclose()
    finally:
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_search_works_after_reopen() -> None:
    """Lexical retrieval should still work after a close/reopen cycle."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    backend_a = PostgresMemoryStore(dsn)

    try:
        await backend_a.aput(
            (owner_id, "semantic"),
            "fact-sarah",
            {"evidence_quote": "I have a sister named Sarah"},
        )
        await backend_a.aclose()

        backend_b = PostgresMemoryStore(dsn)
        try:
            results = await backend_b.asearch((owner_id, "semantic"), query="Sarah")
            assert len(results) == 1
            assert results[0].key == "fact-sarah"
        finally:
            await backend_b.aclose()
    finally:
        await _delete_records_for_owners(dsn, [owner_id])


def test_satisfies_memory_store_protocol() -> None:
    """The class must satisfy the MemoryStore protocol at import time."""

    store = PostgresMemoryStore("postgresql://unused")
    assert isinstance(store, MemoryStore)


@pytest.mark.asyncio
async def test_check_constraint_rejects_invalid_namespace_kind() -> None:
    """The schema CHECK constraint should reject invalid namespace kinds."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    store = PostgresMemoryStore(dsn)

    try:
        with pytest.raises(Exception):  # noqa: B017 - any DB constraint error is fine
            await store.aput(
                (owner_id, "bogus_kind"),  # type: ignore[arg-type]
                "fact-1",
                {"v": 1},
            )
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_unpack_namespace_rejects_wrong_tuple_length() -> None:
    """The namespace tuple must be exactly (owner_id, kind)."""

    dsn = _require_postgres_database_url()
    store = PostgresMemoryStore(dsn)

    try:
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
    finally:
        await store.aclose()
