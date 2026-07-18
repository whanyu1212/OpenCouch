"""Integration tests for the PostgreSQL-backed memory store."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import psycopg

from agent.memory.store import MemoryStore, StoreRecord
from agent.memory.store.postgres import (
    MEMORY_BACKFILL_ADVISORY_LOCK_ID,
    MEMORY_SCHEMA_ADVISORY_LOCK_ID,
    PostgresMemoryStore,
)
from tests.support.persistence_contracts import (
    delete_postgres_memory_records_for_owners as _delete_records_for_owners,
    require_postgres_database_url as _require_postgres_database_url,
)


def _owner_id() -> str:
    """Return a unique owner id for one test namespace cohort."""

    return f"memory-store-test-{uuid4()}"


def _embedding_at(index: int, dimension: int = 3072) -> list[float]:
    """Return a unit-like embedding vector with one activated coordinate."""

    embedding = [0.0] * dimension
    embedding[index] = 1.0
    return embedding


async def _wait_for_advisory_lock_waiter(
    blocker: psycopg.AsyncConnection,
    blocker_pid: int,
    waiter_pid: int,
) -> None:
    """Wait until another backend is blocked on a lock held by ``blocker``."""

    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        cursor = await blocker.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_locks AS held
                JOIN pg_locks AS waiting
                    ON waiting.locktype = held.locktype
                    AND waiting.database IS NOT DISTINCT FROM held.database
                    AND waiting.classid IS NOT DISTINCT FROM held.classid
                    AND waiting.objid IS NOT DISTINCT FROM held.objid
                    AND waiting.objsubid IS NOT DISTINCT FROM held.objsubid
                    AND waiting.pid <> held.pid
                WHERE held.pid = %s
                    AND waiting.pid = %s
                    AND held.granted
                    AND NOT waiting.granted
            )
            """,
            (blocker_pid, waiter_pid),
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("No advisory-lock waiter appeared")


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
async def test_batch_rollback_discards_partial_writes() -> None:
    """A failing batch must not persist earlier rows in the transaction."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        with pytest.raises(Exception):  # noqa: B017 - database constraint error
            await store.aput_batch(
                [
                    (namespace, "batch-first", {"v": "must rollback"}, None, None),
                    (
                        (owner_id, "invalid-kind"),
                        "batch-invalid",
                        {"v": "violates namespace_kind CHECK"},
                        None,
                        None,
                    ),
                ]
            )

        assert await store.aget(namespace, "batch-first") is None
        await store.aput(namespace, "after-rollback", {"v": "connection reusable"})
        assert await store.aget(namespace, "after-rollback") is not None
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
async def test_dense_search_filters_candidates_before_limit_truncation() -> None:
    """The declarative filter must run before the bounded dense result limit."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)
    embedding = _embedding_at(0)

    try:
        await store.aput_batch(
            [
                (
                    namespace,
                    f"inactive-{index}",
                    {"evidence_quote": "dense candidate", "user_visible": False},
                    embedding,
                    "text-embedding-3-large",
                )
                for index in range(50)
            ]
            + [
                (
                    namespace,
                    "active-after-candidate-window",
                    {"evidence_quote": "dense candidate", "user_visible": True},
                    embedding,
                    "text-embedding-3-large",
                )
            ]
        )

        results = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=embedding,
            embedding_model="text-embedding-3-large",
            limit=1,
            record_filter="active_semantic",
        )

        assert [record.key for record in results] == ["active-after-candidate-window"]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_dense_search_filters_by_embedding_model() -> None:
    """Dense retrieval must not mix embeddings produced by different models."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)
    embedding = _embedding_at(0)

    try:
        await store.aput(
            namespace,
            "old-model",
            {"evidence_quote": "old model candidate"},
            embedding=embedding,
            embedding_model="embedding-v1",
        )
        await store.aput(
            namespace,
            "current-model",
            {"evidence_quote": "current model candidate"},
            embedding=embedding,
            embedding_model="embedding-v2",
        )
        await store.aput(
            namespace,
            "unknown-model",
            {"evidence_quote": "unknown model candidate"},
            embedding=embedding,
            embedding_model=None,
        )

        unfiltered = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=embedding,
            embedding_model="embedding-v2",
            limit=10,
        )
        filtered = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=embedding,
            embedding_model="embedding-v2",
            limit=10,
            record_filter="active_semantic",
        )

        assert [record.key for record in unfiltered] == ["current-model"]
        assert [record.key for record in filtered] == ["current-model"]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_dense_search_supports_configured_noncanonical_dimension() -> None:
    """Non-3072 providers use stored arrays instead of the fixed vector column."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)

    try:
        await store.aput(
            namespace,
            "custom-dimension",
            {"evidence_quote": "dense candidate"},
            embedding=[1.0, 0.0],
            embedding_model="custom-provider",
        )

        unfiltered = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=[1.0, 0.0],
            embedding_model="custom-provider",
        )
        filtered = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=[1.0, 0.0],
            embedding_model="custom-provider",
            record_filter="active_semantic",
        )

        assert [record.key for record in unfiltered] == ["custom-dimension"]
        assert [record.key for record in filtered] == ["custom-dimension"]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_dense_search_respects_age_filter() -> None:
    """Dense retrieval excludes records older than the requested age window."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)
    embedding = _embedding_at(0)
    now = datetime.now(UTC)

    try:
        await store.aput(
            namespace,
            "old",
            {
                "evidence_quote": "dense candidate",
                "created_at": (now - timedelta(days=30)).isoformat(),
            },
            embedding=embedding,
            embedding_model="text-embedding-3-large",
        )
        await store.aput(
            namespace,
            "recent",
            {
                "evidence_quote": "dense candidate",
                "created_at": (now - timedelta(days=1)).isoformat(),
            },
            embedding=embedding,
            embedding_model="text-embedding-3-large",
        )

        results = await store.asearch_similar(
            namespace,
            query_text="unrelated lexical query",
            query_embedding=embedding,
            embedding_model="text-embedding-3-large",
            limit=10,
            max_age_days=7,
        )

        assert [record.key for record in results] == ["recent"]
    finally:
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_embedding_clears_on_overwrite_and_dense_search_survives_reopen() -> None:
    """Vector metadata persists across reopen and clears on an embedding-free update."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    embedding = _embedding_at(0)
    first_store = PostgresMemoryStore(dsn)

    try:
        await first_store.aput(
            namespace,
            "fact",
            {"evidence_quote": "dense candidate"},
            embedding=embedding,
            embedding_model="text-embedding-3-large",
        )
        await first_store.aclose()

        second_store = PostgresMemoryStore(dsn)
        try:
            results = await second_store.asearch_similar(
                namespace,
                query_text="unrelated lexical query",
                query_embedding=embedding,
                embedding_model="text-embedding-3-large",
            )
            assert [record.key for record in results] == ["fact"]

            await second_store.aput(
                namespace,
                "fact",
                {"evidence_quote": "updated without embedding"},
            )
            record = await second_store.aget(namespace, "fact")
            assert record is not None
            assert record.embedding is None
            assert record.embedding_model is None
            assert (
                await second_store.asearch_similar(
                    namespace,
                    query_text="unrelated lexical query",
                    query_embedding=embedding,
                    embedding_model="text-embedding-3-large",
                )
                == []
            )
        finally:
            await second_store.aclose()
    finally:
        await first_store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_schema_initialization_bulk_backfills_pgvector_column() -> None:
    """A legacy canonical array is cast into pgvector during initialization."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    embedding = _embedding_at(0)
    first_store = PostgresMemoryStore(dsn)

    try:
        await first_store.aput(
            namespace,
            "legacy-vector",
            {"evidence_quote": "dense candidate"},
            embedding=embedding,
            embedding_model="text-embedding-3-large",
        )
        conn = first_store._connection  # noqa: SLF001
        assert conn is not None
        await conn.execute(
            """
            UPDATE memory_records
            SET embedding_vector_3072 = NULL
            WHERE owner_id = %s AND namespace_kind = %s AND id = %s
            """,
            (owner_id, "semantic", "legacy-vector"),
        )
        await first_store.aclose()

        second_store = PostgresMemoryStore(dsn)
        try:
            results = await second_store.asearch_similar(
                namespace,
                query_text="unrelated lexical query",
                query_embedding=embedding,
                embedding_model="text-embedding-3-large",
            )
            assert [record.key for record in results] == ["legacy-vector"]

            second_conn = second_store._connection  # noqa: SLF001
            assert second_conn is not None
            cursor = await second_conn.execute(
                """
                SELECT embedding_vector_3072 IS NOT NULL AS backfilled
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s AND id = %s
                """,
                (owner_id, "semantic", "legacy-vector"),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["backfilled"] is True
        finally:
            await second_store.aclose()
    finally:
        await first_store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_writes_are_visible_across_live_connections() -> None:
    """Autocommitted writes are immediately visible to another store instance."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    writer = PostgresMemoryStore(dsn)
    reader = PostgresMemoryStore(dsn)

    try:
        await writer.aput(namespace, "fact", {"evidence_quote": "visible"})
        record = await reader.aget(namespace, "fact")
        assert record is not None
        assert record.value["evidence_quote"] == "visible"
    finally:
        await writer.aclose()
        await reader.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_read_waits_for_same_connection_transaction_lock() -> None:
    """Reads cannot enter another task's transaction on the shared connection."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    store = PostgresMemoryStore(dsn)
    read_task: asyncio.Task[StoreRecord | None] | None = None
    manual_lock_held = False

    try:
        await store.aput(namespace, "fact", {"value": "committed"})
        await store._write_lock.acquire()  # noqa: SLF001
        manual_lock_held = True
        read_task = asyncio.create_task(store.aget(namespace, "fact"))
        await asyncio.sleep(0)
        assert read_task.done() is False

        store._write_lock.release()  # noqa: SLF001
        manual_lock_held = False
        record = await asyncio.wait_for(read_task, timeout=5)
        assert record is not None
        assert record.value == {"value": "committed"}
    finally:
        if manual_lock_held:
            store._write_lock.release()  # noqa: SLF001
        if read_task is not None and not read_task.done():
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
        await store.aclose()
        await _delete_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_concurrent_writes_across_connections_all_persist() -> None:
    """Independent store connections cannot lose successful writes."""

    dsn = _require_postgres_database_url()
    owner_id = _owner_id()
    namespace = (owner_id, "semantic")
    stores = [PostgresMemoryStore(dsn), PostgresMemoryStore(dsn)]

    try:
        await asyncio.gather(
            *(
                stores[index % len(stores)].aput(
                    namespace,
                    f"fact-{index}",
                    {"value": index},
                )
                for index in range(10)
            )
        )
        assert await stores[0].arecord_count(namespace) == 10
    finally:
        await asyncio.gather(*(store.aclose() for store in stores))
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


@pytest.mark.asyncio
async def test_schema_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed post-schema backfill leaves initialization retryable."""

    store = PostgresMemoryStore(_require_postgres_database_url())
    original_backfill = PostgresMemoryStore._backfill_embedding_vector_3072  # noqa: SLF001
    calls = 0

    async def fail_once(conn) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("forced schema failure")
        await original_backfill(conn)

    monkeypatch.setattr(
        PostgresMemoryStore,
        "_backfill_embedding_vector_3072",
        staticmethod(fail_once),
    )
    try:
        with pytest.raises(RuntimeError, match="forced schema failure"):
            await store.arecord_count()
        assert store._connection is None  # noqa: SLF001
        assert await store.arecord_count() >= 0
        assert calls == 2
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_close_before_first_use_does_not_connect() -> None:
    """Closing a lazy store must not create a database connection."""

    store = PostgresMemoryStore(_require_postgres_database_url())
    assert store._connection is None  # noqa: SLF001
    await store.aclose()
    assert store._connection is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_store_initialization_waits_for_schema_advisory_lock() -> None:
    """Schema setup directly contends on the process-wide advisory lock."""

    dsn = _require_postgres_database_url()
    warmup = PostgresMemoryStore(dsn)
    await warmup.arecord_count()
    await warmup.aclose()

    blocker = await psycopg.AsyncConnection.connect(dsn)
    worker = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    pid_cursor = await blocker.execute("SELECT pg_backend_pid()")
    pid_row = await pid_cursor.fetchone()
    assert pid_row is not None
    blocker_pid = int(pid_row[0])
    worker_pid_cursor = await worker.execute("SELECT pg_backend_pid()")
    worker_pid_row = await worker_pid_cursor.fetchone()
    assert worker_pid_row is not None
    worker_pid = int(worker_pid_row[0])
    await blocker.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (MEMORY_SCHEMA_ADVISORY_LOCK_ID,),
    )
    initialization = asyncio.create_task(PostgresMemoryStore._ensure_schema(worker))  # noqa: SLF001
    try:
        await _wait_for_advisory_lock_waiter(blocker, blocker_pid, worker_pid)
        assert initialization.done() is False

        await blocker.commit()
        await asyncio.wait_for(initialization, timeout=5)
    finally:
        if not initialization.done():
            initialization.cancel()
            await asyncio.gather(initialization, return_exceptions=True)
        await blocker.rollback()
        await blocker.close()
        await worker.close()


@pytest.mark.asyncio
async def test_bulk_backfill_uses_separate_advisory_lock() -> None:
    """Concurrent cold starts cannot run the full-table backfill together."""

    dsn = _require_postgres_database_url()
    warmup = PostgresMemoryStore(dsn)
    await warmup.arecord_count()
    await warmup.aclose()

    blocker = await psycopg.AsyncConnection.connect(dsn)
    worker = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    pid_cursor = await blocker.execute("SELECT pg_backend_pid()")
    pid_row = await pid_cursor.fetchone()
    assert pid_row is not None
    blocker_pid = int(pid_row[0])
    worker_pid_cursor = await worker.execute("SELECT pg_backend_pid()")
    worker_pid_row = await worker_pid_cursor.fetchone()
    assert worker_pid_row is not None
    worker_pid = int(worker_pid_row[0])
    await blocker.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (MEMORY_BACKFILL_ADVISORY_LOCK_ID,),
    )
    backfill = asyncio.create_task(
        PostgresMemoryStore._backfill_embedding_vector_3072(worker)  # noqa: SLF001
    )
    try:
        await _wait_for_advisory_lock_waiter(blocker, blocker_pid, worker_pid)
        assert backfill.done() is False

        await blocker.commit()
        await asyncio.wait_for(backfill, timeout=5)
    finally:
        if not backfill.done():
            backfill.cancel()
            await asyncio.gather(backfill, return_exceptions=True)
        await blocker.rollback()
        await blocker.close()
        await worker.close()
