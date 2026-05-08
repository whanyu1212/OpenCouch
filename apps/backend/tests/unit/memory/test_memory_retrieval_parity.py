"""Parity tests for shared memory retrieval behavior across store backends."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from agent.memory.reconciliation import is_active_semantic_record_value
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore


def _make_memory_store() -> MemoryStore:
    return OpenCouchMemoryStore()


def _make_sqlite_store() -> MemoryStore:
    return SqliteMemoryStore(":memory:")


def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


StoreFactory = Callable[[], MemoryStore]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_degrades_to_lexical_when_no_query_embedding(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")

    try:
        await store.aput(
            namespace,
            "fact-work",
            {"evidence_quote": "I worry about work stress"},
        )
        await store.aput(
            namespace,
            "fact-sarah",
            {"evidence_quote": "My sister Sarah visited"},
        )

        results = await store.asearch_similar(
            namespace,
            query_text="work stress",
            query_embedding=None,
            limit=10,
        )

        assert [record.key for record in results] == ["fact-work"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_returns_dense_hits_when_lexical_path_misses(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")

    try:
        await store.aput(
            namespace,
            "fact-dense",
            {"evidence_quote": "coffee mug"},
            embedding=[1.0, 0.0],
            embedding_model="test-embed",
        )
        await store.aput(
            namespace,
            "fact-other",
            {"evidence_quote": "desk lamp"},
            embedding=[0.0, 1.0],
            embedding_model="test-embed",
        )

        results = await store.asearch_similar(
            namespace,
            query_text="galaxy orbit",
            query_embedding=[1.0, 0.0],
            embedding_model="test-embed",
            limit=10,
        )

        assert [record.key for record in results] == ["fact-dense"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_prefers_record_ranked_by_both_scorers(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")

    try:
        await store.aput(
            namespace,
            "fact-both",
            {"evidence_quote": "I worry about work stress"},
            embedding=[1.0, 0.0],
            embedding_model="test-embed",
        )
        await store.aput(
            namespace,
            "fact-dense-only",
            {"evidence_quote": "coffee mug"},
            embedding=[1.0, 0.0],
            embedding_model="test-embed",
        )

        results = await store.asearch_similar(
            namespace,
            query_text="work stress",
            query_embedding=[1.0, 0.0],
            embedding_model="test-embed",
            limit=10,
        )

        assert [record.key for record in results[:2]] == [
            "fact-both",
            "fact-dense-only",
        ]
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_respects_max_age_days(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")
    now = datetime.now(UTC)

    try:
        await store.aput(
            namespace,
            "recent",
            {
                "evidence_quote": "I worry about work stress",
                "created_at": _to_utc_z(now - timedelta(days=1)),
            },
        )
        await store.aput(
            namespace,
            "old",
            {
                "evidence_quote": "I worry about work stress",
                "created_at": _to_utc_z(now - timedelta(days=30)),
            },
        )

        results = await store.asearch_similar(
            namespace,
            query_text="work stress",
            query_embedding=None,
            limit=10,
            max_age_days=7,
        )

        assert [record.key for record in results] == ["recent"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_filters_candidates_before_limit_truncation(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")

    try:
        for index in range(25):
            await store.aput(
                namespace,
                f"fact-{index}",
                {
                    "evidence_quote": f"My sister moved entry {index}",
                    "user_visible": True,
                    "dormant_at": "2026-04-20T12:00:00Z" if index < 20 else None,
                    "superseded_by": f"fact-new-{index}" if index < 20 else None,
                },
            )

        results = await store.asearch_similar(
            namespace,
            query_text="sister moved",
            query_embedding=None,
            limit=5,
            record_filter=lambda record: is_active_semantic_record_value(record.value),
        )

        assert [record.key for record in results] == [
            "fact-20",
            "fact-21",
            "fact-22",
            "fact-23",
            "fact-24",
        ]
    finally:
        await store.aclose()
