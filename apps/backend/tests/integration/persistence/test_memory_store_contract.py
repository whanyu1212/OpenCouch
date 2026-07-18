"""Backend-neutral contracts for supported memory-store implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest

from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.memory.store.sqlite import SqliteMemoryStore
from tests.support.persistence_contracts import open_postgres_memory_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store_contract(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[tuple[MemoryStore, str]]:
    """Yield each supported ephemeral/durable store with a unique owner."""

    owner_id = f"memory-contract-{uuid4()}"
    if request.param == "memory":
        store = OpenCouchMemoryStore()
        try:
            yield store, owner_id
        finally:
            await store.aclose()
        return

    if request.param == "sqlite":
        store = SqliteMemoryStore(tmp_path / "memory-contract.sqlite3")
        try:
            yield store, owner_id
        finally:
            await store.aclose()
        return

    async with open_postgres_memory_store(
        owner_ids=[owner_id, f"{owner_id}-other"]
    ) as store:
        yield store, owner_id


async def test_batch_round_trip_overwrite_and_namespace_isolation(
    store_contract: tuple[MemoryStore, str],
) -> None:
    """Batch writes preserve metadata and compound namespace identity."""

    store, owner_id = store_contract
    other_owner = f"{owner_id}-other"
    semantic = (owner_id, "semantic")
    episodic = (owner_id, "episodic")

    await store.aput_batch(
        [
            (semantic, "shared", {"value": "semantic"}, [1.0, 0.0], "test-v1"),
            (episodic, "shared", {"value": "episodic"}, None, None),
            (
                (other_owner, "semantic"),
                "shared",
                {"value": "other-owner"},
                None,
                None,
            ),
        ]
    )

    semantic_record = await store.aget(semantic, "shared")
    episodic_record = await store.aget(episodic, "shared")
    other_record = await store.aget((other_owner, "semantic"), "shared")

    assert semantic_record is not None
    assert semantic_record.value == {"value": "semantic"}
    assert semantic_record.embedding == [1.0, 0.0]
    assert semantic_record.embedding_model == "test-v1"
    assert episodic_record is not None
    assert episodic_record.value == {"value": "episodic"}
    assert other_record is not None
    assert other_record.value == {"value": "other-owner"}

    await store.aput(semantic, "shared", {"value": "updated"})
    updated = await store.aget(semantic, "shared")
    assert updated is not None
    assert updated.value == {"value": "updated"}
    assert updated.embedding is None
    assert updated.embedding_model is None
    assert await store.arecord_count(semantic) == 1


async def test_filtering_happens_before_result_truncation(
    store_contract: tuple[MemoryStore, str],
) -> None:
    """Inactive high-ranked rows cannot hide later active memory records."""

    store, owner_id = store_contract
    namespace = (owner_id, "semantic")
    for index in range(25):
        await store.aput(
            namespace,
            f"fact-{index}",
            {
                "evidence_quote": f"My sister moved entry {index}",
                "dormant_at": "2026-01-01T00:00:00Z" if index < 20 else None,
                "superseded_by": f"replacement-{index}" if index < 20 else None,
                "user_visible": True,
            },
        )

    results = await store.asearch_similar(
        namespace,
        query_text="sister moved",
        query_embedding=None,
        limit=5,
        record_filter="active_semantic",
    )

    assert [record.key for record in results] == [
        "fact-20",
        "fact-21",
        "fact-22",
        "fact-23",
        "fact-24",
    ]


async def test_dense_filtering_happens_before_candidate_truncation(
    store_contract: tuple[MemoryStore, str],
) -> None:
    """Inactive dense hits cannot hide a later active semantic record."""

    store, owner_id = store_contract
    namespace = (owner_id, "semantic")
    embedding = [0.0] * 3072
    embedding[0] = 1.0
    await store.aput_batch(
        [
            (
                namespace,
                f"inactive-{index}",
                {
                    "evidence_quote": "dense candidate",
                    "user_visible": False,
                },
                embedding,
                "text-embedding-3-large",
            )
            for index in range(50)
        ]
        + [
            (
                namespace,
                "active-after-window",
                {
                    "evidence_quote": "dense candidate",
                    "user_visible": True,
                },
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

    assert [record.key for record in results] == ["active-after-window"]


@pytest.mark.parametrize(
    ("result_limit", "dense_window"),
    [(1, 50), (20, 100)],
)
async def test_dense_candidate_window_is_shared_before_rrf(
    store_contract: tuple[MemoryStore, str],
    result_limit: int,
    dense_window: int,
) -> None:
    """All stores apply the same dense bound before reciprocal-rank fusion."""

    store, owner_id = store_contract
    namespace = (owner_id, "semantic")
    exact_embedding = [0.0] * 3072
    exact_embedding[0] = 1.0
    lower_embedding = [0.0] * 3072
    lower_embedding[0] = 0.6
    lower_embedding[1] = 0.8
    await store.aput_batch(
        [
            (
                namespace,
                f"dense-{index}",
                {"evidence_quote": "dense only"},
                exact_embedding,
                "text-embedding-3-large",
            )
            for index in range(dense_window)
        ]
        + [
            (
                namespace,
                f"lexical-at-dense-{dense_window + 1}",
                {"evidence_quote": "special phrase"},
                lower_embedding,
                "text-embedding-3-large",
            )
        ]
    )

    results = await store.asearch_similar(
        namespace,
        query_text="special phrase",
        query_embedding=exact_embedding,
        embedding_model="text-embedding-3-large",
        limit=result_limit,
    )

    assert results[0].key.startswith("dense-")


async def test_rrf_tie_uses_original_candidate_insertion_order(
    store_contract: tuple[MemoryStore, str],
) -> None:
    """Dense SQL ranking preserves the backend-neutral RRF tiebreaker."""

    store, owner_id = store_contract
    namespace = (owner_id, "semantic")
    dense_first = [0.0] * 3072
    dense_first[0] = 1.0
    lexical_first = [0.0] * 3072
    lexical_first[0] = 0.8
    lexical_first[1] = 0.6
    await store.aput_batch(
        [
            (namespace, "unranked", {"evidence_quote": "noise"}, None, None),
            (
                namespace,
                "lexical-first",
                {"evidence_quote": "alpha beta"},
                lexical_first,
                "text-embedding-3-large",
            ),
            (
                namespace,
                "dense-first",
                {"evidence_quote": "alpha"},
                dense_first,
                "text-embedding-3-large",
            ),
        ]
    )

    results = await store.asearch_similar(
        namespace,
        query_text="alpha beta",
        query_embedding=dense_first,
        embedding_model="text-embedding-3-large",
        limit=2,
    )

    assert [record.key for record in results] == ["lexical-first", "dense-first"]
    assert results[0].embedding == pytest.approx(lexical_first)
    assert results[0].embedding_model == "text-embedding-3-large"


async def test_latest_counts_namespaces_and_delete_contract(
    store_contract: tuple[MemoryStore, str],
) -> None:
    """Observability and deletion helpers preserve their shared semantics."""

    store, owner_id = store_contract
    namespace = (owner_id, "procedural")
    await store.aput(namespace, "first", {"value": 1})
    await store.aput(namespace, "second", {"value": 2})

    latest = await store.alatest(namespace)
    assert latest is not None
    assert latest.key == "second"
    assert await store.arecord_count(namespace) == 2
    assert namespace in await store.anamespaces()
    assert await store.adelete(namespace, "first") is True
    assert await store.adelete(namespace, "first") is False
    assert await store.arecord_count(namespace) == 1


async def test_close_contract(store_contract: tuple[MemoryStore, str]) -> None:
    """Close is idempotent, blocks stateful operations, and clears diagnostics."""

    store, owner_id = store_contract
    namespace = (owner_id, "semantic")
    await store.aput(namespace, "fact", {"value": 1})
    await store.aclose()
    await store.aclose()

    assert await store.arecord_count() == 0
    assert await store.anamespaces() == []
    assert await store.alatest(namespace) is None
    with pytest.raises(RuntimeError, match="closed"):
        await store.aput(namespace, "other", {"value": 2})
    with pytest.raises(RuntimeError, match="closed"):
        await store.aget(namespace, "fact")
    with pytest.raises(RuntimeError, match="closed"):
        await store.asearch(namespace, query=None)
