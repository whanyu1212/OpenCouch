"""Parity tests for shared memory retrieval behavior across store backends."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.reconciliation import is_active_semantic_record_value
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from tests.support.memory_fixtures import (
    episodic_namespace,
    seed_episodic_arc,
    seed_procedural_profile,
    seed_semantic_fact,
    semantic_namespace,
)


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
async def test_asearch_similar_recalls_short_record_from_wordy_query(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "semantic")

    try:
        await store.aput(
            namespace,
            "fact-sarah",
            {
                "id": "fact-sarah",
                "category": "relationship",
                "subject": {"type": "User", "identifier": "user-1"},
                "predicate": "KNOWS",
                "object": {"type": "Person", "identifier": "Sarah"},
                "evidence_quote": "My sister Sarah helps when panic starts.",
                "confidence": "high",
                "source_session_id": "seed-session",
                "source_turn_index": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "last_referenced_at": "2026-01-01T00:00:00Z",
                "dormant_at": None,
                "superseded_by": None,
                "user_visible": True,
                "write_reason": "seeded test fact",
            },
        )

        results = await store.asearch_similar(
            namespace,
            query_text=(
                "I'm getting that panic feeling again. "
                "Who did I say I reach out to when panic starts?"
            ),
            query_embedding=None,
            limit=10,
        )

        assert [record.key for record in results] == ["fact-sarah"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_asearch_similar_recalls_episodic_summary_from_reminder_query(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    namespace = ("user-1", "episodic")

    try:
        await store.aput(
            namespace,
            "episode-presentation",
            {
                "id": "episode-presentation",
                "owner_id": "user-1",
                "session_id": "presentation-session",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:30:00Z",
                "duration_seconds": 1800,
                "turn_count": 8,
                "primary_themes": [
                    "presentation anxiety",
                    "catastrophic predictions",
                ],
                "summary": (
                    "The user practiced a short presentation run and identified "
                    "a catastrophic prediction about freezing."
                ),
                "mood_arc": {"opened": "tense", "closed": "steadier"},
                "open_loops": [],
                "resolved_threads": [],
                "approach_used": "cbt",
                "approach_context": None,
                "created_at": "2026-01-01T00:30:00Z",
                "last_referenced_at": "2026-01-01T00:30:00Z",
                "user_visible": True,
                "write_reason": "seeded test episode",
                "crisis_level_max": 0,
            },
        )

        results = await store.asearch_similar(
            namespace,
            query_text=(
                "Before I present again, remind me what we worked out about "
                "freezing during the presentation."
            ),
            query_embedding=None,
            limit=10,
        )

        assert [record.key for record in results] == ["episode-presentation"]
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_memory_fixture_seeds_semantic_fact_for_retrieval_parity(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    user_id = "user-fixture"

    try:
        fact = await seed_semantic_fact(
            store,
            user_id,
            "My sister Sarah helps when panic starts.",
            fact_id="fact-sarah",
        )

        results = await store.asearch_similar(
            semantic_namespace(user_id),
            query_text=(
                "I'm getting that panic feeling again. "
                "Who did I say I reach out to when panic starts?"
            ),
            query_embedding=None,
            limit=10,
        )

        assert [record.key for record in results] == [fact.id]
        assert results[0].value["evidence_quote"] == fact.evidence_quote
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_memory_fixture_seeds_episodic_arc_for_retrieval_parity(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    user_id = "user-fixture"

    try:
        arc = await seed_episodic_arc(
            store,
            user_id,
            (
                "The user practiced a short presentation run and identified "
                "a catastrophic prediction about freezing."
            ),
            arc_id="episode-presentation",
            session_id="presentation-session",
            primary_themes=[
                "presentation anxiety",
                "catastrophic predictions",
            ],
            approach_used="cbt",
        )

        results = await store.asearch_similar(
            episodic_namespace(user_id),
            query_text=(
                "Before I present again, remind me what we worked out about "
                "freezing during the presentation."
            ),
            query_embedding=None,
            limit=10,
        )

        assert [record.key for record in results] == [arc.id]
        assert results[0].value["summary"] == arc.summary
    finally:
        await store.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [_make_memory_store, _make_sqlite_store],
    ids=["memory", "sqlite"],
)
async def test_memory_fixture_seeds_procedural_profile_parity(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    user_id = "user-fixture"

    try:
        seeded_profile = await seed_procedural_profile(
            store,
            user_id,
            [
                "Use concise grounding prompts before offering reframes.",
                "Ask before suggesting breathing exercises.",
            ],
            proactive_recall_enabled=True,
            evidence=["The user asked for brief, consent-based support."],
        )
        stored_profile = await aget_procedural_profile(store, user_id=user_id)

        assert stored_profile.proactive_recall_enabled is True
        assert [rule.rule for rule in stored_profile.rules] == [
            "Use concise grounding prompts before offering reframes.",
            "Ask before suggesting breathing exercises.",
        ]
        assert [rule.rule for rule in seeded_profile.rules] == [
            rule.rule for rule in stored_profile.rules
        ]
    finally:
        await store.aclose()
