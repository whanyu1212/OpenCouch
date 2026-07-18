"""Production memory-service contracts against durable Postgres storage."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent.memory.control.operations import delete_memory_target
from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.types import SemanticFact, StoredSessionArc
from tests.support.memory_fixtures import (
    episodic_namespace,
    seed_episodic_arc,
    seed_procedural_profile,
    seed_semantic_fact,
    semantic_namespace,
)
from tests.support.persistence_contracts import (
    delete_postgres_memory_records_for_owners,
    require_postgres_database_url,
)

pytestmark = pytest.mark.asyncio


async def test_all_memory_shapes_survive_close_and_reopen() -> None:
    """Validated semantic, episodic, and procedural records remain readable."""

    dsn = require_postgres_database_url()
    owner_id = f"memory-services-{uuid4()}"
    first_store = PostgresMemoryStore(dsn)

    try:
        fact = await seed_semantic_fact(
            first_store,
            owner_id,
            "My sister Sarah helps when panic starts.",
        )
        arc = await seed_episodic_arc(
            first_store,
            owner_id,
            "We identified a presentation fear and practiced a short reframe.",
            primary_themes=["presentation anxiety"],
        )
        await seed_procedural_profile(
            first_store,
            owner_id,
            ["Ask before suggesting breathing exercises."],
            proactive_recall_enabled=True,
        )
        await first_store.aclose()

        second_store = PostgresMemoryStore(dsn)
        try:
            semantic_record = await second_store.aget(
                semantic_namespace(owner_id), fact.id
            )
            episodic_record = await second_store.aget(
                episodic_namespace(owner_id), arc.id
            )
            profile = await aget_procedural_profile(
                second_store,
                user_id=owner_id,
            )

            assert semantic_record is not None
            assert SemanticFact.model_validate(semantic_record.value) == fact
            assert episodic_record is not None
            assert StoredSessionArc.model_validate(episodic_record.value) == arc
            assert profile.proactive_recall_enabled is True
            assert [rule.rule for rule in profile.rules] == [
                "Ask before suggesting breathing exercises."
            ]
        finally:
            await second_store.aclose()
    finally:
        await first_store.aclose()
        await delete_postgres_memory_records_for_owners(dsn, [owner_id])


async def test_owner_scoped_deletion_survives_reopen_for_all_memory_shapes() -> None:
    """Product deletion removes only the selected owner's durable memories."""

    dsn = require_postgres_database_url()
    owner_id = f"memory-delete-{uuid4()}"
    other_owner_id = f"memory-delete-other-{uuid4()}"
    store = PostgresMemoryStore(dsn)

    try:
        fact = await seed_semantic_fact(
            store,
            owner_id,
            "My sister Sarah helps when panic starts.",
            fact_id="shared-fact",
        )
        arc = await seed_episodic_arc(
            store,
            owner_id,
            "We practiced grounding before a presentation.",
            arc_id="shared-arc",
        )
        profile = await seed_procedural_profile(
            store,
            owner_id,
            ["Use concise grounding prompts."],
        )
        await seed_semantic_fact(
            store,
            other_owner_id,
            "My sister Sarah helps when panic starts.",
            fact_id="shared-fact",
        )
        await seed_episodic_arc(
            store,
            other_owner_id,
            "We practiced grounding before a presentation.",
            arc_id="shared-arc",
        )
        await seed_procedural_profile(
            store,
            other_owner_id,
            ["Use concise grounding prompts."],
        )

        assert (
            await delete_memory_target(
                store,
                owner_id=owner_id,
                target={
                    "kind": "fact",
                    "namespace": [other_owner_id, "semantic"],
                    "key": fact.id,
                    "rule_id": None,
                    "preview": "wrong owner",
                },
            )
            is False
        )
        assert await delete_memory_target(
            store,
            owner_id=owner_id,
            target={
                "kind": "fact",
                "namespace": [owner_id, "semantic"],
                "key": fact.id,
                "rule_id": None,
                "preview": fact.evidence_quote,
            },
        )
        assert await delete_memory_target(
            store,
            owner_id=owner_id,
            target={
                "kind": "session",
                "namespace": [owner_id, "episodic"],
                "key": arc.id,
                "rule_id": None,
                "preview": arc.summary,
            },
        )
        assert await delete_memory_target(
            store,
            owner_id=owner_id,
            target={
                "kind": "rule",
                "namespace": [owner_id, "procedural"],
                "key": profile.rules[0].id,
                "rule_id": profile.rules[0].id,
                "preview": profile.rules[0].rule,
            },
        )
        await store.aclose()

        reopened = PostgresMemoryStore(dsn)
        try:
            assert await reopened.aget(semantic_namespace(owner_id), fact.id) is None
            assert await reopened.aget(episodic_namespace(owner_id), arc.id) is None
            assert (
                await aget_procedural_profile(reopened, user_id=owner_id)
            ).rules == []

            assert (
                await reopened.aget(
                    semantic_namespace(other_owner_id),
                    "shared-fact",
                )
                is not None
            )
            assert (
                await reopened.aget(
                    episodic_namespace(other_owner_id),
                    "shared-arc",
                )
                is not None
            )
            assert (
                len(
                    (
                        await aget_procedural_profile(
                            reopened,
                            user_id=other_owner_id,
                        )
                    ).rules
                )
                == 1
            )
        finally:
            await reopened.aclose()
    finally:
        await store.aclose()
        await delete_postgres_memory_records_for_owners(
            dsn,
            [owner_id, other_owner_id],
        )
