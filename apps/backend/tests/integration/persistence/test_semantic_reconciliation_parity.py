"""Backend parity tests for semantic reconciliation behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from agent.memory.operations.semantic_writes import (
    BatchWriteItem,
    apply_semantic_writes_batch,
    memory_write_to_semantic_fact,
    write_new_semantic_fact,
)
from agent.memory.policy.candidates import build_semantic_candidate
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.types import EntityRef, MemoryWrite
from llm.base import BaseLLMClient, StructuredResponseT
from tests.support.persistence_contracts import (
    delete_postgres_memory_records_for_owners,
    require_postgres_database_url,
)


class _SupersedeReconciliationLLM(BaseLLMClient):
    """Fake classifier that always supersedes the first collision record."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by reconciliation.")

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        response: dict[str, Any] = {
            "action": "supersede",
            "record_indexes": [0],
            "reason": "the new fact replaces the stale one",
            "confidence": "high",
        }
        return response_schema(**response)


class _CoexistReconciliationLLM(BaseLLMClient):
    """Fake classifier that would allow duplicates if exact dedup missed."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by reconciliation.")

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        response: dict[str, Any] = {
            "action": "coexist",
            "record_indexes": [],
            "reason": "fake classifier would keep both records",
            "confidence": "high",
        }
        return response_schema(**response)


def _sarah_write(
    *,
    subject_identifier: str,
    session_id: str,
) -> MemoryWrite:
    return MemoryWrite(
        category="relationship",
        subject=EntityRef(type="User", identifier=subject_identifier),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier="Sarah"),
        evidence_quote="I have a sister named Sarah",
        confidence="high",
        source_session_id=session_id,
        source_turn_index=0,
    )


@pytest.fixture(params=["memory", "postgres"])
async def reconciliation_store(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[MemoryStore, str]]:
    """Yield each supported memory backend with isolated records."""

    owner_id = f"semantic-reconciliation-{uuid4()}"
    if request.param == "memory":
        store = OpenCouchMemoryStore()
        try:
            yield store, owner_id
        finally:
            await store.aclose()
        return

    dsn = require_postgres_database_url()
    store = PostgresMemoryStore(dsn)
    try:
        yield store, owner_id
    finally:
        await store.aclose()
        await delete_postgres_memory_records_for_owners(dsn, [owner_id])


@pytest.mark.asyncio
async def test_session_end_semantic_reconciliation_dedups_user_subject_aliases(
    reconciliation_store: tuple[MemoryStore, str],
) -> None:
    """A seeded fact plus session-end alias extraction collapses durably."""

    store, owner_id = reconciliation_store
    session_id = f"thread-{uuid4()}"
    seeded_fact = memory_write_to_semantic_fact(
        _sarah_write(subject_identifier=owner_id, session_id=session_id),
        write_timing="immediate",
        write_reason="seeded test fact",
        policy_version="test_v1",
    )
    await write_new_semantic_fact(store, owner_id=owner_id, fact=seeded_fact)

    extracted_alias_write = _sarah_write(
        subject_identifier="test-user",
        session_id=session_id,
    )
    outcome = await apply_semantic_writes_batch(
        store,
        owner_id=owner_id,
        items=[
            BatchWriteItem(
                candidate=build_semantic_candidate(
                    extracted_alias_write,
                    message="I have a sister named Sarah",
                ),
                write_timing="session_end",
                write_reason="session-end extraction",
                policy_version="test_v1",
            )
        ],
        llm_client=_CoexistReconciliationLLM(),
        log_context="semantic_reconciliation_parity_test",
    )

    assert outcome.written == 0
    assert outcome.bumped == 1
    assert outcome.skipped == 0
    assert await store.arecord_count((owner_id, "semantic")) == 1

    [record] = await store.asearch((owner_id, "semantic"), query=None, limit=10)
    assert record.value["subject"]["identifier"] == owner_id
    assert record.value["object"]["identifier"] == "Sarah"


@pytest.mark.asyncio
async def test_duplicate_bump_preserves_dense_retrieval_metadata(
    reconciliation_store: tuple[MemoryStore, str],
) -> None:
    """A duplicate bump keeps the stored fact visible to dense retrieval."""

    store, owner_id = reconciliation_store
    session_id = f"thread-{uuid4()}"
    seeded_embedding = [1.0, 0.0, 0.0]
    seeded_fact = memory_write_to_semantic_fact(
        _sarah_write(subject_identifier=owner_id, session_id=session_id),
        write_timing="immediate",
        write_reason="seeded test fact",
        policy_version="test_v1",
    )
    await write_new_semantic_fact(
        store,
        owner_id=owner_id,
        fact=seeded_fact,
        embedding=seeded_embedding,
        embedding_model="test-embedding",
    )

    duplicate_write = _sarah_write(
        subject_identifier=owner_id,
        session_id=session_id,
    )
    outcome = await apply_semantic_writes_batch(
        store,
        owner_id=owner_id,
        items=[
            BatchWriteItem(
                candidate=build_semantic_candidate(
                    duplicate_write,
                    message="I have a sister named Sarah",
                ),
                write_timing="session_end",
                write_reason="session-end restatement",
                policy_version="test_v1",
            )
        ],
        llm_client=_CoexistReconciliationLLM(),
        log_context="semantic_reconciliation_parity_test",
    )

    assert outcome.written == 0
    assert outcome.bumped == 1
    assert await store.arecord_count((owner_id, "semantic")) == 1

    [record] = await store.asearch((owner_id, "semantic"), query=None, limit=10)
    assert record.value["last_referenced_at"] != seeded_fact.last_referenced_at
    assert record.embedding == seeded_embedding
    assert record.embedding_model == "test-embedding"

    # A paraphrase query with no token overlap can only match densely, so
    # retrieval succeeding here proves the bump kept the record in the
    # dense cohort rather than demoting it to lexical-only visibility.
    dense_hits = await store.asearch_similar(
        (owner_id, "semantic"),
        query_text="worried about tomorrow's presentation",
        query_embedding=seeded_embedding,
        embedding_model="test-embedding",
        limit=5,
    )
    assert [hit.key for hit in dense_hits] == [seeded_fact.id]


def _sister_name_write(
    *,
    owner_id: str,
    session_id: str,
    name: str,
) -> MemoryWrite:
    """Return a sister-name write that collides with the seeded Sarah fact."""

    return MemoryWrite(
        category="relationship",
        subject=EntityRef(type="User", identifier=owner_id),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier=name),
        evidence_quote=f"My sister {name} moved to Berlin last spring.",
        confidence="high",
        source_session_id=session_id,
        source_turn_index=1,
    )


@pytest.mark.asyncio
async def test_reconciliation_plan_commits_replacement_and_supersede_together(
    reconciliation_store: tuple[MemoryStore, str],
) -> None:
    """A supersede plan lands the replacement and the dormant record as a unit."""

    store, owner_id = reconciliation_store
    session_id = f"thread-{uuid4()}"
    stale_fact = memory_write_to_semantic_fact(
        _sarah_write(subject_identifier=owner_id, session_id=session_id),
        write_timing="immediate",
        write_reason="seeded stale fact",
        policy_version="test_v1",
    )
    await write_new_semantic_fact(store, owner_id=owner_id, fact=stale_fact)

    outcome = await apply_semantic_writes_batch(
        store,
        owner_id=owner_id,
        items=[
            BatchWriteItem(
                candidate=build_semantic_candidate(
                    _sister_name_write(
                        owner_id=owner_id,
                        session_id=session_id,
                        name="Sarah Chen",
                    ),
                    message="My sister Sarah Chen moved to Berlin last spring.",
                ),
                write_timing="session_end",
                write_reason="more specific name",
                policy_version="test_v1",
            )
        ],
        llm_client=_SupersedeReconciliationLLM(),
        log_context="semantic_reconciliation_parity_test",
    )

    assert outcome.written == 1
    assert await store.arecord_count((owner_id, "semantic")) == 2

    records = {
        record.key: record
        for record in await store.asearch((owner_id, "semantic"), query=None, limit=10)
    }
    replacement_id = outcome.written_items[0].id
    assert records[stale_fact.id].value["superseded_by"] == replacement_id
    assert records[stale_fact.id].value["dormant_at"] is not None
    assert records[replacement_id].value["superseded_by"] is None


@pytest.mark.asyncio
async def test_failed_reconciliation_commit_leaves_no_partial_state(
    reconciliation_store: tuple[MemoryStore, str],
) -> None:
    """A failing plan commit leaves neither a replacement nor a dormant fact."""

    store, owner_id = reconciliation_store
    session_id = f"thread-{uuid4()}"
    stale_fact = memory_write_to_semantic_fact(
        _sarah_write(subject_identifier=owner_id, session_id=session_id),
        write_timing="immediate",
        write_reason="seeded stale fact",
        policy_version="test_v1",
    )
    await write_new_semantic_fact(store, owner_id=owner_id, fact=stale_fact)

    async def _failing_aput_batch(items: Any) -> None:
        raise RuntimeError("simulated batch failure")

    original_aput_batch = store.aput_batch
    store.aput_batch = _failing_aput_batch  # type: ignore[method-assign]
    try:
        outcome = await apply_semantic_writes_batch(
            store,
            owner_id=owner_id,
            items=[
                BatchWriteItem(
                    candidate=build_semantic_candidate(
                        _sister_name_write(
                            owner_id=owner_id,
                            session_id=session_id,
                            name="Sarah Chen",
                        ),
                        message="My sister Sarah Chen moved to Berlin last spring.",
                    ),
                    write_timing="session_end",
                    write_reason="more specific name",
                    policy_version="test_v1",
                )
            ],
            llm_client=_SupersedeReconciliationLLM(),
            log_context="semantic_reconciliation_parity_test",
        )
    finally:
        store.aput_batch = original_aput_batch  # type: ignore[method-assign]

    assert outcome.written == 0
    assert outcome.skipped == 1

    # The stale fact must remain fully active: a dormant record whose
    # replacement was never written would silently drop the memory entirely.
    [record] = await store.asearch((owner_id, "semantic"), query=None, limit=10)
    assert record.key == stale_fact.id
    assert record.value["superseded_by"] is None
    assert record.value["dormant_at"] is None
