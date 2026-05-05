"""Tests for shared semantic-memory write primitives."""

from __future__ import annotations

from agent.memory.models import EntityRef, MemoryWrite
from agent.memory.semantic_writes import (
    bump_semantic_last_referenced_at,
    fetch_existing_semantic_records,
    mark_semantic_fact_superseded,
    memory_write_to_semantic_fact,
    write_new_semantic_fact,
)
from agent.memory.store import OpenCouchMemoryStore


def _memory_write() -> MemoryWrite:
    return MemoryWrite(
        category="relationship",
        subject=EntityRef(type="User", identifier="user-1"),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier="Sarah"),
        evidence_quote="My sister Sarah called last night.",
        confidence="high",
        source_session_id="thread-1",
        source_turn_index=2,
    )


def test_memory_write_to_semantic_fact_preserves_write_fields() -> None:
    write = _memory_write()

    fact = memory_write_to_semantic_fact(
        write,
        write_timing="session_end",
        write_reason="session supported",
        policy_version="test_v1",
    )

    assert fact.category == write.category
    assert fact.subject == write.subject
    assert fact.predicate == write.predicate
    assert fact.object == write.object
    assert fact.evidence_quote == write.evidence_quote
    assert fact.confidence == write.confidence
    assert fact.source_session_id == write.source_session_id
    assert fact.source_turn_index == write.source_turn_index
    assert fact.write_timing == "session_end"
    assert fact.write_reason == "session supported"
    assert fact.policy_version == "test_v1"
    assert fact.dormant_at is None
    assert fact.superseded_by is None
    assert fact.user_visible is True


async def test_write_and_fetch_existing_semantic_records() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())

    await write_new_semantic_fact(
        store,
        owner_id="user-1",
        fact=fact,
        embedding=[0.1, 0.2],
        embedding_model="test-embedding",
    )

    records = await fetch_existing_semantic_records(store, owner_id="user-1")

    assert len(records) == 1
    assert records[0].key == fact.id
    assert records[0].value["evidence_quote"] == fact.evidence_quote
    assert records[0].embedding == [0.1, 0.2]
    assert records[0].embedding_model == "test-embedding"


async def test_bump_semantic_last_referenced_at_updates_record() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(store, owner_id="user-1", fact=fact)
    record = (await fetch_existing_semantic_records(store, owner_id="user-1"))[0]

    await bump_semantic_last_referenced_at(store, matched_record=record)

    updated = await store.aget(("user-1", "semantic"), fact.id)
    assert updated is not None
    assert updated.value["last_referenced_at"] != fact.last_referenced_at
    assert updated.value["evidence_quote"] == fact.evidence_quote


async def test_mark_semantic_fact_superseded_preserves_embedding_metadata() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(
        store,
        owner_id="user-1",
        fact=fact,
        embedding=[0.3, 0.4],
        embedding_model="test-embedding",
    )
    record = (await fetch_existing_semantic_records(store, owner_id="user-1"))[0]

    await mark_semantic_fact_superseded(
        store,
        matched_record=record,
        replacement_fact_id="replacement-fact",
    )

    updated = await store.aget(("user-1", "semantic"), fact.id)
    assert updated is not None
    assert updated.value["dormant_at"] is not None
    assert updated.value["superseded_by"] == "replacement-fact"
    assert updated.embedding == [0.3, 0.4]
    assert updated.embedding_model == "test-embedding"
