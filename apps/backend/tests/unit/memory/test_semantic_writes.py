"""Tests for shared semantic-memory write primitives."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent.memory.types import EntityRef, MemoryWrite
from agent.memory.operations.semantic_writes import (
    apply_semantic_write,
    bump_semantic_last_referenced_at,
    fetch_existing_semantic_records,
    mark_semantic_fact_superseded,
    memory_write_to_semantic_fact,
    write_new_semantic_fact,
)
from agent.memory.store import OpenCouchMemoryStore
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeReconciliationLLM(BaseLLMClient):
    """Fake semantic reconciliation classifier for unit tests."""

    def __init__(self, action: str = "supersede") -> None:
        self._action = action

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
            "action": self._action,
            "record_indexes": [0],
            "reason": "The new fact is a more specific replacement.",
            "confidence": "high",
        }
        return response_schema(**response)


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


async def test_bump_semantic_last_referenced_at_preserves_embedding_metadata() -> None:
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

    await bump_semantic_last_referenced_at(store, matched_record=record)

    updated = await store.aget(("user-1", "semantic"), fact.id)
    assert updated is not None
    assert updated.value["last_referenced_at"] != fact.last_referenced_at
    assert updated.embedding == [0.3, 0.4]
    assert updated.embedding_model == "test-embedding"


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


async def test_apply_semantic_write_persists_new_fact_and_updates_cache() -> None:
    store = OpenCouchMemoryStore()
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    write = _memory_write()

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=write,
        existing_records=existing_records,
        llm_client=None,
        write_timing="immediate",
        write_reason="test write",
        policy_version="test_v1",
        embedding=[0.5, 0.6],
        embedding_model="test-embedding",
    )

    assert outcome.written == 1
    assert outcome.bumped == 0
    assert outcome.skipped == 0
    assert outcome.fact is not None
    assert len(existing_records) == 1

    stored = await store.aget(("user-1", "semantic"), outcome.fact.id)
    assert stored is not None
    assert stored.embedding == [0.5, 0.6]
    assert stored.embedding_model == "test-embedding"
    assert stored.value["write_reason"] == "test write"


async def test_apply_semantic_write_bumps_duplicate_without_new_record() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())
    seed_value = fact.model_dump(mode="json")
    seed_value["last_referenced_at"] = "2026-01-01T00:00:00Z"
    await store.aput(
        ("user-1", "semantic"),
        key=fact.id,
        value=seed_value,
        embedding=[0.7, 0.8],
        embedding_model="test-embedding",
    )
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=_memory_write(),
        existing_records=existing_records,
        llm_client=None,
        write_timing="immediate",
        write_reason="duplicate",
        policy_version="test_v1",
    )

    assert outcome.written == 0
    assert outcome.bumped == 1
    assert outcome.skipped == 0
    assert outcome.fact is None
    assert len(await fetch_existing_semantic_records(store, owner_id="user-1")) == 1

    updated = await store.aget(("user-1", "semantic"), fact.id)
    assert updated is not None
    assert updated.value["last_referenced_at"] != "2026-01-01T00:00:00Z"
    assert updated.embedding == [0.7, 0.8]
    assert updated.embedding_model == "test-embedding"


async def test_apply_semantic_write_reconciliation_bump_preserves_embedding() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(
        store,
        owner_id="user-1",
        fact=fact,
        embedding=[0.1, 0.9],
        embedding_model="test-embedding",
    )
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    rephrased_write = _memory_write().model_copy(
        update={"evidence_quote": "Sarah, my sister, rang me yesterday evening."}
    )

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=rephrased_write,
        existing_records=existing_records,
        llm_client=_FakeReconciliationLLM(action="bump"),
        write_timing="immediate",
        write_reason="reconciliation bump",
        policy_version="test_v1",
    )

    assert outcome.written == 0
    assert outcome.bumped == 1
    assert len(await fetch_existing_semantic_records(store, owner_id="user-1")) == 1

    updated = await store.aget(("user-1", "semantic"), fact.id)
    assert updated is not None
    assert updated.value["last_referenced_at"] != fact.last_referenced_at
    assert updated.embedding == [0.1, 0.9]
    assert updated.embedding_model == "test-embedding"


async def test_apply_semantic_write_dedups_user_subject_alias_to_owner() -> None:
    store = OpenCouchMemoryStore()
    fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(store, owner_id="user-1", fact=fact)
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    alias_write = _memory_write().model_copy(
        update={"subject": EntityRef(type="User", identifier="test-user")}
    )

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=alias_write,
        existing_records=existing_records,
        llm_client=None,
        write_timing="session_end",
        write_reason="duplicate alias",
        policy_version="test_v1",
    )

    assert outcome.written == 0
    assert outcome.bumped == 1
    assert outcome.skipped == 0
    assert len(await fetch_existing_semantic_records(store, owner_id="user-1")) == 1


async def test_apply_semantic_write_supersedes_stale_same_slot_record() -> None:
    store = OpenCouchMemoryStore()
    old_fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(store, owner_id="user-1", fact=old_fact)
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    new_write = _memory_write().model_copy(
        update={
            "object": EntityRef(type="Person", identifier="Sarah Chen"),
            "evidence_quote": "My sister Sarah Chen called last night.",
        }
    )

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=new_write,
        existing_records=existing_records,
        llm_client=_FakeReconciliationLLM(),
        write_timing="immediate",
        write_reason="more specific",
        policy_version="test_v1",
    )

    assert outcome.written == 1
    assert outcome.fact is not None

    old_record = await store.aget(("user-1", "semantic"), old_fact.id)
    assert old_record is not None
    assert old_record.value["superseded_by"] == outcome.fact.id
    assert old_record.value["dormant_at"] is not None

    active_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    assert len(active_records) == 2
    assert existing_records[-1].key == outcome.fact.id


class _FailingBatchStore(OpenCouchMemoryStore):
    """Store whose batch writes always fail, simulating a broken commit."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    async def aput_batch(self, items: Any) -> None:
        self.batch_calls += 1
        raise RuntimeError("simulated batch failure")


async def test_apply_semantic_write_supersede_plan_commits_as_one_batch() -> None:
    """The replacement and its supersede updates share a single batch call."""

    store = OpenCouchMemoryStore()
    batched: list[list[Any]] = []
    original_aput_batch = store.aput_batch

    async def _recording_aput_batch(items: Any) -> None:
        batched.append(list(items))
        await original_aput_batch(items)

    store.aput_batch = _recording_aput_batch  # type: ignore[method-assign]

    old_fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(store, owner_id="user-1", fact=old_fact)
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    new_write = _memory_write().model_copy(
        update={
            "object": EntityRef(type="Person", identifier="Sarah Chen"),
            "evidence_quote": "My sister Sarah Chen called last night.",
        }
    )

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=new_write,
        existing_records=existing_records,
        llm_client=_FakeReconciliationLLM(),
        write_timing="immediate",
        write_reason="more specific",
        policy_version="test_v1",
    )

    assert outcome.written == 1
    assert outcome.fact is not None
    assert len(batched) == 1
    committed_keys = {item[1] for item in batched[0]}
    assert committed_keys == {outcome.fact.id, old_fact.id}


async def test_apply_semantic_write_rolls_back_when_supersede_commit_fails() -> None:
    """A failed reconciliation commit leaves no replacement and no stale edit."""

    store = _FailingBatchStore()
    old_fact = memory_write_to_semantic_fact(_memory_write())
    await write_new_semantic_fact(store, owner_id="user-1", fact=old_fact)
    existing_records = await fetch_existing_semantic_records(store, owner_id="user-1")
    new_write = _memory_write().model_copy(
        update={
            "object": EntityRef(type="Person", identifier="Sarah Chen"),
            "evidence_quote": "My sister Sarah Chen called last night.",
        }
    )

    outcome = await apply_semantic_write(
        store,
        owner_id="user-1",
        write=new_write,
        existing_records=existing_records,
        llm_client=_FakeReconciliationLLM(),
        write_timing="immediate",
        write_reason="more specific",
        policy_version="test_v1",
    )

    assert store.batch_calls == 1
    assert outcome.written == 0
    assert outcome.skipped == 1
    assert outcome.fact is None

    # The replacement must not be active, and the fact it would have
    # superseded must remain untouched rather than dormant-without-successor.
    remaining = await fetch_existing_semantic_records(store, owner_id="user-1")
    assert len(remaining) == 1
    assert remaining[0].key == old_fact.id
    assert remaining[0].value["superseded_by"] is None
    assert remaining[0].value["dormant_at"] is None

    # The caller's dedup cache must not gain a fact that was never persisted.
    assert [record.key for record in existing_records] == [old_fact.id]
