"""Contracts for non-durable runtime thread stores."""

from __future__ import annotations

import pytest

import agent.runtime.state_store as state_store_module
from agent.models import Channel, CrisisAssessment
from agent.runtime.session.store import (
    InMemoryActiveSessionStore,
    NullActiveSessionStore,
)
from agent.runtime.state_store import InMemoryRuntimeStateStore


@pytest.mark.asyncio
async def test_in_memory_state_round_trip_is_detached() -> None:
    store = InMemoryRuntimeStateStore()
    state = {
        "channel": Channel.WEB,
        "crisis": CrisisAssessment(
            level=1,
            confidence="high",
            reason="unit test",
            needs_crisis_response=False,
            needs_clarification=False,
        ),
        "transcript": [{"role": "user", "content": "hello"}],
    }

    await store.save_state("thread-1", state)
    state["transcript"].append({"role": "assistant", "content": "mutated"})
    loaded = await store.load_state("thread-1")

    assert loaded is not None
    assert loaded["channel"] is Channel.WEB
    assert isinstance(loaded["crisis"], CrisisAssessment)
    assert loaded["transcript"] == [{"role": "user", "content": "hello"}]

    loaded["transcript"].append({"role": "assistant", "content": "mutated"})
    reloaded = await store.load_state("thread-1")
    assert reloaded is not None
    assert reloaded["transcript"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_in_memory_state_recency_overwrite_and_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter(["same", "same", "same"])
    monkeypatch.setattr(state_store_module, "iso_now", lambda: next(timestamps))
    store = InMemoryRuntimeStateStore()

    await store.save_state("thread-a", {"value": 1})
    await store.save_state("thread-b", {"value": 2})
    await store.save_state("thread-a", {"value": 3})

    assert await store.list_thread_ids(limit=2) == ["thread-a", "thread-b"]
    assert await store.load_state("thread-a") == {"value": 3}
    await store.delete_thread("thread-a")
    await store.delete_thread("thread-a")
    assert await store.load_state("thread-a") is None


@pytest.mark.asyncio
async def test_in_memory_state_close_clears_and_rejects_reuse() -> None:
    store = InMemoryRuntimeStateStore()
    await store.save_state("thread-1", {"value": 1})

    await store.aclose()
    await store.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await store.load_state("thread-1")
    with pytest.raises(RuntimeError, match="closed"):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_in_memory_mutation_claim_requires_unclaimed_or_owned_marker() -> None:
    """The in-memory store applies the durable backend's ownership condition."""

    store = InMemoryActiveSessionStore()
    await store.save_payload("thread-claim", "payload")

    await store.set_mutation(
        "thread-claim",
        mutation_token="owner-token",
        mutation_kind="turn",
    )
    await store.set_mutation(
        "thread-claim",
        mutation_token="intruder-token",
        mutation_kind="rotation",
    )
    row = await store.load_row("thread-claim")
    assert row is not None
    assert row[1] == "owner-token"
    assert row[2] == "turn"

    # The owning token may update its own claim, and release frees the marker.
    await store.set_mutation(
        "thread-claim",
        mutation_token="owner-token",
        mutation_kind="finalize",
    )
    row = await store.load_row("thread-claim")
    assert row is not None
    assert row[2] == "finalize"

    await store.clear_mutation("thread-claim", "owner-token")
    await store.set_mutation(
        "thread-claim",
        mutation_token="next-token",
        mutation_kind="turn",
    )
    row = await store.load_row("thread-claim")
    assert row is not None
    assert row[1] == "next-token"


@pytest.mark.asyncio
async def test_in_memory_active_session_store_preserves_coordination_state() -> None:
    store = InMemoryActiveSessionStore()

    await store.save_payload("thread-b", "payload-b")
    await store.save_payload("thread-a", "payload-a")
    assert await store.list_ids() == ["thread-a", "thread-b"]

    await store.set_mutation(
        "thread-a",
        mutation_token="token-1",
        mutation_kind="turn",
        finalize_required_reason="interrupted",
    )
    await store.clear_mutation("thread-a", "wrong-token")
    assert await store.load_row("thread-a") == (
        "payload-a",
        "token-1",
        "turn",
        False,
        "interrupted",
    )

    await store.clear_mutation("thread-a", "token-1")
    await store.set_rotation_required("thread-a")
    assert await store.load_row("thread-a") == (
        "payload-a",
        None,
        None,
        True,
        "interrupted",
    )

    await store.delete_session("thread-a")
    await store.delete_session("thread-a")
    assert await store.load_row("thread-a") is None

    await store.aclose()
    await store.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await store.list_ids()


@pytest.mark.asyncio
async def test_null_active_session_store_is_a_safe_no_op() -> None:
    store = NullActiveSessionStore()

    await store.ensure_schema()
    await store.save_payload("thread-1", "{}")
    await store.set_mutation(
        "thread-1",
        mutation_token="token-1",
        mutation_kind="turn",
        finalize_required_reason="interrupted",
    )
    await store.clear_mutation("thread-1", "token-1")
    await store.set_rotation_required("thread-1")
    await store.delete_session("thread-1")
    await store.aclose()
    await store.aclose()

    assert await store.load_row("thread-1") is None
    assert await store.list_ids() == []
