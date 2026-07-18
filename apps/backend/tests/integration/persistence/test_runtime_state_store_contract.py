"""Persistence contracts for runtime state stores."""

from __future__ import annotations

from uuid import uuid4

import pytest

import agent.runtime.state_store as runtime_state_store_module
from agent.models import Channel, CrisisAssessment
from tests.support.persistence_contracts import open_postgres_runtime_state_store

pytestmark = pytest.mark.asyncio


def _thread_id(prefix: str) -> str:
    """Return a unique thread id scoped to one contract test."""

    return f"{prefix}-{uuid4().hex}"


async def test_runtime_state_store_round_trip_preserves_serialized_state() -> None:
    """A stored runtime snapshot should round-trip with expected type restoration."""

    thread_id = _thread_id("postgres-round-trip")
    expected_crisis = CrisisAssessment(
        level=2,
        confidence="high",
        reason="contract test",
        needs_crisis_response=True,
        needs_clarification=False,
    )
    state = {
        "session_id": thread_id,
        "channel": Channel.WEB,
        "crisis": expected_crisis,
        "transcript": [{"role": "user", "content": "hello"}],
        "session_progress": {"turn_count": 2},
        "diagnostics": {"source": "contract-test"},
    }

    async with open_postgres_runtime_state_store() as store:
        assert await store.load_state(thread_id) is None
        await store.save_state(thread_id, state)

    async with open_postgres_runtime_state_store() as store:
        try:
            loaded = await store.load_state(thread_id)

            assert loaded is not None
            assert loaded["session_id"] == thread_id
            assert loaded["channel"] == Channel.WEB
            assert loaded["transcript"] == state["transcript"]
            assert loaded["session_progress"] == {"turn_count": 2}
            assert loaded["diagnostics"] == {"source": "contract-test"}
            crisis = loaded["crisis"]
            assert isinstance(crisis, CrisisAssessment)
            assert crisis.model_dump() == expected_crisis.model_dump()
        finally:
            await store.delete_thread(thread_id)


async def test_runtime_state_store_overwrite_updates_latest_value_and_recency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The latest save should replace prior state and drive recency ordering."""

    timestamps = iter(
        [
            "2026-05-25T00:00:00Z",
            "2026-05-25T00:00:01Z",
            "2026-05-25T00:00:02Z",
        ]
    )
    monkeypatch.setattr(runtime_state_store_module, "iso_now", lambda: next(timestamps))

    thread_a = _thread_id("postgres-state-a")
    thread_b = _thread_id("postgres-state-b")

    async with open_postgres_runtime_state_store() as store:
        await store.save_state(thread_a, {"session_progress": {"turn_count": 1}})
        await store.save_state(thread_b, {"session_progress": {"turn_count": 2}})
        await store.save_state(thread_a, {"session_progress": {"turn_count": 3}})

    async with open_postgres_runtime_state_store() as store:
        try:
            loaded_a = await store.load_state(thread_a)
            assert loaded_a is not None
            assert loaded_a["session_progress"] == {"turn_count": 3}
            assert await store.list_thread_ids(limit=2) == [thread_a, thread_b]
        finally:
            await store.delete_thread(thread_a)
            await store.delete_thread(thread_b)


async def test_runtime_state_store_delete_is_idempotent() -> None:
    """Deleting a thread should remove it and remain safe when repeated."""

    thread_id = _thread_id("postgres-delete")

    async with open_postgres_runtime_state_store() as store:
        await store.save_state(thread_id, {"session_progress": {"turn_count": 1}})
        await store.delete_thread(thread_id)

    async with open_postgres_runtime_state_store() as store:
        assert await store.load_state(thread_id) is None
        await store.delete_thread(thread_id)
        assert await store.load_state(thread_id) is None
