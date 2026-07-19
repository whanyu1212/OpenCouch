"""Guard tests for the VoiceRuntimeFacade wiring.

These tests protect two invariants that are easy to break during
future refactoring:

1. **Lock identity** — the facade and the runtime must share the same
   per-thread lock instances.  If they diverge, concurrent voice and
   text mutations on the same thread race.

2. **Facade existence** — ``PersistentAgentRuntime.voice`` must be a
   ``VoiceRuntimeFacade`` so callers can rely on attribute access.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from agent.voice.concurrent_safety import VoiceConcurrentSafetyResult
from llm.base import BaseLLMClient
from tests.support.persistence import (
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)
from agent.voice.runtime_facade import VoiceRuntimeFacade


def test_runtime_exposes_voice_facade() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    assert isinstance(runtime.voice, VoiceRuntimeFacade)


@pytest.mark.asyncio
async def test_voice_facade_shares_thread_lock_with_runtime() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    async with runtime:
        assert runtime.voice._lock_for("t") is runtime._thread_lock("t")
        assert runtime.voice._session_lifecycle is runtime._session_lifecycle


class _ReferenceStateStore:
    def __init__(self, state: dict[str, Any] | None, *, fail: bool = False) -> None:
        self.state = state
        self.fail = fail
        self.save_calls = 0

    async def load_state(self, thread_id: str):
        assert thread_id == "voice-thread"
        if self.fail:
            raise RuntimeError("snapshot unavailable")
        return self.state

    async def save_state(self, thread_id: str, state: object) -> None:
        del thread_id, state
        self.save_calls += 1


class _InspectingConcurrentSafetyService:
    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock
        self.prior_transcript: list[dict[str, Any]] | None = None

    async def assess_turn(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        prior_transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
    ) -> VoiceConcurrentSafetyResult:
        del thread_id, user_id, user_text, llm_client
        assert self.lock.locked() is False
        self.prior_transcript = prior_transcript
        return VoiceConcurrentSafetyResult("skipped", "test", None, 1.0)


@pytest.mark.asyncio
async def test_concurrent_safety_releases_lock_and_isolates_snapshot() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    source_state = {
        "transcript": [
            {"role": "user", "content": "earlier", "metadata": {"value": 1}},
            {"role": "assistant", "content": "current turn already persisted"},
        ]
    }
    state_store = _ReferenceStateStore(source_state)
    lock = runtime._thread_lock("voice-thread")
    concurrent_service = _InspectingConcurrentSafetyService(lock)

    async with runtime:
        runtime.voice._state_store = state_store  # type: ignore[assignment]
        runtime.voice._concurrent_safety_service = concurrent_service  # type: ignore[assignment]
        result = await runtime.voice.assess_voice_turn_safety(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="current",
            prior_message_count=1,
            pending_prior_transcript=[
                {"role": "user", "content": "pending prior"},
                {"role": "assistant", "content": "pending answer"},
            ],
            llm_client=None,
        )
        source_state["transcript"][0]["metadata"]["value"] = 2

    assert result.status == "skipped"
    assert concurrent_service.prior_transcript == [
        {"role": "user", "content": "earlier", "metadata": {"value": 1}},
        {"role": "user", "content": "pending prior"},
        {"role": "assistant", "content": "pending answer"},
    ]
    assert concurrent_service.prior_transcript is not source_state["transcript"]
    assert state_store.save_calls == 0


@pytest.mark.asyncio
async def test_concurrent_safety_snapshot_failure_fails_open() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    state_store = _ReferenceStateStore(None, fail=True)

    async with runtime:
        runtime.voice._state_store = state_store  # type: ignore[assignment]
        result = await runtime.voice.assess_voice_turn_safety(
            thread_id="voice-thread",
            user_id=None,
            user_text="current",
            prior_message_count=0,
            pending_prior_transcript=[],
            llm_client=None,
        )

    assert result.status == "failed"
    assert result.reason == "state_snapshot_failed"
    assert result.assessment is None
    assert result.duration_ms >= 0
    assert state_store.save_calls == 0
