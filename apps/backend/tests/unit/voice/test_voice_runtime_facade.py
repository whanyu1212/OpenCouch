"""Guard tests for the VoiceRuntimeFacade wiring.

These tests protect three invariants that are easy to break during
future refactoring:

1. **Lock identity** — the facade and the runtime must share the same
   per-thread lock instances.  If they diverge, concurrent voice and
   text mutations on the same thread race.

2. **Facade existence** — ``PersistentAgentRuntime.voice`` must be a
   ``VoiceRuntimeFacade`` so callers can rely on attribute access.

3. **Narrow collaboration** — the facade must not retain a full runtime
   back-pointer; it receives only the operations needed to coordinate voice turns.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.memory.modes import MemoryMode
from agent.models import AgentInput, Channel
from agent.runtime import PersistentAgentRuntime
from agent.runtime.turn import build_initial_state
from agent.runtime.workflow_context import WorkflowContext
from agent.voice.concurrent_safety import VoiceConcurrentSafetyResult
from agent.voice.runtime_collaboration import VoiceRuntimeCollaboration
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
    assert not hasattr(runtime.voice, "_runtime")


@pytest.mark.asyncio
async def test_voice_facade_builds_tool_context_through_collaboration() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    calls: list[str] = []

    async def get_state(thread_id: str) -> None:
        assert thread_id == "voice-thread"
        calls.append("get_state")
        return None

    def build_turn_initial_state(**kwargs: object) -> object:
        calls.append("build_turn_initial_state")
        return build_initial_state(
            AgentInput(
                message=str(kwargs["message"]),
                channel=Channel.VOICE,
                user_id=kwargs["user_id"]
                if isinstance(kwargs["user_id"], str)
                else None,
                session_id=str(kwargs["thread_id"]),
            ),
            prior_turn_count=int(kwargs["prior_turn_count"]),
        )

    def build_workflow_context(**kwargs: object) -> WorkflowContext:
        calls.append("build_workflow_context")
        return WorkflowContext(
            llm_client=kwargs["llm_client"]
            if isinstance(kwargs["llm_client"], BaseLLMClient)
            else None,
            response_llm=kwargs["response_llm_client"]
            if isinstance(kwargs["response_llm_client"], BaseLLMClient)
            else None,
            memory_store=runtime._memory_store,  # noqa: SLF001
            crisis_log_backend=runtime._crisis_log_backend,  # noqa: SLF001
            memory_mode=runtime.memory_mode,
        )

    async def prepare_session_for_turn(**kwargs: object) -> None:
        raise AssertionError(f"Unexpected session preparation: {kwargs!r}")

    def remember_llm_client(thread_id: str, llm_client: BaseLLMClient | None) -> None:
        del thread_id, llm_client
        raise AssertionError("Unexpected LLM client tracking")

    async def ensure_sdk_turn_recorded(**kwargs: object) -> None:
        raise AssertionError(f"Unexpected SDK turn recording: {kwargs!r}")

    async with runtime:
        facade = VoiceRuntimeFacade(
            collaboration=VoiceRuntimeCollaboration(
                get_state=get_state,
                build_turn_initial_state=build_turn_initial_state,
                build_workflow_context=build_workflow_context,
                prepare_session_for_turn=prepare_session_for_turn,
                remember_llm_client=remember_llm_client,
                ensure_sdk_turn_recorded=ensure_sdk_turn_recorded,
            ),
            state_store=runtime._state_store,  # noqa: SLF001
            memory_store=runtime._memory_store,  # noqa: SLF001
            active_session_manager=runtime._active_session_manager,  # noqa: SLF001
            session_lifecycle=runtime._session_lifecycle,  # noqa: SLF001
            lock_for=runtime._thread_lock,  # noqa: SLF001
            memory_mode=runtime.memory_mode,
        )
        context = await facade.build_voice_tool_context(
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Please show my saved memory.",
            transcript=[],
        )

    assert context.current_user_message == "Please show my saved memory."
    assert calls == [
        "get_state",
        "build_turn_initial_state",
        "build_workflow_context",
    ]
    assert not hasattr(facade, "_runtime")


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
