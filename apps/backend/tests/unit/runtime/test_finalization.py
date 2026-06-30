from __future__ import annotations

from typing import Any, cast

import pytest

import agent.runtime.finalization as finalization_module
from agent.audit.capture import SafetyEventCaptureResult
from agent.runtime.finalization import finalize_successful_turn
from agent.runtime.session.manager import ActiveSessionManager
from agent.runtime.state_store import RuntimeStateStore
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState


class _RecordingStateStore:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def save_state(self, thread_id: str, state: dict[str, Any]) -> None:
        assert thread_id == "thread-1"
        assert state["response_text"] == "final response"
        self.calls.append("save_state")


class _RecordingActiveSessionManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def clear_active_session_mutation(
        self,
        thread_id: str,
        mutation_token: str,
    ) -> None:
        assert thread_id == "thread-1"
        assert mutation_token == "mutation-token"
        self.calls.append("clear_mutation")


@pytest.mark.asyncio
async def test_finalize_successful_turn_runs_shared_ordering_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    state = cast(AgentState, {"response_text": "final response"})
    context = cast(WorkflowContext, object())

    async def capture_safety_event(
        final_state: AgentState,
        workflow_context: WorkflowContext,
    ) -> SafetyEventCaptureResult:
        assert final_state is state
        assert workflow_context is context
        calls.append("capture_safety")
        return SafetyEventCaptureResult(kind="crisis_response", status="captured")

    async def ensure_sdk_turn_recorded(
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        assert thread_id == "thread-1"
        assert user_message == "hello"
        assert final_state is state
        calls.append("sdk_history")

    monkeypatch.setattr(
        finalization_module,
        "capture_post_save_safety_event",
        capture_safety_event,
    )

    result = await finalize_successful_turn(
        thread_id="thread-1",
        user_message="hello",
        final_state=state,
        workflow_context=context,
        state_store=cast(RuntimeStateStore, _RecordingStateStore(calls)),
        active_session_manager=cast(
            ActiveSessionManager,
            _RecordingActiveSessionManager(calls),
        ),
        mutation_token="mutation-token",
        ensure_sdk_turn_recorded=ensure_sdk_turn_recorded,
    )

    assert result.status == "captured"
    assert calls == [
        "save_state",
        "capture_safety",
        "sdk_history",
        "clear_mutation",
    ]


@pytest.mark.asyncio
async def test_finalize_successful_turn_captures_before_sdk_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    state = cast(AgentState, {"response_text": "final response"})

    async def capture_safety_event(
        final_state: AgentState,
        workflow_context: WorkflowContext,
    ) -> SafetyEventCaptureResult:
        del final_state, workflow_context
        calls.append("capture_safety")
        return SafetyEventCaptureResult(kind="crisis_response", status="captured")

    async def fail_sdk_turn_recorded(
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        del thread_id, user_message, final_state
        calls.append("sdk_history")
        raise RuntimeError("simulated sdk failure")

    monkeypatch.setattr(
        finalization_module,
        "capture_post_save_safety_event",
        capture_safety_event,
    )

    with pytest.raises(RuntimeError, match="simulated sdk failure"):
        await finalize_successful_turn(
            thread_id="thread-1",
            user_message="hello",
            final_state=state,
            workflow_context=cast(WorkflowContext, object()),
            state_store=cast(RuntimeStateStore, _RecordingStateStore(calls)),
            active_session_manager=cast(
                ActiveSessionManager,
                _RecordingActiveSessionManager(calls),
            ),
            mutation_token="mutation-token",
            ensure_sdk_turn_recorded=fail_sdk_turn_recorded,
        )

    assert calls == ["save_state", "capture_safety", "sdk_history"]
