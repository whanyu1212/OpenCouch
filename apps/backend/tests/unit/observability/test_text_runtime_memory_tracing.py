"""Tests for text runtime and memory trace instrumentation."""

from __future__ import annotations

from typing import Any

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.guardrails.assessment import assess_crisis_gate
from agent.guardrails.service import CrisisRiskResult
from agent.memory.modes import MemoryMode
from agent.memory.retrieval.service import LoadMemoryResult
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput, CrisisAssessment
from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import (
    MEMORY_READ,
    MEMORY_READ_COMPLETED,
    MEMORY_READ_SKIPPED,
    RUNTIME_TEXT_TURN,
    RUNTIME_TEXT_TURN_FINALIZED,
    SAFETY_ASSESS,
    SAFETY_ASSESS_COMPLETED,
    SDK_OPENAI_CALL,
    SDK_OPENAI_CALL_COMPLETED,
)
from agent.observability.recorder import InMemoryTraceRecorder
from agent.runtime import OpenAITextRuntime, build_initial_state
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.memory_context import build_turn_memory_delta
from agent.runtime.state_ops import finalize_openai_turn
from agent.runtime.workflow_context import WorkflowContext
from tests.support.openai_text import FakeOpenAISDKRunner
from tests.support.persistence import FakeCrossRestartLLM


class _FakeCrisisRiskService:
    async def assess_turn(
        self,
        state: dict[str, Any],
        *,
        llm_client: Any | None,
    ) -> CrisisRiskResult:
        del state, llm_client
        return CrisisRiskResult(
            assessment=CrisisAssessment(
                level=2,
                confidence="high",
                reason="sensitive model-derived reason",
                needs_crisis_response=True,
                needs_clarification=False,
            ),
            classifier_path="llm_primary",
            override_kind="none",
            llm_failure_occurred=False,
        )


def _workflow_context(*, memory_mode: MemoryMode = MemoryMode.LOCAL) -> WorkflowContext:
    return WorkflowContext(
        llm_client=FakeCrossRestartLLM(),
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=memory_mode,
    )


def _state(message: str = "I feel tense today") -> dict[str, Any]:
    return dict(
        build_initial_state(
            AgentInput(message=message, user_id="user-1", session_id="thread-1")
        )
    )


@pytest.mark.asyncio
async def test_safety_assessment_emits_privacy_safe_trace_event() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        result = await assess_crisis_gate(
            _state(),
            llm_client=FakeCrossRestartLLM(),
            service=_FakeCrisisRiskService(),  # type: ignore[arg-type]
        )

    assert result.assessment.reason == "sensitive model-derived reason"
    assert [span.name for span in recorder.completed_spans] == [SAFETY_ASSESS]
    event = next(
        event for event in recorder.events if event.name == SAFETY_ASSESS_COMPLETED
    )
    assert event.attributes == {
        "level": 2,
        "needs_crisis_response": True,
        "needs_clarification": False,
        "classifier_path": "llm_primary",
        "override_kind": "none",
        "llm_failure_occurred": False,
    }
    assert "reason" not in event.attributes


@pytest.mark.asyncio
async def test_memory_read_incognito_emits_skipped_event() -> None:
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(trace_context, recorder):
        delta = await build_turn_memory_delta(
            _state(),
            _workflow_context(memory_mode=MemoryMode.INCOGNITO),
        )

    assert delta["working_memory"] == []
    span = recorder.completed_spans[0]
    assert span.name == MEMORY_READ
    assert span.attributes == {"memory_mode": "incognito"}
    event = recorder.events[0]
    assert event.name == MEMORY_READ_SKIPPED
    assert event.attributes == {"reason": "incognito"}


@pytest.mark.asyncio
async def test_memory_read_completed_event_uses_counts_not_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve_memory_result(
        **kwargs: Any,
    ) -> tuple[LoadMemoryResult, str, float]:
        del kwargs
        return (
            LoadMemoryResult(
                working_memory=["private memory"],  # type: ignore[list-item]
                summary="private summary",
                procedural_rules=["private rule"],
                proactive_recall_enabled=True,
                diagnostics={"existing": "diagnostic"},
            ),
            "used",
            3.21,
        )

    monkeypatch.setattr(
        "agent.runtime.memory_context._resolve_memory_result",
        fake_resolve_memory_result,
    )
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(trace_context, recorder):
        delta = await build_turn_memory_delta(_state(), _workflow_context())

    assert delta["diagnostics"]["existing"] == "diagnostic"
    event = next(
        event for event in recorder.events if event.name == MEMORY_READ_COMPLETED
    )
    assert event.attributes == {
        "speculation_status": "used",
        "speculation_used": True,
        "speculation_wait_ms": 3.21,
        "working_memory_count": 1,
        "procedural_rule_count": 1,
        "proactive_recall_enabled": True,
    }
    assert "private memory" not in str(event.attributes)
    assert "private summary" not in str(event.attributes)


def test_finalize_openai_turn_emits_safe_metadata_event() -> None:
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))
    state = _state()

    with use_trace_context(trace_context, recorder):
        final_state = finalize_openai_turn(
            state,
            response_text="private assistant response",
            runtime_mode="safe_therapeutic",
            response_style="therapeutic_response",
            selected_agent="therapeutic",
            sdk_duration_ms=12.345,
            streamed=False,
        )

    assert final_state["diagnostics"]["openai_sdk_ms"] == 12.35
    event = next(
        event for event in recorder.events if event.name == RUNTIME_TEXT_TURN_FINALIZED
    )
    assert event.attributes == {
        "runtime_mode": "safe_therapeutic",
        "response_style": "therapeutic_response",
        "selected_agent": "therapeutic",
        "streamed": False,
        "sdk_duration_ms": 12.35,
    }
    assert "private assistant response" not in str(event.attributes)


@pytest.mark.asyncio
async def test_openai_sdk_call_emits_span_and_completion_event_without_payloads() -> (
    None
):
    runner = FakeOpenAISDKRunner("private sdk response")
    runtime = OpenAITextRuntime(runner=runner, model="gpt-test")
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))
    state = _state()
    run_context = OpenAITextRunContext(
        thread_id="thread-1",
        workflow_context=_workflow_context(),
        current_user_message="private user message",
        user_id="user-1",
        session_id="thread-1",
        agent_state=state,
    )

    with use_trace_context(trace_context, recorder):
        text, duration_ms = await runtime._run_openai_agent_with(
            state,
            agent=runtime._roster.therapeutic_agent,
            input_text="private prompt",
            run_context=run_context,
        )

    assert text == "private sdk response"
    assert duration_ms >= 0
    span = recorder.completed_spans[0]
    assert span.name == SDK_OPENAI_CALL
    assert span.attributes == {
        "model": "gpt-test",
        "agent_name": runtime._roster.therapeutic_agent.name,
    }
    event = next(
        event for event in recorder.events if event.name == SDK_OPENAI_CALL_COMPLETED
    )
    assert event.attributes["response_text_length"] == len("private sdk response")
    assert "private sdk response" not in str(event.attributes)
    assert "private prompt" not in str(span.attributes)


@pytest.mark.asyncio
async def test_run_turn_smoke_path_emits_root_text_turn_span() -> None:
    runtime = OpenAITextRuntime(runner=FakeOpenAISDKRunner(), model="gpt-test")
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))
    context = WorkflowContext(
        llm_client=None,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )

    with use_trace_context(trace_context, recorder):
        await runtime.run_turn(
            _state(),
            config={"configurable": {"thread_id": "thread-1"}},
            context=context,
        )

    span = recorder.completed_spans[0]
    assert span.name == RUNTIME_TEXT_TURN
    assert span.attributes == {"runtime_mode": "text", "streamed": False}
