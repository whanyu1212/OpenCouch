"""Tests for the hybrid OpenAI text-runtime adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.runtime_context import WorkflowContext
from agent.text_runtime import (
    LangGraphTextAgentAdapter,
    OpenAITextAgentAdapter,
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
)
from agent.text_runtime.openai_agents import THERAPEUTIC_AGENT_NAME
from tests.support.openai_text import FakeOpenAISDKRunner
from tests.support.persistence import FakeCrossRestartLLM


class _StatefulWorkflow:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None
        self.ainvoke_calls = 0

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(values=self.state)

    async def ainvoke(
        self,
        initial_state: dict[str, Any],
        *,
        config: dict[str, Any],
        context: WorkflowContext,
    ) -> dict[str, Any]:
        self.ainvoke_calls += 1
        return {"response_text": "fallback reply"}

    async def astream(
        self,
        initial_state: dict[str, Any],
        *,
        config: dict[str, Any],
        context: WorkflowContext,
        stream_mode: tuple[str, ...],
        subgraphs: bool,
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        self.ainvoke_calls += 1
        yield {"type": "custom", "data": {"type": "chunk", "text": "fallback"}}
        yield {
            "type": "values",
            "ns": (),
            "data": {"response_text": "fallback reply"},
        }

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        if self.state is None:
            self.state = dict(values)
            return
        updated = dict(self.state)
        for key, value in values.items():
            if key == "transcript":
                updated[key] = [*updated.get(key, []), *value]
            elif key in {
                "session_memory",
                "procedural_profile",
                "session_progress",
                "exercise_state",
                "memory_control",
                "grounded_lookup",
                "diagnostics",
            }:
                updated[key] = {**updated.get(key, {}), **value}
            else:
                updated[key] = value
        self.state = updated


class _RouteLLM(FakeCrossRestartLLM):
    def __init__(
        self,
        *,
        route: str,
        crisis_level: int = 0,
        memory_reference_mode: str = "none",
    ) -> None:
        super().__init__()
        self.route = route
        self.crisis_level = crisis_level
        self.memory_reference_mode = memory_reference_mode

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
    ) -> Any:
        schema_name = response_schema.__name__
        if schema_name == "CrisisAssessmentSchema":
            return response_schema(
                level=self.crisis_level,
                confidence="high",
                reason="scripted crisis verdict",
                needs_crisis_response=self.crisis_level >= 2,
                needs_clarification=self.crisis_level == 1,
            )
        if schema_name == "TurnDispatchDecision":
            kwargs: dict[str, Any] = {
                "route": self.route,
                "active_flow_action": "none",
                "reasoning": f"scripted {self.route} route",
                "confidence": "high",
                "memory_reference_mode": self.memory_reference_mode,
            }
            if self.route == "memory_control":
                kwargs["memory_action_type"] = "status"
            if self.route == "grounded_lookup":
                kwargs["query"] = "grounded query"
            return response_schema(**kwargs)
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )


def _adapter(
    workflow: _StatefulWorkflow,
    runner: FakeOpenAISDKRunner,
) -> OpenAITextAgentAdapter:
    return OpenAITextAgentAdapter(
        fallback=LangGraphTextAgentAdapter(cast(Any, workflow)),
        runner=cast(Any, runner),
        model="gpt-test",
    )


def _initial_state(message: str = "I feel tense today") -> dict[str, Any]:
    return dict(
        build_initial_state(
            AgentInput(
                message=message,
                user_id="user-1",
                session_id="thread-1",
            )
        )
    )


def _context(llm: Any | None = None) -> WorkflowContext:
    return WorkflowContext(
        llm_client=llm or FakeCrossRestartLLM(),
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )


@pytest.mark.asyncio
async def test_openai_adapter_runs_safe_therapeutic_turn_and_persists_state() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    adapter = _adapter(workflow, runner)

    state = await adapter.run_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert workflow.ainvoke_calls == 0
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].tools == []
    assert "Write the next assistant message" in runner.run_calls[0]["input_text"]
    assert state["response_text"] == "openai reply"
    assert state["response_style"] == "supportive"
    assert state["therapeutic_approach"] == "none"
    assert state["diagnostics"]["text_agent_runtime"] == "openai"
    assert workflow.state is not None
    assert [turn["role"] for turn in workflow.state["transcript"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_openai_adapter_falls_back_for_unsupported_safe_route() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("What do you remember about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="memory_control")),
    )

    assert result == {"response_text": "fallback reply"}
    assert workflow.ainvoke_calls == 1
    assert runner.run_calls == []


@pytest.mark.asyncio
async def test_openai_adapter_falls_back_for_explicit_memory_reference() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("What did we work out last time?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(route="therapeutic", memory_reference_mode="explicit")
        ),
    )

    assert result == {"response_text": "fallback reply"}
    assert workflow.ainvoke_calls == 1
    assert runner.run_calls == []


@pytest.mark.asyncio
async def test_openai_adapter_falls_back_for_crisis_or_clarification() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("I might hurt myself")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="therapeutic", crisis_level=1)),
    )

    assert result == {"response_text": "fallback reply"}
    assert workflow.ainvoke_calls == 1
    assert runner.run_calls == []


@pytest.mark.asyncio
async def test_openai_adapter_shadow_runs_without_persisting_state() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("shadow reply")
    adapter = _adapter(workflow, runner)

    result = await adapter.run_shadow_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.response_text_length == len("shadow reply")
    assert result.response_text_preview == "shadow reply"
    assert result.response_text_sha256 is not None
    assert result.sdk_duration_ms is not None
    assert result.shadow_duration_ms is not None
    assert runner.run_calls
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None


@pytest.mark.asyncio
async def test_openai_adapter_shadow_reports_fallback_reason() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)

    result = await adapter.run_shadow_turn(
        cast(Any, _initial_state("What do you remember about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="memory_control")),
    )

    assert result.status == "fallback"
    assert result.eligible is False
    assert result.fallback_reason == "unsupported_route:memory_control"
    assert result.route == "memory_control"
    assert result.memory_action_type == "status"
    assert result.selected_agent is None
    assert runner.run_calls == []
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None


@pytest.mark.asyncio
async def test_openai_adapter_streams_safe_therapeutic_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("streamed reply")
    adapter = _adapter(workflow, runner)

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(Any, _initial_state()),
            config={"configurable": {"thread_id": "thread-1"}},
            context=_context(),
        )
    ]

    assert events[:3] == [
        TextRuntimeStatusEvent(stage="load_memory", turn_finalized=False),
        TextRuntimeStatusEvent(stage="therapeutic", turn_finalized=False),
        TextRuntimeChunkEvent(text="streamed reply"),
    ]
    assert events[-2] == TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    assert isinstance(events[-1], TextRuntimeStateEvent)
    assert events[-1].state["response_text"] == "streamed reply"
    assert runner.stream_calls
