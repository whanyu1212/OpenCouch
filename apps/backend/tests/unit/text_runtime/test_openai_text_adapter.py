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
from agent.text_runtime.openai_agents import (
    CRISIS_AGENT_NAME,
    GUIDED_EXERCISE_AGENT_NAME,
    THERAPEUTIC_AGENT_NAME,
)
from tests.support.openai_text import (
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM as _RouteLLM,
)
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


def _adapter(
    workflow: _StatefulWorkflow,
    runner: FakeOpenAISDKRunner,
) -> OpenAITextAgentAdapter:
    return OpenAITextAgentAdapter(
        checkpoint_adapter=LangGraphTextAgentAdapter(cast(Any, workflow)),
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
    assert [tool.name for tool in runner.run_calls[0]["agent"].tools] == [
        "show_saved_memory",
        "show_memory_status",
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
        "answer_grounded_lookup",
    ]
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
async def test_openai_adapter_passes_sdk_session_to_safe_turn() -> None:
    """OpenAI serving turns should pass the configured SDK session through."""

    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    adapter = _adapter(workflow, runner)
    sdk_session = object()

    await adapter.run_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
        session=sdk_session,
    )

    assert runner.run_calls[0]["session"] is sdk_session


@pytest.mark.asyncio
async def test_openai_adapter_passes_sdk_session_to_streamed_safe_turn() -> None:
    """Streaming OpenAI serving turns should also use the SDK session."""

    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    adapter = _adapter(workflow, runner)
    sdk_session = object()

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(Any, _initial_state()),
            config={"configurable": {"thread_id": "thread-1"}},
            context=_context(),
            session=sdk_session,
        )
    ]

    assert any(isinstance(event, TextRuntimeStateEvent) for event in events)
    assert runner.stream_calls[0]["session"] is sdk_session


@pytest.mark.asyncio
async def test_openai_adapter_runs_memory_status_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("What do you remember about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result["route"] == "memory_control"
    assert result["response_style"] == "memory_control"
    assert "Memory status:" in result["response_text"]
    assert result["diagnostics"]["text_agent_runtime"] == "openai"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "memory_control"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert result["diagnostics"]["openai_memory_tool_expected"] == "show_memory_status"
    assert result["diagnostics"]["openai_memory_tool_calls"] == ["show_memory_status"]
    assert result["diagnostics"]["openai_memory_tool_fallback"] is False
    assert result["memory_control"]["pending_action"] is None
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == THERAPEUTIC_AGENT_NAME
    assert "Required tool:" not in runner.run_calls[0]["input_text"]


@pytest.mark.asyncio
async def test_openai_adapter_runs_saved_memory_list_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_saved_memory", {})],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)
    context = _context()
    await context.memory_store.aput(
        ("user-1", "semantic"),
        "fact-presentations",
        {
            "category": "trigger",
            "predicate": "WORRIES_ABOUT",
            "object": {"identifier": "presentations"},
            "evidence_quote": "Presentations make me anxious.",
        },
    )

    result = await adapter.run_turn(
        cast(Any, _initial_state("What is saved in memory about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "memory_control"
    assert result["response_style"] == "memory_control"
    assert "Here's what I currently have saved:" in result["response_text"]
    assert "Presentations make me anxious." in result["response_text"]
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert result["diagnostics"]["openai_memory_tool_expected"] == "show_saved_memory"
    assert result["diagnostics"]["openai_memory_tool_calls"] == ["show_saved_memory"]
    assert result["memory_control"]["pending_action"] is None
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls
    assert "Required tool:" not in runner.run_calls[0]["input_text"]


@pytest.mark.asyncio
async def test_openai_adapter_runs_memory_recall_update_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("set_proactive_memory_recall", {"enabled": True})],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("Turn proactive recall on.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result["route"] == "memory_control"
    assert "I turned proactive recall on." in result["response_text"]
    assert result["procedural_profile"]["proactive_recall_enabled"] is True
    assert result["memory_control"]["pending_action"] is None
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert (
        result["diagnostics"]["openai_memory_tool_expected"]
        == "set_proactive_memory_recall"
    )
    assert result["diagnostics"]["openai_memory_tool_calls"] == [
        "set_proactive_memory_recall"
    ]
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_runs_preference_save_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[
            (
                "save_response_preference",
                {"preference_text": "direct answers when I am spiraling"},
            )
        ],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic"))

    result = await adapter.run_turn(
        cast(Any, _initial_state("Please give me direct answers when I spiral.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "memory_control"
    assert result["response_text"].startswith("Saved:")
    assert result["diagnostics"]["openai_memory_tool_expected"] == (
        "save_response_preference"
    )
    assert result["diagnostics"]["openai_memory_tool_side_effects"] == [
        "procedural_profile_update"
    ]
    assert await context.memory_store.arecord_count(("user-1", "procedural")) == 1
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_runs_grounded_lookup_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("Can you look up the current rule?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="grounded_lookup")),
    )

    assert result["route"] == "grounded_lookup"
    assert result["response_style"] == "grounded_lookup"
    assert result["response_text"] == "Official answer.\n\nSources:\n- Official source"
    assert result["grounded_lookup"]["query"] == "grounded query"
    assert result["grounded_lookup"]["status"] == "answered"
    assert result["diagnostics"]["text_agent_runtime"] == "openai"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "grounded_lookup"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert (
        result["diagnostics"]["openai_grounded_tool_expected"]
        == "answer_grounded_lookup"
    )
    assert result["diagnostics"]["openai_grounded_tool_calls"] == [
        "answer_grounded_lookup"
    ]
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_handles_pending_memory_cancel() -> None:
    workflow = _StatefulWorkflow()
    workflow.state = _initial_state("Please delete that saved fact")
    workflow.state["memory_control"] = {
        "pending_action": {
            "type": "delete",
            "target": {
                "kind": "fact",
                "id": "fact-1",
                "key": "fact-1",
                "namespace": ["user-1", "semantic"],
                "preview": "Presentations make me anxious.",
            },
        },
    }
    runner = FakeOpenAISDKRunner(invoke_required_tool=True)
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("cancel that")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="memory_control",
                memory_action_type="cancel_pending",
                active_flow_action="continue",
            )
        ),
    )

    assert result["route"] == "memory_control"
    assert result["response_text"] == "Cancelled. I didn't change your memory."
    assert result["memory_control"]["pending_action"] is None
    assert result["diagnostics"]["openai_memory_tool_expected"] == (
        "cancel_memory_deletion"
    )
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_confirms_pending_memory_deletion_through_sdk_tool() -> (
    None
):
    workflow = _StatefulWorkflow()
    workflow.state = _initial_state("Please delete that saved fact")
    workflow.state["memory_control"] = {
        "pending_action": {
            "type": "delete",
            "target": {
                "kind": "fact",
                "id": "fact-1",
                "key": "fact-1",
                "namespace": ["user-1", "semantic"],
                "preview": "Presentations make me anxious.",
            },
        },
    }
    context = _context(
        _RouteLLM(
            route="memory_control",
            memory_action_type="confirm_pending",
            active_flow_action="continue",
        )
    )
    await context.memory_store.aput(
        ("user-1", "semantic"),
        "fact-1",
        {
            "category": "trigger",
            "predicate": "WORRIES_ABOUT",
            "object": {"identifier": "presentations"},
            "evidence_quote": "Presentations make me anxious.",
        },
    )
    runner = FakeOpenAISDKRunner(invoke_required_tool=True)
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("yes, delete it")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "memory_control"
    assert result["response_text"] == "Deleted that saved fact."
    assert result["diagnostics"]["openai_memory_tool_expected"] == (
        "confirm_memory_deletion"
    )
    assert result["diagnostics"]["openai_memory_tool_side_effects"] == ["delete_memory"]
    assert await context.memory_store.arecord_count(("user-1", "semantic")) == 0
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_preserves_exercise_during_app_owned_side_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(invoke_required_tool=True)
    adapter = _adapter(workflow, runner)
    state = _initial_state("Can you look this up before we continue?")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 1,
    }

    result = await adapter.run_turn(
        cast(Any, state),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(route="grounded_lookup", active_flow_action="preserve")
        ),
    )

    assert result["route"] == "grounded_lookup"
    assert result["response_style"] == "grounded_lookup"
    assert result["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
    assert result["exercise_state"]["exercise_step"] == 1
    assert result["diagnostics"]["openai_text_runtime_mode"] == "grounded_lookup"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls
    assert runner.stream_calls == []


@pytest.mark.asyncio
async def test_openai_adapter_starts_guided_exercise_with_guided_agent() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("guided start")
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("Can you walk me through box breathing?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="therapeutic",
                therapeutic_response_style="guided_exercise",
                exercise_start_basis="explicit_user_request",
                exercise_type="grounding_box_breathing",
            )
        ),
    )

    assert result["response_text"] == "guided start"
    assert result["response_style"] == "guided_exercise"
    assert result["exercise_state"]["exercise_type"] == "grounding_box_breathing"
    assert result["exercise_state"]["exercise_step"] == 0
    assert result["exercise_state"]["exercise_step_id"] == "inhale"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "guided_exercise"
    assert result["diagnostics"]["openai_selected_agent"] == GUIDED_EXERCISE_AGENT_NAME
    assert workflow.ainvoke_calls == 0
    assert runner.stream_calls
    sdk_call = runner.stream_calls[0]
    assert sdk_call["agent"].name == GUIDED_EXERCISE_AGENT_NAME
    assert "Exercise skill:" in sdk_call["input_text"]
    assert "grounding_box_breathing" in sdk_call["input_text"]


@pytest.mark.asyncio
async def test_openai_adapter_continues_active_guided_exercise() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("next step")
    adapter = _adapter(workflow, runner)
    state = _initial_state("lamp, window, mug")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_step_id": "see",
        "exercise_therapeutic_approach": "none",
    }

    result = await adapter.run_turn(
        cast(Any, state),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="therapeutic", active_flow_action="continue")),
    )

    assert result["response_text"] == "next step"
    assert result["response_style"] == "guided_exercise"
    assert result["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
    assert result["exercise_state"]["exercise_step"] == 1
    assert result["exercise_state"]["exercise_step_id"] == "hear"
    assert result["diagnostics"]["openai_selected_agent"] == GUIDED_EXERCISE_AGENT_NAME
    assert workflow.ainvoke_calls == 0
    assert runner.stream_calls


@pytest.mark.asyncio
async def test_openai_adapter_handles_explicit_memory_reference_on_sdk_path() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("memory-aware reply")
    adapter = _adapter(workflow, runner)

    result = await adapter.run_turn(
        cast(Any, _initial_state("What did we work out last time?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result["route"] == "therapeutic"
    assert result["response_text"] == "memory-aware reply"
    assert result["response_style"] == "supportive"
    assert result["memory_reference"]["mode"] == "explicit"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert workflow.ainvoke_calls == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_adapter_uses_crisis_agent_for_safety_clarification() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("Are you in immediate danger right now?")
    adapter = _adapter(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=1))

    result = await adapter.run_turn(
        cast(Any, _initial_state("I might hurt myself")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "therapeutic"
    assert result["response_text"] == "Are you in immediate danger right now?"
    assert result["response_style"] == "clarifying"
    assert result["crisis"].level == 1
    assert result["diagnostics"]["text_agent_runtime"] == "openai"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "crisis_clarification"
    assert result["diagnostics"]["openai_selected_agent"] == CRISIS_AGENT_NAME
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME
    assert "Safety-check override" in runner.run_calls[0]["agent"].instructions
    assert workflow.ainvoke_calls == 0
    assert await context.crisis_log_backend.arecord_count() == 0


@pytest.mark.asyncio
async def test_openai_adapter_uses_crisis_agent_for_crisis_response() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("Please call local emergency services now.")
    adapter = _adapter(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=3))

    result = await adapter.run_turn(
        cast(Any, _initial_state("I'm in Singapore and I will kill myself tonight.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "crisis"
    assert result["response_text"] == "Please call local emergency services now."
    assert result["response_style"] == "crisis_response"
    assert result["crisis"].level == 3
    assert result["resource_lookup_status"] == "found"
    assert result["found_resources"][0]["phone"] == "1767"
    assert result["diagnostics"]["text_agent_runtime"] == "openai"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "crisis_response"
    assert result["diagnostics"]["openai_selected_agent"] == CRISIS_AGENT_NAME
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME
    assert "Verified local crisis resources" in runner.run_calls[0]["input_text"]
    assert workflow.ainvoke_calls == 0
    assert await context.crisis_log_backend.arecord_count() == 1


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
    assert runner.run_calls[0]["agent"].tools == []
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None


@pytest.mark.asyncio
async def test_openai_adapter_shadow_keeps_tools_disabled_for_memory_requests() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("shadow memory reply")
    adapter = _adapter(workflow, runner)

    result = await adapter.run_shadow_turn(
        cast(Any, _initial_state("What do you remember about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.fallback_reason is None
    assert result.route == "therapeutic"
    assert result.memory_action_type is None
    assert result.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.response_text_length == len("shadow memory reply")
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].tools == []
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None


@pytest.mark.asyncio
async def test_openai_adapter_shadow_reports_guided_exercise_agent() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)

    result = await adapter.run_shadow_turn(
        cast(Any, _initial_state("Can you guide me through grounding?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="therapeutic",
                therapeutic_response_style="guided_exercise",
                exercise_start_basis="explicit_user_request",
            )
        ),
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.selected_agent == GUIDED_EXERCISE_AGENT_NAME
    assert result.response_text_length is None
    assert runner.run_calls == []
    assert runner.stream_calls == []
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None


@pytest.mark.asyncio
async def test_openai_adapter_shadow_reports_crisis_agent_without_side_effects() -> (
    None
):
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    adapter = _adapter(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=2))

    result = await adapter.run_shadow_turn(
        cast(Any, _initial_state("I want to end my life.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.crisis_level == 2
    assert result.needs_crisis_response is True
    assert result.selected_agent == CRISIS_AGENT_NAME
    assert result.response_text_length is None
    assert runner.run_calls == []
    assert workflow.ainvoke_calls == 0
    assert workflow.state is None
    assert await context.crisis_log_backend.arecord_count() == 0


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


@pytest.mark.asyncio
async def test_openai_adapter_streams_guided_exercise_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("guided chunk")
    adapter = _adapter(workflow, runner)

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(Any, _initial_state("Can we do a grounding exercise?")),
            config={"configurable": {"thread_id": "thread-1"}},
            context=_context(
                _RouteLLM(
                    route="therapeutic",
                    therapeutic_response_style="guided_exercise",
                    exercise_start_basis="explicit_user_request",
                )
            ),
        )
    ]

    assert events[:3] == [
        TextRuntimeStatusEvent(stage="load_memory", turn_finalized=False),
        TextRuntimeStatusEvent(stage="guided_exercise", turn_finalized=False),
        TextRuntimeChunkEvent(text="guided chunk"),
    ]
    assert events[-2] == TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    assert isinstance(events[-1], TextRuntimeStateEvent)
    assert events[-1].state["response_text"] == "guided chunk"
    assert events[-1].state["response_style"] == "guided_exercise"
    assert (
        events[-1].state["diagnostics"]["openai_selected_agent"]
        == GUIDED_EXERCISE_AGENT_NAME
    )
    assert runner.stream_calls


@pytest.mark.asyncio
async def test_openai_adapter_streams_crisis_response_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("streamed crisis reply")
    adapter = _adapter(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=3))

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(Any, _initial_state("I'm in Singapore and I may act tonight.")),
            config={"configurable": {"thread_id": "thread-1"}},
            context=context,
        )
    ]

    assert events[:3] == [
        TextRuntimeStatusEvent(stage="crisis_resource_lookup", turn_finalized=False),
        TextRuntimeStatusEvent(stage="crisis_response", turn_finalized=False),
        TextRuntimeChunkEvent(text="streamed crisis reply"),
    ]
    assert TextRuntimeStatusEvent(stage="crisis_log", turn_finalized=False) in events
    assert events[-2] == TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    assert isinstance(events[-1], TextRuntimeStateEvent)
    assert events[-1].state["route"] == "crisis"
    assert events[-1].state["response_style"] == "crisis_response"
    assert events[-1].state["diagnostics"]["openai_selected_agent"] == CRISIS_AGENT_NAME
    assert runner.stream_calls
    assert await context.crisis_log_backend.arecord_count() == 1


@pytest.mark.asyncio
async def test_openai_adapter_streams_memory_control_turn_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    adapter = _adapter(workflow, runner)

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(Any, _initial_state("What is my memory status?")),
            config={"configurable": {"thread_id": "thread-1"}},
            context=_context(),
        )
    ]

    assert events[0] == TextRuntimeStatusEvent(
        stage="load_memory",
        turn_finalized=False,
    )
    assert events[-2] == TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    assert isinstance(events[-1], TextRuntimeStateEvent)
    assert events[-1].state["route"] == "memory_control"
    assert "Memory status:" in events[-1].state["response_text"]
    assert events[-1].state["diagnostics"]["openai_selected_agent"] == (
        THERAPEUTIC_AGENT_NAME
    )
    assert events[-1].state["diagnostics"]["openai_memory_tool_calls"] == [
        "show_memory_status"
    ]
    assert runner.run_calls == []
    assert runner.stream_calls
