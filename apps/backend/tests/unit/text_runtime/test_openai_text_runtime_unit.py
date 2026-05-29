"""Tests for the OpenAI text runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.flows.therapeutic import sanitize_response_llm_text
from agent.memory.procedural_profile import aset_proactive_recall
from agent.memory.types import TurnDispatchDecision
from agent.runtime import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.runtime_context import WorkflowContext
from agent.runtime import (
    OpenAITextRuntime,
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
)
from agent.specialists.crisis import CRISIS_AGENT_NAME
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_NAME
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.specialists.triage import TRIAGE_AGENT_NAME
from tests.support.openai_text import (
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM as _RouteLLM,
)
from tests.support.persistence import FakeCrossRestartLLM


class _StatefulWorkflow:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None


def _runtime(
    workflow: _StatefulWorkflow,  # noqa: ARG001 - keeps tests concise
    runner: FakeOpenAISDKRunner,
) -> OpenAITextRuntime:
    return OpenAITextRuntime(
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


class _RecordingResponseLLM(FakeCrossRestartLLM):
    def __init__(self, text: str, *, stream_chunks: list[str] | None = None) -> None:
        super().__init__()
        self.text = text
        self.stream_chunks = stream_chunks or [text]
        self.prompts: list[str] = []
        self.system_instructions: list[str | None] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        del use_search
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        return self.text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        for chunk in self.stream_chunks:
            yield chunk


class _StaticTriageDecisionLLM(FakeCrossRestartLLM):
    def __init__(self, decision: TurnDispatchDecision) -> None:
        super().__init__()
        self.decision = decision

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        del prompt, system_instruction, use_search
        if response_schema.__name__ == "TurnDispatchDecision":
            return self.decision
        return await super().generate_structured(
            prompt="",
            response_schema=response_schema,
            system_instruction=None,
        )


@pytest.mark.parametrize(
    ("raw_text", "expected_text"),
    [
        (
            '<tool_call>{"name":"load_therapeutic_response_skill"}</tool_call>'
            "I can help you plan the first minute.",
            "I can help you plan the first minute.",
        ),
        (
            'load_therapeutic_response_skill({"response_style":"supportive"})'
            "I can help you plan the first minute.",
            "I can help you plan the first minute.",
        ),
        (
            'to=load_therapeutic_response_skill {"response_style":"supportive"}'
            "I can help you plan the first minute.",
            "I can help you plan the first minute.",
        ),
    ],
)
def test_sanitize_response_llm_text_strips_leading_pseudo_tool_calls(
    raw_text: str,
    expected_text: str,
) -> None:
    response_text = sanitize_response_llm_text(raw_text)

    assert response_text.text == expected_text
    assert response_text.sanitized is True


@pytest.mark.asyncio
async def test_openai_runtime_runs_safe_therapeutic_turn_and_persists_state() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    runtime = _runtime(workflow, runner)

    state = await runtime.run_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == THERAPEUTIC_AGENT_NAME
    assert runner.run_calls[0]["agent"].name == runtime._roster.therapeutic_agent.name
    assert runner.run_calls[0]["agent"].handoff_description == (
        runtime._roster.therapeutic_agent.handoff_description
    )
    assert [tool.name for tool in runner.run_calls[0]["agent"].tools] == [
        "load_therapeutic_response_skill",
        "show_saved_memory",
        "show_memory_status",
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
        "list_guided_exercise_skills",
        "answer_grounded_lookup",
    ]
    assert "Write the next assistant message" in runner.run_calls[0]["input_text"]
    assert state["response_text"] == "openai reply"
    assert state["response_style"] == "supportive"
    assert state["therapeutic_approach"] == "none"
    assert state["diagnostics"]["text_agent_runtime"] == "openai"
    assert state["diagnostics"]["openai_text_route_plan_kind"] == "therapeutic"
    assert [turn["role"] for turn in state["transcript"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_response_llm_omits_tool_prompt_and_sanitizes_pseudo_tool_text() -> None:
    """Response-LLM output should keep tool traces out of user-facing text."""

    runtime = _runtime(_StatefulWorkflow(), FakeOpenAISDKRunner("unused sdk reply"))
    response_llm = _RecordingResponseLLM(
        '<tool_call>{"name":"load_therapeutic_response_skill","arguments":'
        '{"response_style":"supportive"}}</tool_call>I can help you plan '
        "the first minute."
    )
    context = WorkflowContext(
        llm_client=FakeCrossRestartLLM(),
        response_llm=response_llm,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )

    state = await runtime.run_turn(
        cast(Any, _initial_state("Can we make a tiny plan?")),
        config={"configurable": {"thread_id": "thread-response-llm"}},
        context=context,
    )

    assert "load_therapeutic_response_skill" not in response_llm.prompts[-1]
    assert "load_therapeutic_response_skill" not in (
        response_llm.system_instructions[-1] or ""
    )
    assert state["response_text"] == "I can help you plan the first minute."
    assert state["diagnostics"]["openai_response_llm_output_sanitized"] is True
    assert "openai_response_llm_raw_text_sha256" in state["diagnostics"]
    assert (
        "load_therapeutic_response_skill"
        in (state["diagnostics"]["openai_response_llm_raw_text_preview"])
    )


@pytest.mark.asyncio
async def test_triage_clarification_does_not_mutate_decision_route() -> None:
    decision = TurnDispatchDecision(
        route="grounded_lookup",
        reasoning="ambiguous grounded lookup request",
        confidence="low",
        query="grounding techniques",
    )
    runtime = _runtime(_StatefulWorkflow(), FakeOpenAISDKRunner("unused"))
    state = cast(Any, _initial_state("Maybe look this up, or maybe just help?"))

    result = await runtime._apply_triage_turn_dispatch(
        state,
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_StaticTriageDecisionLLM(decision)),
    )

    assert decision.route == "grounded_lookup"
    assert result["route"] == "therapeutic"
    assert result["turn_lifecycle"]["tentative_route"] == "grounded_lookup"
    assert result["diagnostics"]["openai_triage_tentative_route"] == "grounded_lookup"


@pytest.mark.asyncio
async def test_response_llm_stream_sanitizes_pseudo_tool_text() -> None:
    runtime = _runtime(_StatefulWorkflow(), FakeOpenAISDKRunner("unused sdk reply"))
    response_llm = _RecordingResponseLLM(
        "unused",
        stream_chunks=[
            "<tool_call>{",
            '"name":"load_therapeutic_response_skill"',
            "}</tool_call>",
            "I can help you plan the first minute.",
        ],
    )
    context = WorkflowContext(
        llm_client=FakeCrossRestartLLM(),
        response_llm=response_llm,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )

    events = [
        event
        async for event in runtime.run_turn_stream(
            cast(Any, _initial_state("Can we make a tiny plan?")),
            config={"configurable": {"thread_id": "thread-response-llm-stream"}},
            context=context,
        )
    ]

    chunks = [
        event.text for event in events if isinstance(event, TextRuntimeChunkEvent)
    ]
    state_event = next(
        event for event in events if isinstance(event, TextRuntimeStateEvent)
    )
    assert chunks == ["I can help you plan the first minute."]
    assert state_event.state["response_text"] == "I can help you plan the first minute."
    assert (
        state_event.state["diagnostics"]["openai_response_llm_output_sanitized"] is True
    )
    assert "load_therapeutic_response_skill" not in state_event.state["response_text"]


@pytest.mark.asyncio
async def test_openai_runtime_uses_therapeutic_response_skill_metadata() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "reflective reply",
        tool_calls=[
            (
                "load_therapeutic_response_skill",
                {
                    "response_style": "reflective",
                    "therapeutic_approach": "cbt",
                },
            )
        ],
    )
    runtime = _runtime(workflow, runner)

    state = await runtime.run_turn(
        cast(Any, _initial_state("I keep getting stuck in the same loop.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert state["response_text"] == "reflective reply"
    assert state["response_style"] == "reflective"
    assert state["therapeutic_approach"] == "cbt"
    assert state["diagnostics"]["openai_therapeutic_skill_tool_calls"] == [
        "load_therapeutic_response_skill"
    ]
    assert state["diagnostics"]["openai_therapeutic_skill_response_style"] == (
        "reflective"
    )


@pytest.mark.asyncio
async def test_openai_runtime_passes_sdk_session_to_safe_turn() -> None:
    """OpenAI serving turns should pass the configured SDK session through."""

    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    runtime = _runtime(workflow, runner)
    sdk_session = object()

    await runtime.run_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
        session=sdk_session,
    )

    assert runner.run_calls[0]["session"] is sdk_session


@pytest.mark.asyncio
async def test_openai_runtime_omits_prior_state_history_when_sdk_session_is_used() -> (
    None
):
    """SDK sessions own history, while retrieved memory remains prompt-visible."""

    workflow = _StatefulWorkflow()
    workflow.state = _initial_state("old turn")
    workflow.state["transcript"] = [
        {"role": "user", "content": "old user detail"},
        {"role": "assistant", "content": "old assistant detail"},
    ]
    runner = FakeOpenAISDKRunner("openai reply")
    runtime = _runtime(workflow, runner)
    context = _context()
    await context.memory_store.aput(
        ("user-1", "semantic"),
        "fact-hiking",
        {"evidence_quote": "I love hiking on weekends"},
    )
    await aset_proactive_recall(context.memory_store, user_id="user-1", enabled=True)

    await runtime.run_turn(
        cast(Any, _initial_state("hiking")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
        session=object(),
        prior_state=cast(Any, workflow.state),
    )

    prompt = runner.run_calls[0]["input_text"]
    assert "old user detail" not in prompt
    assert "old assistant detail" not in prompt
    assert "Recent conversation:\n(no prior history)" in prompt
    assert "I love hiking on weekends" in prompt
    assert "Current user message:\nuser: hiking" in prompt


@pytest.mark.asyncio
async def test_openai_runtime_passes_sdk_session_to_streamed_safe_turn() -> None:
    """Streaming OpenAI serving turns should also use the SDK session."""

    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("openai reply")
    runtime = _runtime(workflow, runner)
    sdk_session = object()

    events = [
        event
        async for event in runtime.run_turn_stream(
            cast(Any, _initial_state()),
            config={"configurable": {"thread_id": "thread-1"}},
            context=_context(),
            session=sdk_session,
        )
    ]

    assert any(isinstance(event, TextRuntimeStateEvent) for event in events)
    assert runner.stream_calls[0]["session"] is sdk_session


@pytest.mark.asyncio
async def test_openai_runtime_runs_memory_status_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
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
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == THERAPEUTIC_AGENT_NAME
    assert "Required tool:" not in runner.run_calls[0]["input_text"]


@pytest.mark.asyncio
async def test_openai_runtime_runs_saved_memory_list_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_saved_memory", {})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)
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

    result = await runtime.run_turn(
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
    assert runner.run_calls
    assert "Required tool:" not in runner.run_calls[0]["input_text"]


@pytest.mark.asyncio
async def test_openai_runtime_runs_memory_recall_update_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("set_proactive_memory_recall", {"enabled": True})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
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
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_memory_control_route_does_not_synthesize_state_action() -> (
    None
):
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("I need an explicit memory tool call to change that.")
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("Turn proactive recall on.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="memory_control")),
    )

    assert "action" not in dict(result.get("memory_control", {}) or {})
    assert result["route"] == "therapeutic"
    assert "openai_memory_tool_calls" not in result["diagnostics"]


@pytest.mark.asyncio
async def test_openai_runtime_runs_preference_save_through_sdk_tool() -> None:
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
    runtime = _runtime(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic"))

    result = await runtime.run_turn(
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
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_runs_grounded_lookup_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
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
    assert result["diagnostics"]["openai_text_route_plan_kind"] == "grounded_lookup"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "grounded_lookup"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert (
        result["diagnostics"]["openai_grounded_tool_expected"]
        == "answer_grounded_lookup"
    )
    assert result["diagnostics"]["openai_grounded_tool_calls"] == [
        "answer_grounded_lookup"
    ]
    assert runner.triage_calls
    assert runner.triage_calls[0]["agent"].name == TRIAGE_AGENT_NAME
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_records_no_verified_grounded_lookup_status() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("Can you verify whether this is current?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="grounded_lookup",
                grounded_status="no_verified_answer",
                grounded_answer="I couldn’t verify that from reliable sources.",
            )
        ),
    )

    assert result["route"] == "grounded_lookup"
    assert result["response_text"] == "I couldn’t verify that from reliable sources."
    assert result["grounded_lookup"]["status"] == "no_verified_answer"
    assert result["diagnostics"]["openai_grounded_tool_expected"] == (
        "answer_grounded_lookup"
    )
    assert result["diagnostics"]["openai_grounded_tool_calls"] == [
        "answer_grounded_lookup"
    ]
    assert result["diagnostics"]["openai_grounded_tool_fallback"] is False
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_handles_pending_memory_cancel() -> None:
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
    runner = FakeOpenAISDKRunner(
        tool_calls=[("cancel_memory_deletion", {})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("cancel that")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="memory_control",
                active_flow_action="continue",
            )
        ),
        prior_state=cast(Any, workflow.state),
    )

    assert result["route"] == "memory_control"
    assert result["response_text"] == "Cancelled. I didn't change your memory."
    assert result["memory_control"]["pending_action"] is None
    assert result["diagnostics"]["openai_memory_tool_expected"] == (
        "cancel_memory_deletion"
    )
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_confirms_pending_memory_deletion_through_sdk_tool() -> (
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
    runner = FakeOpenAISDKRunner(
        tool_calls=[("confirm_memory_deletion", {})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("yes, delete it")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
        prior_state=cast(Any, workflow.state),
    )

    assert result["route"] == "memory_control"
    assert result["response_text"] == "Deleted that saved fact."
    assert result["diagnostics"]["openai_memory_tool_expected"] == (
        "confirm_memory_deletion"
    )
    assert result["diagnostics"]["openai_memory_tool_side_effects"] == ["delete_memory"]
    assert await context.memory_store.arecord_count(("user-1", "semantic")) == 0
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_does_not_mutate_memory_without_sdk_tool_call() -> None:
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
    runner = FakeOpenAISDKRunner("I will not change memory without the tool.")
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("yes, delete it")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
        prior_state=cast(Any, workflow.state),
    )

    assert result["route"] == "therapeutic"
    assert result["response_text"] == "I will not change memory without the tool."
    assert result["memory_control"]["pending_action"]["target"]["key"] == "fact-1"
    assert "openai_memory_tool_calls" not in result["diagnostics"]
    assert await context.memory_store.arecord_count(("user-1", "semantic")) == 1
    assert runner.run_calls
    assert "Required tool:" not in runner.run_calls[0]["input_text"]


@pytest.mark.asyncio
async def test_openai_runtime_preserves_exercise_during_app_owned_side_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)
    state = _initial_state("Can you look this up before we continue?")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 1,
    }

    result = await runtime.run_turn(
        cast(Any, state),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="grounded_lookup",
                active_flow_action="preserve",
                therapeutic_response_style="clarifying",
            )
        ),
    )

    assert result["route"] == "grounded_lookup"
    assert result["response_style"] == "grounded_lookup"
    assert result["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
    assert result["exercise_state"]["exercise_step"] == 1
    assert result["diagnostics"]["openai_text_runtime_mode"] == "grounded_lookup"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert runner.run_calls
    assert runner.stream_calls == []


@pytest.mark.asyncio
async def test_openai_runtime_starts_guided_exercise_with_guided_agent() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "guided start",
        tool_calls=[
            (
                "load_guided_exercise_skill",
                {
                    "exercise_type": "grounding_box_breathing",
                    "current_step_index": 0,
                    "runtime_action": "start",
                },
            )
        ],
    )
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
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
    assert runner.stream_calls
    sdk_call = runner.stream_calls[0]
    assert sdk_call["agent"].name == GUIDED_EXERCISE_AGENT_NAME
    assert sdk_call["agent"].name == runtime._roster.guided_exercise_agent.name
    assert sdk_call["agent"].handoff_description == (
        runtime._roster.guided_exercise_agent.handoff_description
    )
    assert [tool.name for tool in sdk_call["agent"].tools] == [
        "load_guided_exercise_skill",
        "record_guided_exercise_progress",
    ]
    assert "Required tool: load_guided_exercise_skill" in sdk_call["input_text"]
    assert "grounding_box_breathing" in sdk_call["input_text"]
    assert result["diagnostics"]["openai_guided_exercise_tool_calls"] == [
        "load_guided_exercise_skill"
    ]
    assert result["diagnostics"]["openai_guided_exercise_tool_fallback"] is False


@pytest.mark.asyncio
async def test_openai_runtime_continues_active_guided_exercise() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "next step",
        tool_calls=[
            (
                "load_guided_exercise_skill",
                {
                    "exercise_type": "grounding_5_4_3_2_1",
                    "current_step_index": 1,
                    "runtime_action": "advance",
                },
            )
        ],
    )
    runtime = _runtime(workflow, runner)
    state = _initial_state("lamp, window, mug")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_step_id": "see",
        "exercise_therapeutic_approach": "none",
    }

    result = await runtime.run_turn(
        cast(Any, state),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="therapeutic",
                active_flow_action="continue",
                therapeutic_response_style="guided_exercise",
            )
        ),
    )

    assert result["response_text"] == "next step"
    assert result["response_style"] == "guided_exercise"
    assert result["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
    assert result["exercise_state"]["exercise_step"] == 1
    assert result["exercise_state"]["exercise_step_id"] == "hear"
    assert result["diagnostics"]["openai_selected_agent"] == GUIDED_EXERCISE_AGENT_NAME
    assert result["diagnostics"]["openai_guided_exercise_tool_calls"] == [
        "load_guided_exercise_skill"
    ]
    assert result["diagnostics"]["openai_guided_exercise_tool_runtime_action"] == (
        "advance"
    )
    assert runner.stream_calls
    sdk_call = runner.stream_calls[0]
    assert sdk_call["agent"].name == GUIDED_EXERCISE_AGENT_NAME
    assert sdk_call["agent"].name == runtime._roster.guided_exercise_agent.name
    assert sdk_call["agent"].handoff_description == (
        runtime._roster.guided_exercise_agent.handoff_description
    )


@pytest.mark.asyncio
async def test_openai_runtime_handles_explicit_memory_reference_on_sdk_path() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("memory-aware reply")
    runtime = _runtime(workflow, runner)

    result = await runtime.run_turn(
        cast(Any, _initial_state("What did we work out last time?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(route="therapeutic", memory_reference_mode="explicit")
        ),
    )

    assert result["route"] == "therapeutic"
    assert result["response_text"] == "memory-aware reply"
    assert result["response_style"] == "supportive"
    assert result["memory_reference"]["mode"] == "explicit"
    assert result["diagnostics"]["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_uses_crisis_agent_for_safety_clarification() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("Are you in immediate danger right now?")
    runtime = _runtime(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=1))

    result = await runtime.run_turn(
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
    assert runner.run_calls[0]["agent"].name == runtime._roster.crisis_agent.name
    assert runner.run_calls[0]["agent"].handoff_description == (
        runtime._roster.crisis_agent.handoff_description
    )
    assert runner.run_calls[0]["agent"].tools == []
    assert "Safety-check override" in runner.run_calls[0]["agent"].instructions
    assert await context.crisis_log_backend.arecord_count() == 0


@pytest.mark.asyncio
async def test_openai_runtime_uses_crisis_agent_for_crisis_response() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "Please call local emergency services now.",
        tool_calls=[("lookup_crisis_resources", {})],
    )
    runtime = _runtime(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=3))

    result = await runtime.run_turn(
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
    assert result["diagnostics"]["openai_crisis_tool_expected"] == (
        "lookup_crisis_resources"
    )
    assert result["diagnostics"]["openai_crisis_tool_calls"] == [
        "lookup_crisis_resources"
    ]
    assert result["diagnostics"]["openai_crisis_tool_fallback"] is False
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME
    assert runner.run_calls[0]["agent"].name == runtime._roster.crisis_agent.name
    assert runner.run_calls[0]["agent"].handoff_description == (
        runtime._roster.crisis_agent.handoff_description
    )
    assert [tool.name for tool in runner.run_calls[0]["agent"].tools] == [
        "lookup_crisis_resources",
        "get_crisis_support_template",
    ]
    assert [tool.name for tool in runtime._roster.crisis_agent.tools] == [
        "lookup_crisis_resources",
        "get_crisis_support_template",
    ]
    assert "Required tool: lookup_crisis_resources" in runner.run_calls[0]["input_text"]
    assert await context.crisis_log_backend.arecord_count() == 1


@pytest.mark.asyncio
async def test_openai_runtime_records_crisis_no_verified_resource_status() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "No verified, actionable local crisis line was found. Please contact local emergency services now.",
        tool_calls=[("lookup_crisis_resources", {})],
    )
    runtime = _runtime(workflow, runner)
    context = _context(
        _RouteLLM(
            route="therapeutic",
            crisis_level=3,
            crisis_location_status="provided",
            crisis_location="Singapore",
            crisis_resource_status="no_verified_results",
        )
    )

    result = await runtime.run_turn(
        cast(Any, _initial_state("I'm in Singapore and I am about to hurt myself.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    assert result["route"] == "crisis"
    assert result["response_style"] == "crisis_response"
    assert result["resource_lookup_status"] == "no_verified_results"
    assert result["found_resources"] == []
    assert result["diagnostics"]["openai_crisis_tool_expected"] == (
        "lookup_crisis_resources"
    )
    assert result["diagnostics"]["openai_crisis_tool_calls"] == [
        "lookup_crisis_resources"
    ]
    assert result["diagnostics"]["openai_crisis_tool_fallback"] is False
    assert runner.run_calls


@pytest.mark.asyncio
async def test_openai_runtime_shadow_runs_without_persisting_state() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("shadow reply")
    runtime = _runtime(workflow, runner)

    result = await runtime.run_shadow_turn(
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


@pytest.mark.asyncio
async def test_openai_runtime_shadow_keeps_tools_disabled_for_memory_requests() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("shadow memory reply")
    runtime = _runtime(workflow, runner)

    result = await runtime.run_shadow_turn(
        cast(Any, _initial_state("What do you remember about me?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(),
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.fallback_reason is None
    assert result.route == "therapeutic"
    assert result.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.response_text_length == len("shadow memory reply")
    assert runner.run_calls
    assert runner.run_calls[0]["agent"].tools == []


@pytest.mark.asyncio
async def test_openai_runtime_shadow_reports_guided_exercise_agent() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    runtime = _runtime(workflow, runner)

    result = await runtime.run_shadow_turn(
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


@pytest.mark.asyncio
async def test_openai_runtime_shadow_reports_grounded_lookup_route() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    runtime = _runtime(workflow, runner)

    result = await runtime.run_shadow_turn(
        cast(Any, _initial_state("Can you look up the current rule?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(_RouteLLM(route="grounded_lookup")),
    )

    assert result.status == "eligible"
    assert result.eligible is True
    assert result.route == "grounded_lookup"
    assert result.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.grounded_lookup_query == "grounded query"
    assert result.response_text_length is None
    assert runner.run_calls == []
    assert runner.stream_calls == []


@pytest.mark.asyncio
async def test_openai_runtime_shadow_reports_crisis_agent_without_side_effects() -> (
    None
):
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner()
    runtime = _runtime(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=2))

    result = await runtime.run_shadow_turn(
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
    assert await context.crisis_log_backend.arecord_count() == 0


@pytest.mark.asyncio
async def test_openai_runtime_streams_safe_therapeutic_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("streamed reply")
    runtime = _runtime(workflow, runner)

    events = [
        event
        async for event in runtime.run_turn_stream(
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
async def test_openai_runtime_streams_guided_exercise_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "guided chunk",
        tool_calls=[
            (
                "load_guided_exercise_skill",
                {
                    "exercise_type": "grounding_5_4_3_2_1",
                    "current_step_index": 0,
                    "runtime_action": "start",
                },
            )
        ],
    )
    runtime = _runtime(workflow, runner)

    events = [
        event
        async for event in runtime.run_turn_stream(
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
    assert events[-1].state["diagnostics"]["openai_guided_exercise_tool_calls"] == [
        "load_guided_exercise_skill"
    ]
    assert runner.stream_calls
    sdk_call = runner.stream_calls[0]
    assert sdk_call["agent"].name == GUIDED_EXERCISE_AGENT_NAME
    assert sdk_call["agent"].name == runtime._roster.guided_exercise_agent.name
    assert sdk_call["agent"].handoff_description == (
        runtime._roster.guided_exercise_agent.handoff_description
    )


@pytest.mark.asyncio
async def test_openai_runtime_streams_crisis_response_turn() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        "streamed crisis reply",
        tool_calls=[("lookup_crisis_resources", {})],
    )
    runtime = _runtime(workflow, runner)
    context = _context(_RouteLLM(route="therapeutic", crisis_level=3))

    events = [
        event
        async for event in runtime.run_turn_stream(
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
    assert events[-1].state["diagnostics"]["openai_crisis_tool_calls"] == [
        "lookup_crisis_resources"
    ]
    assert events[-1].state["diagnostics"]["openai_crisis_tool_fallback"] is False
    assert runner.stream_calls
    assert await context.crisis_log_backend.arecord_count() == 1


@pytest.mark.asyncio
async def test_openai_runtime_streams_memory_control_turn_through_sdk_tool() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner(
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    runtime = _runtime(workflow, runner)

    events = [
        event
        async for event in runtime.run_turn_stream(
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
