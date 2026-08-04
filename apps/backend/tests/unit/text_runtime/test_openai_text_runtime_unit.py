"""Tests for the OpenAI text runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.flows.therapeutic import (
    TherapeuticResponseLLMOutput,
    sanitize_response_llm_text,
)
from agent.memory.operations.procedural_profile import aset_proactive_recall
from agent.runtime.dispatch_models import TurnDispatchDecision
from agent.runtime import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.runtime.triage_dispatch import apply_triage_decision_to_state
from agent.runtime.workflow_context import PrefetchedTurnMemory, WorkflowContext
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
        self.text_calls = 0
        self.stream_calls = 0
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        del use_search
        self.text_calls += 1
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        return self.text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        for chunk in self.stream_chunks:
            yield chunk

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        del use_search
        self.structured_calls += 1
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        if response_schema is TherapeuticResponseLLMOutput:
            return response_schema(response_text=self.text)
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
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
async def test_response_llm_uses_structured_contract_and_omits_tool_prompt() -> None:
    """Response-LLM output should use the structured final-text contract."""

    runtime = _runtime(_StatefulWorkflow(), FakeOpenAISDKRunner("unused sdk reply"))
    response_llm = _RecordingResponseLLM("I can help you plan the first minute.")
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

    assert response_llm.structured_calls == 1
    assert response_llm.text_calls == 0
    assert "load_therapeutic_response_skill" not in response_llm.prompts[-1]
    assert "load_therapeutic_response_skill" not in (
        response_llm.system_instructions[-1] or ""
    )
    assert state["response_text"] == "I can help you plan the first minute."
    assert state["diagnostics"]["openai_response_llm_output_structured"] is True
    assert state["diagnostics"]["openai_response_llm_output_sanitized"] is False
    assert state["diagnostics"]["openai_response_llm_response_text_length"] == 37
    assert "openai_response_llm_raw_text_sha256" not in state["diagnostics"]


@pytest.mark.asyncio
async def test_response_llm_uses_triage_selected_style_guidance() -> None:
    """Response-LLM overrides should receive dynamic style guidance too."""

    runtime = _runtime(_StatefulWorkflow(), FakeOpenAISDKRunner("unused sdk reply"))
    response_llm = _RecordingResponseLLM("Try naming one thing you can do next.")
    context = WorkflowContext(
        llm_client=_RouteLLM(
            route="therapeutic",
            therapeutic_response_style="technique",
            therapeutic_approach="dbt_skills",
        ),
        response_llm=response_llm,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )

    state = await runtime.run_turn(
        cast(Any, _initial_state("Can you give me a concrete skill?")),
        config={"configurable": {"thread_id": "thread-response-llm-style"}},
        context=context,
    )

    assert state["response_style"] == "technique"
    assert state["therapeutic_approach"] == "dbt_skills"
    assert response_llm.structured_calls == 1
    system_instruction = response_llm.system_instructions[-1] or ""
    assert "Therapeutic response guidance:" in system_instruction
    assert "- response_style: technique" in system_instruction
    assert "- therapeutic_approach: dbt_skills" in system_instruction
    assert "TECHNIQUE response style" in system_instruction


@pytest.mark.asyncio
async def test_response_llm_structured_contract_sanitizes_pseudo_tool_text() -> None:
    """Structured response text is still sanitized before user exposure."""

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
        config={"configurable": {"thread_id": "thread-response-llm-sanitize"}},
        context=context,
    )

    assert response_llm.structured_calls == 1
    assert response_llm.text_calls == 0
    assert state["response_text"] == "I can help you plan the first minute."
    assert state["diagnostics"]["openai_response_llm_output_structured"] is True
    assert state["diagnostics"]["openai_response_llm_output_sanitized"] is True
    assert "openai_response_llm_raw_text_sha256" in state["diagnostics"]
    assert (
        "load_therapeutic_response_skill"
        in state["diagnostics"]["openai_response_llm_raw_text_preview"]
    )


def test_triage_clarification_does_not_mutate_decision_route() -> None:
    decision = TurnDispatchDecision(
        route="grounded_lookup",
        reasoning="ambiguous grounded lookup request",
        confidence="low",
        query="grounding techniques",
    )
    state = cast(Any, _initial_state("Maybe look this up, or maybe just help?"))

    result = apply_triage_decision_to_state(state, decision)

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
    assert response_llm.stream_calls == 1
    assert response_llm.structured_calls == 0
    assert chunks == ["I can help you plan the first minute."]
    assert state_event.state["response_text"] == "I can help you plan the first minute."
    assert (
        state_event.state["diagnostics"]["openai_response_llm_output_structured"]
        is False
    )
    assert (
        state_event.state["diagnostics"]["openai_response_llm_output_sanitized"] is True
    )
    assert "load_therapeutic_response_skill" not in state_event.state["response_text"]


@pytest.mark.asyncio
async def test_openai_runtime_injects_therapeutic_style_guidance() -> None:
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("reflective reply")
    runtime = _runtime(workflow, runner)

    state = await runtime.run_turn(
        cast(Any, _initial_state("I keep getting stuck in the same loop.")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=_context(
            _RouteLLM(
                route="therapeutic",
                therapeutic_response_style="reflective",
                therapeutic_approach="cbt",
            )
        ),
    )

    assert state["response_text"] == "reflective reply"
    assert state["response_style"] == "reflective"
    assert state["therapeutic_approach"] == "cbt"
    assert "Therapeutic response guidance:" in runner.run_calls[0]["input_text"]
    assert "- response_style: reflective" in runner.run_calls[0]["input_text"]
    assert "- therapeutic_approach: cbt" in runner.run_calls[0]["input_text"]
    assert "load_therapeutic_response_skill" not in runner.run_calls[0]["input_text"]
    assert (
        state["diagnostics"]["openai_therapeutic_style_guidance_response_style"]
        == "reflective"
    )
    assert state["diagnostics"]["openai_therapeutic_style_guidance_approach"] == "cbt"
    assert state["diagnostics"]["openai_triage_therapeutic_response_style"] == (
        "reflective"
    )
    assert state["diagnostics"]["openai_triage_therapeutic_approach"] == "cbt"
    assert "openai_therapeutic_skill_tool_calls" not in state["diagnostics"]


def test_triage_decision_sets_therapeutic_style_guidance_state() -> None:
    state = cast(Any, _initial_state("Can you give me a concrete skill?"))
    decision = TurnDispatchDecision(
        route="therapeutic",
        therapeutic_response_style="technique",
        therapeutic_approach="dbt_skills",
        reasoning="user asked for a concrete coping skill",
        confidence="high",
    )

    result = apply_triage_decision_to_state(state, decision)

    assert result["route"] == "therapeutic"
    assert result["response_style"] == "technique"
    assert result["therapeutic_approach"] == "dbt_skills"
    assert result["diagnostics"]["openai_triage_therapeutic_response_style"] == (
        "technique"
    )
    assert result["diagnostics"]["openai_triage_therapeutic_approach"] == "dbt_skills"


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
    ]
    assert "Required tool: load_guided_exercise_skill" in sdk_call["input_text"]
    assert "grounding_box_breathing" in sdk_call["input_text"]
    assert result["diagnostics"]["openai_guided_exercise_tool_calls"] == [
        "load_guided_exercise_skill"
    ]
    assert result["diagnostics"]["openai_guided_exercise_tool_fallback"] is False


@pytest.mark.asyncio
async def test_openai_runtime_falls_back_when_guided_stream_skips_required_skill_tool() -> (
    None
):
    workflow = _StatefulWorkflow()
    runner = FakeOpenAISDKRunner("fallback guided start")
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

    assert result["response_text"] == "fallback guided start"
    assert result["response_style"] == "guided_exercise"
    assert result["diagnostics"]["openai_text_runtime_mode"] == "guided_exercise"
    assert result["diagnostics"]["openai_guided_exercise_tool_calls"] == []
    assert result["diagnostics"]["openai_guided_exercise_tool_fallback"] is True
    assert runner.stream_calls
    assert len(runner.run_calls) == 1
    assert (
        "Required tool: load_guided_exercise_skill"
        in runner.stream_calls[0]["input_text"]
    )
    assert (
        "Required tool: load_guided_exercise_skill"
        not in runner.run_calls[0]["input_text"]
    )


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
    # The low-level text flow builds crisis response state only; outer runtime
    # persistence owns bounded safety-event capture.
    assert await context.crisis_log_backend.arecord_count() == 0


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


class _MidStreamFailRunner(FakeOpenAISDKRunner):
    """SDK runner whose stream emits one chunk and then raises a fallback-eligible
    error, to exercise the mid-stream-failure branch of run_turn_stream."""

    def run_streamed(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)

        class _FailingStream:
            final_output = ""

            async def stream_events(self) -> AsyncIterator[Any]:
                from types import SimpleNamespace

                yield SimpleNamespace(
                    type="raw_response_event",
                    data=SimpleNamespace(
                        type="response.output_text.delta", delta="partial "
                    ),
                )
                from openai import APIConnectionError

                raise APIConnectionError(request=cast(Any, None))

        return _FailingStream()


@pytest.mark.asyncio
async def test_streaming_sdk_failure_after_chunks_does_not_double_stream() -> None:
    # Regression for #165: if the SDK stream fails AFTER chunks have already been
    # emitted to the client, we must NOT fall back to a fresh full control-LLM
    # reply (that would duplicate/garble output, and there is no reset event in
    # the protocol). The turn re-raises instead — a clean error beats garbled
    # output, which matters most on the voice response path.
    workflow = _StatefulWorkflow()
    runtime = _runtime(workflow, _MidStreamFailRunner("unused"))
    # llm_client (NOT response_llm — that would divert to a different stream path)
    # is set so the control-LLM fallback would be ELIGIBLE if no chunks had
    # streamed. The fallback would stream this client's text; asserting it never
    # streams proves the `if chunks: raise` guard (not the predicate) blocks it.
    fallback_llm = _RecordingResponseLLM("FALLBACK FULL REPLY")
    context = WorkflowContext(
        llm_client=fallback_llm,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )

    emitted: list[Any] = []
    with pytest.raises(Exception):  # noqa: B017 - any SDK error re-raise is fine
        async for event in runtime.run_turn_stream(
            cast(Any, _initial_state()),
            config={"configurable": {"thread_id": "thread-1"}},
            context=context,
        ):
            emitted.append(event)

    # The one partial chunk was emitted; the fallback full reply was NOT.
    chunk_texts = [e.text for e in emitted if isinstance(e, TextRuntimeChunkEvent)]
    assert "partial " in chunk_texts
    assert "FALLBACK FULL REPLY" not in chunk_texts
    assert fallback_llm.stream_calls == 0  # fallback stream never invoked


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
    assert (
        TextRuntimeStatusEvent(stage="crisis_log", turn_finalized=False) not in events
    )
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
    assert await context.crisis_log_backend.arecord_count() == 0


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


async def _never_resolves() -> Any:
    # Simulates an in-flight prefetch that the turn's route never consumes.
    await asyncio.sleep(3600)


async def _assert_settles(task: asyncio.Task[Any]) -> None:
    # task.cancel() only requests cancellation; let the loop deliver it, bounded
    # so a regression (prefetch left pending) fails fast and cleanly instead of
    # hanging on the task's long sleep.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
    except asyncio.CancelledError:
        pass  # expected: the dispatch finally cancelled the orphaned prefetch
    except TimeoutError:  # pragma: no cover - only hit if the drain regresses
        pass
    assert task.done(), (
        "prefetch task left pending — dispatch boundary did not drain it"
    )


_USE_GROUNDED_ROUTE = object()


def _context_with_orphan_prefetch(
    *,
    llm_client: Any = _USE_GROUNDED_ROUTE,
) -> tuple[WorkflowContext, asyncio.Task[Any]]:
    # Attaches a never-resolving prefetch that the turn's route never consumes,
    # so it can only be settled by the dispatch-boundary finally. By default
    # drives the grounded_lookup route (never calls load_turn_memory); pass
    # llm_client=None to exercise the deterministic no-LLM smoke path. (owner_id
    # matches the turn's owner so a mismatch cancel cannot mask the behavior
    # under test.)
    task = asyncio.ensure_future(_never_resolves())
    prefetch = PrefetchedTurnMemory(
        task=task,
        owner_id="user-1",
        query="Can you look up the current rule?",
        is_first_turn=True,
    )
    resolved_client = (
        _RouteLLM(route="grounded_lookup")
        if llm_client is _USE_GROUNDED_ROUTE
        else llm_client
    )
    context = WorkflowContext(
        llm_client=resolved_client,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
        pre_fetched_memory=prefetch,
    )
    return context, task


def _grounded_runner() -> FakeOpenAISDKRunner:
    return FakeOpenAISDKRunner(
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )


@pytest.mark.asyncio
async def test_run_turn_drains_orphaned_prefetch_task() -> None:
    # Regression for #160/#161: the speculative memory prefetch is scheduled on
    # every turn but only consumed by routes that call load_turn_memory. On
    # grounded_lookup (and crisis_response) nothing consumes it, so the dispatch
    # boundary must settle (cancel/retrieve) it by turn end — never orphaned.
    workflow = _StatefulWorkflow()
    runtime = _runtime(workflow, _grounded_runner())
    context, task = _context_with_orphan_prefetch()

    await runtime.run_turn(
        cast(Any, _initial_state("Can you look up the current rule?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    # cancel() only requests cancellation; let the loop deliver it. Bounded so a
    # regression (task left pending) fails fast and cleanly instead of hanging on
    # the task's long sleep.
    await _assert_settles(task)


@pytest.mark.asyncio
async def test_run_turn_stream_drains_orphaned_prefetch_task() -> None:
    # Same invariant on the streaming path; the finally also covers early
    # consumer abandonment (aclose()).
    workflow = _StatefulWorkflow()
    runtime = _runtime(workflow, _grounded_runner())
    context, task = _context_with_orphan_prefetch()

    async for _event in runtime.run_turn_stream(
        cast(Any, _initial_state("Can you look up the current rule?")),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    ):
        pass

    await _assert_settles(task)


@pytest.mark.asyncio
async def test_run_turn_drains_prefetch_on_deterministic_no_llm_turn() -> None:
    # Codex P2 follow-up: the prefetch is scheduled independently of llm_client,
    # so a deterministic (no-LLM) smoke turn can still carry a live prefetch. The
    # drain must run on the no-LLM early-return path too, not only the real path.
    workflow = _StatefulWorkflow()
    runtime = _runtime(workflow, _grounded_runner())
    context, task = _context_with_orphan_prefetch(llm_client=None)

    await runtime.run_turn(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    )

    await _assert_settles(task)


@pytest.mark.asyncio
async def test_run_turn_stream_drains_prefetch_on_deterministic_no_llm_turn() -> None:
    # Same no-LLM smoke-path coverage for the streaming method.
    workflow = _StatefulWorkflow()
    runtime = _runtime(workflow, _grounded_runner())
    context, task = _context_with_orphan_prefetch(llm_client=None)

    async for _event in runtime.run_turn_stream(
        cast(Any, _initial_state()),
        config={"configurable": {"thread_id": "thread-1"}},
        context=context,
    ):
        pass

    await _assert_settles(task)
