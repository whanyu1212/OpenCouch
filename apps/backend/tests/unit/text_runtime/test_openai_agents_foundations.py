"""Tests for OpenAI Agents SDK text-runtime foundations."""

from __future__ import annotations

import json
from importlib.metadata import version
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from agents import Agent
from agents.tool_context import ToolContext

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.flows.guided_exercise import _build_guided_exercise_agent
from agent.flows.therapeutic import run_therapeutic_turn
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.types import TurnDispatchDecision
from agent.models import AgentInput
from agent.runtime import build_initial_state
from agent.runtime.workflow_context import WorkflowContext
from agent.runtime.services import TextRuntimeServices
from agent.runtime.state_ops import finalize_openai_turn
from agent.specialists.crisis import CRISIS_AGENT_NAME
from agent.specialists.guided_exercise import (
    GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
    GUIDED_EXERCISE_AGENT_NAME,
)
from agent.specialists.roster import build_openai_text_agent_roster
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.runtime.context import MemoryToolCallRecord, OpenAITextRunContext
from agent.tools import (
    CrisisResourceLookupToolResult,
    CrisisSupportTemplateToolResult,
    GroundedLookupToolResult,
    GuidedExerciseSkillDiscoveryToolResult,
    GuidedExerciseSkillToolResult,
    MemoryReadToolResult,
    MemoryToolResult,
    TherapeuticResponseSkillToolResult,
    answer_grounded_lookup,
    build_memory_tools,
    execute_crisis_resource_lookup_tool,
    execute_crisis_support_template_tool,
    execute_guided_exercise_discovery_tool,
    execute_guided_exercise_skill_tool,
    execute_grounded_lookup_tool,
    execute_read_only_memory_action,
    execute_therapeutic_response_skill_tool,
    list_guided_exercise_skills,
    load_guided_exercise_skill,
    get_crisis_support_template,
    load_therapeutic_response_skill,
    lookup_crisis_resources,
    memory_control_request_from_context,
    save_response_preference,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)
from tests.support.openai_text import (
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM,
)


def _workflow_context(
    *,
    store: OpenCouchMemoryStore | None = None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
    llm: Any | None = None,
) -> WorkflowContext:
    return WorkflowContext(
        llm_client=llm,
        memory_store=store or OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=memory_mode,
    )


def _run_context(
    *,
    store: OpenCouchMemoryStore | None = None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
    user_id: str | None = "user-1",
    llm: Any | None = None,
) -> OpenAITextRunContext:
    return OpenAITextRunContext(
        thread_id="thread-1",
        user_id=user_id,
        session_id="session-1",
        current_user_message="What do you remember about me?",
        workflow_context=_workflow_context(
            store=store,
            memory_mode=memory_mode,
            llm=llm,
        ),
    )


async def _seed_fact(store: OpenCouchMemoryStore, *, owner_id: str = "user-1") -> None:
    await store.aput(
        (owner_id, "semantic"),
        "fact-presentations",
        {
            "category": "trigger",
            "predicate": "WORRIES_ABOUT",
            "object": {"identifier": "presentations"},
            "evidence_quote": "Presentations make me anxious.",
        },
    )


async def _seeded_context() -> tuple[OpenAITextRunContext, OpenCouchMemoryStore, int]:
    store = OpenCouchMemoryStore()
    await _seed_fact(store)
    return (
        _run_context(store=store),
        store,
        await store.arecord_count(("user-1", "semantic")),
    )


async def _invoke_tool(
    tool: Any,
    context: OpenAITextRunContext,
    arguments: dict[str, Any] | None = None,
) -> Any:
    payload = json.dumps(arguments or {})
    return await tool.on_invoke_tool(
        ToolContext(
            context,
            tool_name=tool.name,
            tool_call_id="call-test",
            tool_arguments=payload,
        ),
        payload,
    )


async def _assert_semantic_count_unchanged(
    store: OpenCouchMemoryStore,
    before: int,
    action: Callable[[], Awaitable[MemoryReadToolResult]],
) -> MemoryReadToolResult:
    result = await action()

    assert isinstance(result, MemoryReadToolResult)
    assert await store.arecord_count(("user-1", "semantic")) == before
    return result


def test_openai_agents_dependency_imports() -> None:
    """The backend venv should expose the SDK package this slice depends on."""

    from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

    assert version("openai-agents") >= "0.17.2"
    assert Agent.__name__ == "Agent"
    assert SQLAlchemySession.__name__ == "SQLAlchemySession"


def test_agent_roster_builds_dormant_specialists() -> None:
    """The dormant roster should define specialists without wiring runtime use."""

    roster = build_openai_text_agent_roster(model="gpt-test")

    assert roster.triage_agent.output_type is TurnDispatchDecision
    assert roster.therapeutic_agent.name == THERAPEUTIC_AGENT_NAME
    assert roster.crisis_agent.name == CRISIS_AGENT_NAME
    assert roster.guided_exercise_agent.name == GUIDED_EXERCISE_AGENT_NAME
    assert roster.therapeutic_agent.model == "gpt-test"
    assert roster.crisis_agent.model == "gpt-test"
    assert roster.guided_exercise_agent.model == "gpt-test"
    assert [tool.name for tool in roster.therapeutic_agent.tools] == [
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
    assert [tool.name for tool in roster.crisis_agent.tools] == [
        "lookup_crisis_resources",
        "get_crisis_support_template",
    ]
    assert [tool.name for tool in roster.guided_exercise_agent.tools] == [
        "load_guided_exercise_skill",
    ]
    assert roster.therapeutic_agent.handoffs == []


def test_runtime_guided_exercise_agent_does_not_mutate_roster_agent() -> None:
    """Runtime prompt variants must not mutate the cached roster agent."""

    roster = build_openai_text_agent_roster(model="gpt-test")
    base_agent = roster.guided_exercise_agent
    original_instructions = base_agent.instructions

    first_agent = _build_guided_exercise_agent(
        base_agent,
        system_instruction="first turn instructions",
        runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
    )
    second_agent = _build_guided_exercise_agent(
        base_agent,
        system_instruction="second turn instructions",
        runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
    )

    assert first_agent is not base_agent
    assert second_agent is not base_agent
    assert first_agent is not second_agent
    assert base_agent.instructions == original_instructions
    assert "first turn instructions" in first_agent.instructions
    assert "second turn instructions" in second_agent.instructions
    assert "first turn instructions" not in second_agent.instructions


@pytest.mark.asyncio
async def test_therapeutic_flow_uses_explicit_runtime_services_boundary() -> None:
    """Therapeutic flows should depend on services, not runtime internals."""

    runner = FakeOpenAISDKRunner("services reply")
    roster = build_openai_text_agent_roster(model="gpt-test")
    state = build_initial_state(
        AgentInput(
            message="I feel tense today",
            user_id="user-1",
            session_id="thread-1",
        )
    )

    async def run_agent_with(
        state,
        *,
        agent,
        input_text,
        run_context,
        session=None,
    ):
        result = await runner.run(
            agent=agent,
            input_text=input_text,
            context=run_context,
            session=session,
        )
        return str(result.final_output), 12.0

    async def finalize_turn(state, **kwargs):
        kwargs.pop("config", None)
        return finalize_openai_turn(state, **kwargs)

    async def load_turn_memory(state, context):
        return state

    services = TextRuntimeServices(
        runner=runner,
        roster=roster,
        build_run_context=lambda state, config, context: OpenAITextRunContext(
            thread_id="thread-1",
            workflow_context=context,
            current_user_message=str(state.get("message") or ""),
            user_id="user-1",
            session_id="thread-1",
            agent_state=state,
        ),
        build_agent=lambda state: roster.therapeutic_agent,
        input_text_for_state=lambda state, include_recent_history=True: (
            f"services prompt: {state['message']}"
        ),
        crisis_input_text_for_state=lambda *args, **kwargs: "crisis prompt",
        run_openai_agent_with=run_agent_with,
        finalize_turn=finalize_turn,
        load_turn_memory=load_turn_memory,
    )

    result = await run_therapeutic_turn(
        services,
        state,
        config={"configurable": {"thread_id": "thread-1"}},
        context=_workflow_context(llm=ScriptedOpenAITextRouteLLM(route="therapeutic")),
    )

    assert result.response_text == "services reply"
    assert runner.run_calls[0]["agent"].name == THERAPEUTIC_AGENT_NAME
    assert runner.run_calls[0]["input_text"] == "services prompt: I feel tense today"


def test_operational_tool_metadata_is_explicit() -> None:
    """Operational tools should state side effects and retry safety."""

    tools = build_memory_tools()

    assert [tool.name for tool in tools] == [
        "show_saved_memory",
        "show_memory_status",
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
    ]
    for tool in tools:
        assert "Side effects:" in tool.description
        assert "Retry safety:" in tool.description
        assert tool.strict_json_schema is True
        assert tool.params_json_schema["additionalProperties"] is False


def test_local_context_builds_neutral_memory_request_without_runtime_state() -> None:
    """Local SDK context should hand tools neutral input, not runtime state."""

    context = _run_context()

    request = memory_control_request_from_context(context, {"type": "status"})

    assert request.owner_id == "user-1"
    assert request.action == {"type": "status"}
    assert not hasattr(context, "agent_state_for_memory_action")
    assert not hasattr(context, "agent_state_for_grounded_lookup")
    assert not hasattr(context, "agent_state_for_crisis_resources")
    assert context.memory_tool_calls == []


@pytest.mark.asyncio
async def test_show_saved_memory_action_is_read_only() -> None:
    """The OpenAI read tool should reuse existing memory behavior without writes."""

    context, store, before = await _seeded_context()
    result = await _assert_semantic_count_unchanged(
        store,
        before,
        lambda: execute_read_only_memory_action(context, {"type": "list"}),
    )

    assert "Here's what I currently have saved:" in result.response_text
    assert "Saved facts:" in result.response_text
    assert "Presentations make me anxious." in result.response_text
    assert result.memory_control == {"pending_action": None}
    assert context.memory_tool_calls == [
        MemoryToolCallRecord(
            tool_name="show_saved_memory",
            action_type="list",
            response_text=result.response_text,
            memory_control={"pending_action": None},
        )
    ]


@pytest.mark.asyncio
async def test_show_memory_status_tool_invocation_is_read_only() -> None:
    """The SDK function tool should invoke with local context and no mutation."""

    context, store, before = await _seeded_context()
    result = await _assert_semantic_count_unchanged(
        store,
        before,
        lambda: _invoke_tool(show_memory_status, context),
    )

    assert result.side_effect == "none"
    assert result.retry_safe is True
    assert "Saved facts: 1" in result.response_text
    assert "Proactive recall: off" in result.response_text
    assert [call.tool_name for call in context.memory_tool_calls] == [
        "show_memory_status"
    ]


@pytest.mark.asyncio
async def test_show_saved_memory_tool_respects_incognito_mode() -> None:
    """Read-only tools should still honor app-owned memory mode boundaries."""

    context = _run_context(memory_mode=MemoryMode.INCOGNITO)

    result = await _invoke_tool(show_saved_memory, context)

    assert isinstance(result, MemoryReadToolResult)
    assert "guest mode" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_set_proactive_memory_recall_tool_updates_profile() -> None:
    """Mutating memory tools should reuse the app-owned service implementation."""

    context = _run_context()

    result = await _invoke_tool(
        set_proactive_memory_recall,
        context,
        {"enabled": True},
    )

    assert isinstance(result, MemoryToolResult)
    assert "I turned proactive recall on." in result.response_text
    assert result.memory_control == {"pending_action": None}
    assert result.procedural_profile == {"proactive_recall_enabled": True}
    assert result.side_effect == "procedural_profile_update"
    assert result.retry_safe is True
    assert context.memory_tool_calls[-1].tool_name == "set_proactive_memory_recall"


@pytest.mark.asyncio
async def test_save_response_preference_tool_writes_procedural_memory() -> None:
    """Preference saves should stay service-owned but be invocable as SDK tools."""

    store = OpenCouchMemoryStore()
    context = _run_context(
        store=store,
        llm=ScriptedOpenAITextRouteLLM(route="memory_control"),
    )

    result = await _invoke_tool(
        save_response_preference,
        context,
        {"preference_text": "direct answers when I am spiraling"},
    )

    assert isinstance(result, MemoryToolResult)
    assert result.response_text.startswith("Saved:")
    assert result.side_effect == "procedural_profile_update"
    assert result.retry_safe is False
    assert await store.arecord_count(("user-1", "procedural")) == 1


@pytest.mark.asyncio
async def test_grounded_lookup_tool_records_lookup_result() -> None:
    """Grounded lookup tools should reuse the existing search service."""

    context = _run_context(llm=ScriptedOpenAITextRouteLLM(route="grounded_lookup"))

    result = await execute_grounded_lookup_tool(
        context,
        query="grounded query",
    )

    assert isinstance(result, GroundedLookupToolResult)
    assert result.response_text == "Official answer.\n\nSources:\n- Official source"
    assert result.grounded_lookup == {
        "query": "grounded query",
        "status": "answered",
    }
    assert context.grounded_tool_calls[-1].tool_name == "answer_grounded_lookup"


@pytest.mark.asyncio
async def test_grounded_lookup_function_tool_invokes_with_context() -> None:
    """The SDK function tool wrapper should pass arguments into local context."""

    context = _run_context(llm=ScriptedOpenAITextRouteLLM(route="grounded_lookup"))

    result = await _invoke_tool(
        answer_grounded_lookup,
        context,
        {"query": "grounded query"},
    )

    assert isinstance(result, GroundedLookupToolResult)
    assert result.status == "answered"
    assert result.side_effect == "none"
    assert result.retry_safe is True


@pytest.mark.asyncio
async def test_therapeutic_response_skill_tool_records_skill_context() -> None:
    """TherapeuticResponseAgent should own response-style prompt skills."""

    context = _run_context()
    context.agent_state = {
        "message": "I feel tense today",
        "history": [],
        "transcript": [],
        "working_memory": [],
        "session_memory": {},
        "procedural_profile": {},
        "turn_lifecycle": {"active_flow": "none", "action": "none"},
        "memory_reference": {"mode": "none"},
        "session_progress": {"turn_count": 1},
        "response_guidance": "",
    }

    result = await execute_therapeutic_response_skill_tool(
        context,
        response_style="supportive",
        therapeutic_approach="none",
    )

    assert isinstance(result, TherapeuticResponseSkillToolResult)
    assert result.response_style == "supportive"
    assert result.therapeutic_approach == "none"
    assert result.side_effect == "none"
    assert result.retry_safe is True
    assert result.skill_context.startswith("Therapeutic response skill:")
    assert "SUPPORTIVE response style" in result.skill_context
    assert context.therapeutic_response_skill_tool_calls[-1].tool_name == (
        "load_therapeutic_response_skill"
    )


@pytest.mark.asyncio
async def test_therapeutic_response_skill_function_tool_invokes_with_context() -> None:
    """The SDK response skill wrapper should pass selected style args."""

    context = _run_context()
    context.agent_state = {
        "message": "I feel tense today",
        "history": [],
        "transcript": [],
        "working_memory": [],
        "session_memory": {},
        "procedural_profile": {},
        "turn_lifecycle": {"active_flow": "none", "action": "none"},
        "memory_reference": {"mode": "none"},
        "session_progress": {"turn_count": 1},
        "response_guidance": "",
    }

    result = await _invoke_tool(
        load_therapeutic_response_skill,
        context,
        {
            "response_style": "reflective",
            "therapeutic_approach": "cbt",
        },
    )

    assert isinstance(result, TherapeuticResponseSkillToolResult)
    assert result.response_style == "reflective"
    assert result.therapeutic_approach == "cbt"
    assert "REFLECTIVE response style" in result.skill_context


@pytest.mark.asyncio
async def test_crisis_resource_tool_records_lookup_result() -> None:
    """CrisisResponseAgent should own crisis-resource lookup as an SDK tool."""

    context = _run_context(
        llm=ScriptedOpenAITextRouteLLM(route="therapeutic", crisis_level=3)
    )

    result = await execute_crisis_resource_lookup_tool(context)

    assert isinstance(result, CrisisResourceLookupToolResult)
    assert result.resource_lookup_status == "found"
    assert result.found_resources[0]["phone"] == "1767"
    assert "Verified local crisis resources for Singapore" in result.response_text
    assert context.crisis_resource_tool_calls[-1].tool_name == (
        "lookup_crisis_resources"
    )


@pytest.mark.asyncio
async def test_crisis_resource_function_tool_invokes_with_context() -> None:
    """The SDK crisis-resource tool wrapper should use local context."""

    context = _run_context(
        llm=ScriptedOpenAITextRouteLLM(route="therapeutic", crisis_level=3)
    )

    result = await _invoke_tool(lookup_crisis_resources, context)

    assert isinstance(result, CrisisResourceLookupToolResult)
    assert result.resource_lookup_status == "found"
    assert result.side_effect == "none"
    assert result.retry_safe is True


@pytest.mark.asyncio
async def test_crisis_support_template_tool_returns_imminent_scaffold() -> None:
    """The crisis template tool should provide deterministic safety scaffolding."""

    result = await execute_crisis_support_template_tool(
        risk_level="imminent",
        inferred_location="Singapore",
        found_resources=[
            {
                "name": "Samaritans of Singapore",
                "phone": "1767",
                "url": "https://www.sos.org.sg",
                "region": "Singapore",
            }
        ],
        resource_lookup_status="found",
    )

    assert isinstance(result, CrisisSupportTemplateToolResult)
    assert result.risk_level == "imminent"
    assert "emergency services" in result.immediate_safety_step
    assert "1767" in result.resource_guidance
    assert "Samaritans of Singapore" in result.response_text
    assert any("Do not invent phone numbers" in item for item in result.avoid)
    assert result.side_effect == "none"
    assert result.retry_safe is True


@pytest.mark.asyncio
async def test_crisis_support_template_tool_uses_fallback_without_resources() -> None:
    """The template should stay useful when verified local resources are absent."""

    result = await execute_crisis_support_template_tool(
        risk_level="moderate",
        resource_lookup_status="no_location",
    )

    assert result.risk_level == "moderate"
    assert "local emergency services" in result.resource_guidance
    assert "Do not invent phone numbers" in result.resource_guidance
    assert "Are you somewhere safe enough" in result.one_question


@pytest.mark.asyncio
async def test_crisis_support_template_function_tool_invokes_with_arguments() -> None:
    """The SDK template wrapper should pass explicit scaffold arguments."""

    context = _run_context()

    result = await _invoke_tool(
        get_crisis_support_template,
        context,
        {
            "risk_level": "level 3",
            "inferred_location": "Singapore",
            "resource_lookup_status": "found",
            "resource_name": "Samaritans of Singapore",
            "resource_phone": "1767",
            "resource_url": "https://www.sos.org.sg",
            "resource_region": "Singapore",
        },
    )

    assert isinstance(result, CrisisSupportTemplateToolResult)
    assert result.risk_level == "imminent"
    assert "1767" in result.resource_guidance


@pytest.mark.asyncio
async def test_guided_exercise_discovery_tool_returns_metadata_only() -> None:
    """TherapeuticAgent should discover guided exercise metadata without scripts."""

    context = _run_context()
    context.agent_state = {"therapeutic_approach": "none"}

    result = await execute_guided_exercise_discovery_tool(context)

    assert isinstance(result, GuidedExerciseSkillDiscoveryToolResult)
    assert result.side_effect == "none"
    assert result.retry_safe is True
    assert result.skills
    assert all(skill.skill_id for skill in result.skills)
    assert all(skill.description for skill in result.skills)
    assert all(
        "canonical_instruction" not in skill.model_dump_json()
        for skill in result.skills
    )


@pytest.mark.asyncio
async def test_guided_exercise_discovery_function_tool_invokes_with_context() -> None:
    """The SDK discovery wrapper should pass filtering args."""

    context = _run_context()

    result = await _invoke_tool(
        list_guided_exercise_skills,
        context,
        {
            "therapeutic_approach": "none",
            "channel": "text",
        },
    )

    assert isinstance(result, GuidedExerciseSkillDiscoveryToolResult)
    assert result.channel == "text"
    assert result.skills


@pytest.mark.asyncio
async def test_guided_exercise_skill_tool_records_skill_context() -> None:
    """GuidedExerciseAgent should own loading exercise skill context."""

    context = _run_context()

    result = await execute_guided_exercise_skill_tool(
        context,
        exercise_type="grounding_5_4_3_2_1",
        current_step_index=0,
        runtime_action="start",
    )

    assert isinstance(result, GuidedExerciseSkillToolResult)
    assert result.exercise_type == "grounding_5_4_3_2_1"
    assert result.current_step_index == 0
    assert result.runtime_action == "start"
    assert "Exercise skill:" in result.skill_context
    assert "grounding_5_4_3_2_1" in result.skill_context
    assert context.guided_exercise_skill_tool_calls[-1].tool_name == (
        "load_guided_exercise_skill"
    )


@pytest.mark.asyncio
async def test_guided_exercise_skill_function_tool_invokes_with_context() -> None:
    """The SDK exercise skill tool wrapper should pass runtime-selected args."""

    context = _run_context()

    result = await _invoke_tool(
        load_guided_exercise_skill,
        context,
        {
            "exercise_type": "grounding_5_4_3_2_1",
            "current_step_index": 0,
            "runtime_action": "start",
        },
    )

    assert isinstance(result, GuidedExerciseSkillToolResult)
    assert result.side_effect == "none"
    assert result.retry_safe is True
    assert result.skill_context.startswith("Exercise skill:")
