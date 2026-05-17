"""Tests for dormant OpenAI Agents SDK text-runtime foundations."""

from __future__ import annotations

import json
from importlib.metadata import version
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from agents import Agent
from agents.tool_context import ToolContext

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.text_runtime import resolve_text_agent_runtime
from agent.text_runtime.openai_agents import (
    CRISIS_AGENT_NAME,
    CrisisResourceLookupToolResult,
    GUIDED_EXERCISE_AGENT_NAME,
    GuidedExerciseSkillToolResult,
    THERAPEUTIC_AGENT_NAME,
    GroundedLookupToolResult,
    MemoryReadToolResult,
    MemoryToolResult,
    MemoryToolCallRecord,
    OpenAITextRunContext,
    answer_grounded_lookup,
    build_memory_tools,
    build_openai_text_agent_roster,
    build_read_only_memory_tools,
    execute_crisis_resource_lookup_tool,
    execute_guided_exercise_skill_tool,
    execute_grounded_lookup_tool,
    save_response_preference,
    set_proactive_memory_recall,
    execute_read_only_memory_action,
    load_guided_exercise_skill,
    lookup_crisis_resources,
    show_memory_status,
    show_saved_memory,
)
from agent.text_runtime.openai_agents.memory_tools import (
    memory_control_request_from_context,
)
from tests.support.openai_text import ScriptedOpenAITextRouteLLM


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


def test_openai_runtime_selector_is_enabled_for_hybrid_slice() -> None:
    """The OpenAI selector is now enabled behind explicit runtime config."""

    assert resolve_text_agent_runtime("openai") == "openai"


def test_agent_roster_builds_dormant_specialists() -> None:
    """The dormant roster should define specialists without wiring runtime use."""

    roster = build_openai_text_agent_roster(model="gpt-test")

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
        "answer_grounded_lookup",
    ]
    assert [tool.name for tool in roster.crisis_agent.tools] == [
        "lookup_crisis_resources"
    ]
    assert [tool.name for tool in roster.guided_exercise_agent.tools] == [
        "load_guided_exercise_skill"
    ]
    assert roster.therapeutic_agent.handoffs == []


def test_read_only_memory_tool_metadata_is_explicit() -> None:
    """Tool contracts should state scope, side effects, and retry safety."""

    tools = build_read_only_memory_tools()

    assert [tool.name for tool in tools] == [
        "show_saved_memory",
        "show_memory_status",
    ]
    for tool in tools:
        assert "Side effects: none" in tool.description
        assert "Retry safety: safe" in tool.description
        assert tool.strict_json_schema is True
        assert tool.params_json_schema["additionalProperties"] is False
        assert tool.params_json_schema["required"] == []


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


def test_local_context_builds_neutral_memory_request_without_graph_state() -> None:
    """Local SDK context should hand tools neutral input, not graph state."""

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
