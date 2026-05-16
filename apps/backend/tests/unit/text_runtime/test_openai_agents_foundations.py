"""Tests for dormant OpenAI Agents SDK text-runtime foundations."""

from __future__ import annotations

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
from agent.state import resolve_owner_id
from agent.text_runtime import resolve_text_agent_runtime
from agent.text_runtime.openai_agents import (
    CRISIS_AGENT_NAME,
    GUIDED_EXERCISE_AGENT_NAME,
    THERAPEUTIC_AGENT_NAME,
    MemoryReadToolResult,
    OpenAITextRunContext,
    build_openai_text_agent_roster,
    build_read_only_memory_tools,
    execute_read_only_memory_action,
    show_memory_status,
    show_saved_memory,
)


def _workflow_context(
    *,
    store: OpenCouchMemoryStore | None = None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
) -> WorkflowContext:
    return WorkflowContext(
        llm_client=None,
        memory_store=store or OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=memory_mode,
    )


def _run_context(
    *,
    store: OpenCouchMemoryStore | None = None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
    user_id: str | None = "user-1",
) -> OpenAITextRunContext:
    return OpenAITextRunContext(
        thread_id="thread-1",
        user_id=user_id,
        session_id="session-1",
        current_user_message="What do you remember about me?",
        workflow_context=_workflow_context(store=store, memory_mode=memory_mode),
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


async def _invoke_tool(tool: Any, context: OpenAITextRunContext) -> Any:
    return await tool.on_invoke_tool(
        ToolContext(
            context,
            tool_name=tool.name,
            tool_call_id="call-test",
            tool_arguments="{}",
        ),
        "{}",
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

    assert version("openai-agents") >= "0.17.2"
    assert Agent.__name__ == "Agent"


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


def test_local_context_adapts_to_memory_service_state_without_clients() -> None:
    """Local SDK context should not leak backend clients into model state."""

    context = _run_context()

    state = context.agent_state_for_memory_action({"type": "status"})

    assert resolve_owner_id(state) == "user-1"
    assert state["memory_control"]["action"] == {"type": "status"}
    assert "workflow_context" not in state
    assert "memory_store" not in state


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


@pytest.mark.asyncio
async def test_show_saved_memory_tool_respects_incognito_mode() -> None:
    """Read-only tools should still honor app-owned memory mode boundaries."""

    context = _run_context(memory_mode=MemoryMode.INCOGNITO)

    result = await _invoke_tool(show_saved_memory, context)

    assert isinstance(result, MemoryReadToolResult)
    assert "guest mode" in result.response_text
    assert result.memory_control == {"pending_action": None}
