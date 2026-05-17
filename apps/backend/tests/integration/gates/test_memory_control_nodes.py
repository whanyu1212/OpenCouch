"""Tests for conversational memory-control routing and actions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural_profile import (
    aadd_procedural_rule,
    aget_procedural_profile,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.turn_dispatch import run_turn_dispatch_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT


class _Runtime:
    """Minimal runtime wrapper exposing ``runtime.context``."""

    def __init__(
        self,
        *,
        store: OpenCouchMemoryStore | None = None,
        llm_client: BaseLLMClient | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=store or OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode,
        )


class _FakeTurnDispatchLLM(BaseLLMClient):
    """Fake structured client for turn-dispatch tests."""

    def __init__(self, decision: dict[str, Any] | Exception) -> None:
        self.decision = decision
        self.structured_calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by turn dispatch.")

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        self.structured_calls.append(
            {"prompt": prompt, "system_instruction": system_instruction}
        )
        if isinstance(self.decision, Exception):
            raise self.decision
        return response_schema(**self.decision)


def _state(message: str, *, user_id: str = "user-1") -> AgentState:
    state = build_initial_state(
        AgentInput(message=message, user_id=user_id, session_id="thread-1"),
        include_input_history=True,
    )
    return cast(AgentState, dict(state))


def _command_update(command: Any) -> dict[str, Any]:
    """Return a non-optional command update for assertions.

    Args:
        command: runtime command returned by a routing gate.

    Returns:
        Command update cast to a concrete dictionary.
    """

    return cast(dict[str, Any], command.update)


async def _seed_memory(
    store: OpenCouchMemoryStore, *, owner_id: str = "user-1"
) -> None:
    now = iso_now()
    fact = SemanticFact(
        id="fact-presentations",
        category="trigger",
        subject=EntityRef(type="User", identifier=owner_id),
        predicate="WORRIES_ABOUT",
        object=EntityRef(type="Event", identifier="presentations"),
        evidence_quote="Presentations make me anxious.",
        confidence="high",
        source_session_id="thread-1",
        source_turn_index=0,
        created_at=now,
        last_referenced_at=now,
    )
    await store.aput((owner_id, "semantic"), fact.id, fact.model_dump(mode="json"))
    await aadd_procedural_rule(
        store,
        user_id=owner_id,
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep replies short."],
        ),
    )


@pytest.mark.asyncio
async def test_turn_dispatch_routes_therapeutic_turns_to_memory_load() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "therapeutic",
            "reasoning": "The user is asking for ordinary support.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    command = await run_turn_dispatch_node(
        _state("I keep remembering the argument."),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "load_memory_node"
    update = _command_update(command)
    assert update["route"] == "therapeutic"
    assert update["memory_control"]["action"] == {}
    assert update["grounded_lookup"] == {"query": "", "status": "not_attempted"}
    assert update["diagnostics"]["turn_dispatch_classifier_path"] == "llm_primary"
    assert len(llm.structured_calls) == 1


@pytest.mark.asyncio
async def test_turn_dispatch_routes_memory_control_action() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "memory_control",
            "memory_action_type": "list",
            "reasoning": "The user asked to inspect saved assistant memory.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    command = await run_turn_dispatch_node(
        _state("What do you remember about me?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["route"] == "memory_control"
    assert update["memory_control"]["action"] == {"type": "list"}
    assert update["grounded_lookup"] == {"query": "", "status": "not_attempted"}
    trace = update["diagnostics"]["routing_trace"]
    assert trace[-1]["stage"] == "turn_dispatch"
    assert trace[-1]["decision"] == "list"


@pytest.mark.asyncio
async def test_turn_dispatch_routes_pending_confirmation() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "memory_control",
            "memory_action_type": "confirm_pending",
            "reasoning": "The user confirmed the pending deletion.",
            "confidence": "high",
            "active_flow_action": "continue",
        }
    )
    state = _state("yes, delete it")
    state["memory_control"] = {
        "pending_action": {
            "type": "delete",
            "target": {
                "kind": "fact",
                "namespace": ["user-1", "semantic"],
                "key": "fact-1",
                "rule_id": None,
                "preview": "trigger: presentations",
            },
        }
    }

    command = await run_turn_dispatch_node(
        state,
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {"type": "confirm_pending"}


@pytest.mark.asyncio
async def test_turn_dispatch_clears_pending_when_user_moves_on() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "therapeutic",
            "reasoning": "The user moved on from the pending deletion.",
            "confidence": "high",
            "active_flow_action": "clear",
        }
    )
    state = _state("Actually, can we talk about work stress?")
    state["memory_control"] = {
        "pending_action": {
            "type": "delete",
            "target": {
                "kind": "fact",
                "namespace": ["user-1", "semantic"],
                "key": "fact-1",
                "rule_id": None,
                "preview": "trigger: presentations",
            },
        }
    }

    command = await run_turn_dispatch_node(
        state,
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "load_memory_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {}
    assert update["memory_control"]["pending_action"] is None


@pytest.mark.asyncio
async def test_turn_dispatch_preserves_save_preference_text() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "memory_control",
            "memory_action_type": "save_preference",
            "preference_text": "shorter replies when I am panicking",
            "reasoning": "The user asked to save a response preference.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    command = await run_turn_dispatch_node(
        _state("Could you keep in mind that I prefer shorter replies?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    update = _command_update(command)
    assert update["memory_control"]["action"] == {
        "type": "save_preference",
        "preference_text": "shorter replies when I am panicking",
    }


@pytest.mark.asyncio
async def test_turn_dispatch_requires_llm_client() -> None:
    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await run_turn_dispatch_node(
            _state("What do you remember about me?"),
            cast(Any, _Runtime(llm_client=None)),
        )


@pytest.mark.asyncio
async def test_turn_dispatch_rejects_incomplete_memory_payload() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "memory_control",
            "memory_action_type": "forget_by_query",
            "query": "that",
            "reasoning": "The model selected a vague deletion target.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    with pytest.raises(ValueError, match="vague target"):
        await run_turn_dispatch_node(
            _state("Please forget that."),
            cast(Any, _Runtime(llm_client=llm)),
        )


@pytest.mark.asyncio
async def test_turn_dispatch_propagates_llm_failure() -> None:
    llm = _FakeTurnDispatchLLM(RuntimeError("classifier unavailable"))

    with pytest.raises(RuntimeError, match="classifier unavailable"):
        await run_turn_dispatch_node(
            _state("Can you keep in mind that I prefer shorter replies?"),
            cast(Any, _Runtime(llm_client=llm)),
        )


@pytest.mark.asyncio
async def test_memory_control_node_lists_saved_memory() -> None:
    store = OpenCouchMemoryStore()
    await _seed_memory(store)
    state = _state("What do you remember about me?")
    state["memory_control"]["action"] = {"type": "list"}

    delta = await run_memory_control_node(state, cast(Any, _Runtime(store=store)))

    assert delta["response_style"] == "memory_control"
    assert "Presentations make me anxious" in delta["response_text"]
    assert "You prefer shorter responses" in delta["response_text"]
    assert delta["memory_control"]["pending_action"] is None


@pytest.mark.asyncio
async def test_memory_control_node_updates_proactive_recall() -> None:
    store = OpenCouchMemoryStore()
    state = _state("Don't bring up past sessions unless I ask.")
    state["memory_control"]["action"] = {"type": "set_recall", "enabled": False}

    delta = await run_memory_control_node(state, cast(Any, _Runtime(store=store)))
    profile = await aget_procedural_profile(store, user_id="user-1")

    assert profile.proactive_recall_enabled is False
    assert delta["procedural_profile"]["proactive_recall_enabled"] is False
    assert "proactive recall off" in delta["response_text"].lower()


@pytest.mark.asyncio
async def test_memory_control_delete_by_query_requires_confirmation_then_deletes() -> (
    None
):
    store = OpenCouchMemoryStore()
    await _seed_memory(store)

    first_state = _state("Forget what you remember about presentations.")
    first_state["memory_control"]["action"] = {
        "type": "forget_by_query",
        "query": "presentations",
    }
    first_delta = await run_memory_control_node(
        first_state,
        cast(Any, _Runtime(store=store)),
    )

    assert "Do you want me to delete it" in first_delta["response_text"]
    assert first_delta["memory_control"]["pending_action"]["type"] == "delete"
    owner_id = cast(str, first_state["user_id"])
    assert await store.aget((owner_id, "semantic"), "fact-presentations")

    confirm_state = _state("yes, delete it")
    confirm_state["memory_control"] = first_delta["memory_control"]
    confirm_state["memory_control"]["action"] = {"type": "confirm_pending"}
    confirm_delta = await run_memory_control_node(
        confirm_state,
        cast(Any, _Runtime(store=store)),
    )

    assert "Deleted that saved fact" in confirm_delta["response_text"]
    assert confirm_delta["memory_control"]["pending_action"] is None
    confirm_owner_id = cast(str, confirm_state["user_id"])
    assert (
        await store.aget((confirm_owner_id, "semantic"), "fact-presentations") is None
    )


@pytest.mark.asyncio
async def test_memory_control_node_saves_explicit_preference_rule() -> None:
    store = OpenCouchMemoryStore()
    llm = _FakeTurnDispatchLLM(
        {
            "rule_text": "You prefer very short replies when you are panicking.",
            "reasoning": "The user explicitly stated a response preference.",
            "confidence": "high",
        }
    )
    state = _state("Remember that I prefer very short replies when I'm panicking.")
    state["memory_control"]["action"] = {
        "type": "save_preference",
        "preference_text": "very short replies when I'm panicking",
    }

    delta = await run_memory_control_node(
        state,
        cast(Any, _Runtime(store=store, llm_client=llm)),
    )
    profile = await aget_procedural_profile(store, user_id="user-1")

    assert len(profile.rules) == 1
    assert "very short replies" in profile.rules[0].rule
    assert "Saved:" in delta["response_text"]
