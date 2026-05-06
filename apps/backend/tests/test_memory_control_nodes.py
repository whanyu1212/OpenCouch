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
from agent.gates.memory_control.router import (
    MemoryControlAction,
    detect_memory_control_action,
    resolve_memory_control_action,
)
from agent.models import AgentInput
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.memory_control_gate import run_memory_control_gate_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from services.llm.base import BaseLLMClient, StructuredResponseT


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


class _FakeMemoryControlLLM(BaseLLMClient):
    """Fake structured client for memory-control routing tests."""

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
        raise AssertionError("Text generation is not used by memory-control routing.")

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
        command: LangGraph command returned by a routing gate.

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


def test_detect_memory_control_action_returns_typed_action() -> None:
    action = detect_memory_control_action("What do you remember about me?")

    assert action == MemoryControlAction({"type": "list"})
    assert action.to_state_action() == {"type": "list"}


@pytest.mark.asyncio
async def test_resolve_memory_control_action_routes_deterministic_request() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "none",
            "reasoning": "should not be called",
            "confidence": "high",
        }
    )

    route = await resolve_memory_control_action(
        _state("What do you remember about me?"),
        llm_client=llm,
    )

    assert route.action == MemoryControlAction({"type": "list"})
    assert route.classifier_path == "deterministic"
    assert route.llm_failure_occurred is False
    assert llm.structured_calls == []


@pytest.mark.asyncio
async def test_resolve_memory_control_action_skips_non_control_message() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "list",
            "reasoning": "should not be called",
            "confidence": "high",
        }
    )

    route = await resolve_memory_control_action(
        _state("I keep remembering the argument."),
        llm_client=llm,
    )

    assert route.action is None
    assert route.classifier_path == "not_attempted"
    assert route.llm_failure_occurred is False
    assert llm.structured_calls == []


@pytest.mark.asyncio
async def test_resolve_memory_control_action_uses_llm_for_preference() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "save_preference",
            "rule_text": "shorter replies when I am panicking",
            "reasoning": "User asks to keep a response preference in mind.",
            "confidence": "high",
        }
    )

    route = await resolve_memory_control_action(
        _state("Could you keep in mind that I prefer shorter replies?"),
        llm_client=llm,
    )

    assert route.action == MemoryControlAction(
        {
            "type": "save_preference",
            "rule_text": "You prefer shorter replies when I am panicking.",
        }
    )
    assert route.classifier_path == "llm_primary"
    assert route.llm_failure_occurred is False
    assert len(llm.structured_calls) == 1


@pytest.mark.asyncio
async def test_resolve_memory_control_action_rejects_low_confidence_decision() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "forget_by_query",
            "query": "the argument",
            "reasoning": "Ambiguous user wording.",
            "confidence": "low",
        }
    )

    route = await resolve_memory_control_action(
        _state("Can you forget what I said earlier?"),
        llm_client=llm,
    )

    assert route.action is None
    assert route.classifier_path == "llm_primary"
    assert route.llm_failure_occurred is False
    assert len(llm.structured_calls) == 1


@pytest.mark.asyncio
async def test_resolve_memory_control_action_rejects_vague_delete_target() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "forget_by_query",
            "query": "that",
            "reasoning": "Target is too vague.",
            "confidence": "high",
        }
    )

    route = await resolve_memory_control_action(
        _state("Please forget that."),
        llm_client=llm,
    )

    assert route.action is None
    assert route.classifier_path == "llm_primary"
    assert route.llm_failure_occurred is False
    assert len(llm.structured_calls) == 1


@pytest.mark.asyncio
async def test_resolve_memory_control_action_falls_back_without_llm() -> None:
    route = await resolve_memory_control_action(
        _state("Can you keep in mind that I prefer shorter replies?"),
        llm_client=None,
    )

    assert route.action is None
    assert route.classifier_path == "deterministic"
    assert route.llm_failure_occurred is False


@pytest.mark.asyncio
async def test_resolve_memory_control_action_marks_llm_failure() -> None:
    llm = _FakeMemoryControlLLM(RuntimeError("classifier unavailable"))

    route = await resolve_memory_control_action(
        _state("Can you keep in mind that I prefer shorter replies?"),
        llm_client=llm,
    )

    assert route.action is None
    assert route.classifier_path == "deterministic"
    assert route.llm_failure_occurred is True
    assert len(llm.structured_calls) == 1


@pytest.mark.asyncio
async def test_memory_control_gate_routes_only_explicit_memory_commands() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "list",
            "reasoning": "should not be called",
            "confidence": "high",
        }
    )
    normal = await run_memory_control_gate_node(
        _state("I keep remembering the argument."),
        cast(Any, _Runtime(llm_client=llm)),
    )
    explicit = await run_memory_control_gate_node(
        _state("What do you remember about me?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert normal.goto == "grounded_lookup_gate_node"
    assert explicit.goto == "memory_control_node"
    explicit_update = _command_update(explicit)
    assert explicit_update["route"] == "memory_control"
    assert explicit_update["memory_control"]["action"] == {"type": "list"}
    assert llm.structured_calls == []


@pytest.mark.asyncio
async def test_memory_control_gate_routes_pending_confirmation() -> None:
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

    command = await run_memory_control_gate_node(state, cast(Any, _Runtime()))

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {"type": "confirm_pending"}


@pytest.mark.asyncio
async def test_memory_control_gate_routes_unless_asked_recall_request() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "set_recall",
            "enabled": False,
            "reasoning": "User wants the assistant to stop bringing past material up.",
            "confidence": "high",
        }
    )

    command = await run_memory_control_gate_node(
        _state("Can you stop bringing that up unless I ask?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {
        "type": "set_recall",
        "enabled": False,
    }
    assert update["diagnostics"]["memory_control_classifier_path"] == "deterministic"
    assert llm.structured_calls == []


@pytest.mark.asyncio
async def test_memory_control_gate_routes_do_not_remember_topic() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "none",
            "reasoning": "should not be called",
            "confidence": "high",
        }
    )

    command = await run_memory_control_gate_node(
        _state("Please don't remember the thing I said about my ex."),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {
        "type": "forget_by_query",
        "query": "the thing I said about my ex",
    }
    assert update["diagnostics"]["memory_control_classifier_path"] == "deterministic"
    assert llm.structured_calls == []


@pytest.mark.asyncio
async def test_memory_control_gate_llm_routes_ambiguous_preference() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "save_preference",
            "rule_text": "shorter replies when I am panicking",
            "reasoning": "User asks to keep a response preference in mind.",
            "confidence": "high",
        }
    )

    command = await run_memory_control_gate_node(
        _state("Could you keep in mind that I prefer shorter replies?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "memory_control_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {
        "type": "save_preference",
        "rule_text": "You prefer shorter replies when I am panicking.",
    }


@pytest.mark.asyncio
async def test_memory_control_gate_llm_rejects_low_confidence_decision() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "forget_by_query",
            "query": "the argument",
            "reasoning": "Ambiguous user wording.",
            "confidence": "low",
        }
    )

    command = await run_memory_control_gate_node(
        _state("Can you forget what I said earlier?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "grounded_lookup_gate_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {}
    assert update["diagnostics"]["memory_control_classifier_path"] == "llm_primary"


@pytest.mark.asyncio
async def test_memory_control_gate_llm_rejects_vague_delete_target() -> None:
    llm = _FakeMemoryControlLLM(
        {
            "action_type": "forget_by_query",
            "query": "that",
            "reasoning": "Target is too vague.",
            "confidence": "high",
        }
    )

    command = await run_memory_control_gate_node(
        _state("Please forget that."),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "grounded_lookup_gate_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {}


@pytest.mark.asyncio
async def test_memory_control_gate_llm_failure_falls_back_to_normal_route() -> None:
    llm = _FakeMemoryControlLLM(RuntimeError("classifier unavailable"))

    command = await run_memory_control_gate_node(
        _state("Can you keep in mind that I prefer shorter replies?"),
        cast(Any, _Runtime(llm_client=llm)),
    )

    assert command.goto == "grounded_lookup_gate_node"
    update = _command_update(command)
    assert update["memory_control"]["action"] == {}
    assert update["diagnostics"]["memory_control_classifier_path"] == "deterministic"
    assert update["diagnostics"]["memory_control_llm_failure_occurred"] is True


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
    state = _state("Remember that I prefer very short replies when I'm panicking.")
    state["memory_control"]["action"] = {
        "type": "save_preference",
        "rule_text": "You prefer very short replies when I'm panicking.",
    }

    delta = await run_memory_control_node(state, cast(Any, _Runtime(store=store)))
    profile = await aget_procedural_profile(store, user_id="user-1")

    assert len(profile.rules) == 1
    assert "very short replies" in profile.rules[0].rule
    assert "Saved:" in delta["response_text"]
