"""Tests for conversational memory-control routing and actions."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural import (
    aadd_procedural_rule,
    aget_procedural_profile,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.memory_control_gate import run_memory_control_gate_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


class _Runtime:
    """Minimal runtime wrapper exposing ``runtime.context``."""

    def __init__(
        self,
        *,
        store: OpenCouchMemoryStore | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=store or OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode,
        )


def _state(message: str, *, user_id: str = "user-1") -> AgentState:
    state = build_initial_state(
        AgentInput(message=message, user_id=user_id, session_id="thread-1"),
        include_input_history=True,
    )
    return cast(AgentState, dict(state))


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
async def test_memory_control_gate_routes_only_explicit_memory_commands() -> None:
    normal = await run_memory_control_gate_node(
        _state("I keep remembering the argument."),
        cast(Any, _Runtime()),
    )
    explicit = await run_memory_control_gate_node(
        _state("What do you remember about me?"),
        cast(Any, _Runtime()),
    )

    assert normal.goto == "grounded_lookup_gate_node"
    assert explicit.goto == "memory_control_node"
    assert explicit.update["route"] == "memory_control"
    assert explicit.update["memory_control_action"] == {"type": "list"}


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
    assert command.update["memory_control_action"] == {"type": "confirm_pending"}


@pytest.mark.asyncio
async def test_memory_control_node_lists_saved_memory() -> None:
    store = OpenCouchMemoryStore()
    await _seed_memory(store)
    state = _state("What do you remember about me?")
    state["memory_control_action"] = {"type": "list"}

    delta = await run_memory_control_node(state, cast(Any, _Runtime(store=store)))

    assert delta["response_style"] == "memory_control"
    assert "Presentations make me anxious" in delta["response_text"]
    assert "You prefer shorter responses" in delta["response_text"]
    assert delta["memory_control"]["pending_action"] is None


@pytest.mark.asyncio
async def test_memory_control_node_updates_proactive_recall() -> None:
    store = OpenCouchMemoryStore()
    state = _state("Don't bring up past sessions unless I ask.")
    state["memory_control_action"] = {"type": "set_recall", "enabled": False}

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
    first_state["memory_control_action"] = {
        "type": "forget_by_query",
        "query": "presentations",
    }
    first_delta = await run_memory_control_node(
        first_state,
        cast(Any, _Runtime(store=store)),
    )

    assert "Do you want me to delete it" in first_delta["response_text"]
    assert first_delta["memory_control"]["pending_action"]["type"] == "delete"
    assert await store.aget((first_state["user_id"], "semantic"), "fact-presentations")

    confirm_state = _state("yes, delete it")
    confirm_state["memory_control"] = first_delta["memory_control"]
    confirm_state["memory_control_action"] = {"type": "confirm_pending"}
    confirm_delta = await run_memory_control_node(
        confirm_state,
        cast(Any, _Runtime(store=store)),
    )

    assert "Deleted that saved fact" in confirm_delta["response_text"]
    assert confirm_delta["memory_control"]["pending_action"] is None
    assert (
        await store.aget((confirm_state["user_id"], "semantic"), "fact-presentations")
        is None
    )


@pytest.mark.asyncio
async def test_memory_control_node_saves_explicit_preference_rule() -> None:
    store = OpenCouchMemoryStore()
    state = _state("Remember that I prefer very short replies when I'm panicking.")
    state["memory_control_action"] = {
        "type": "save_preference",
        "rule_text": "You prefer very short replies when I'm panicking.",
    }

    delta = await run_memory_control_node(state, cast(Any, _Runtime(store=store)))
    profile = await aget_procedural_profile(store, user_id="user-1")

    assert len(profile.rules) == 1
    assert "very short replies" in profile.rules[0].rule
    assert "Saved:" in delta["response_text"]
