"""Tests for memory-control service behavior."""

from __future__ import annotations

from typing import cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural import aget_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.user_controls.service import execute_memory_control_action
from agent.models import AgentInput
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


def _state(message: str, *, user_id: str = "user-1") -> AgentState:
    state = build_initial_state(
        AgentInput(message=message, user_id=user_id, session_id="thread-1"),
        include_input_history=True,
    )
    return cast(AgentState, dict(state))


def _context(
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


async def _seed_fact(store: OpenCouchMemoryStore, *, owner_id: str = "user-1") -> None:
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


@pytest.mark.asyncio
async def test_service_lists_saved_memory() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(store)
    state = _state("What do you remember about me?")
    state["memory_control"]["action"] = {"type": "list"}

    result = await execute_memory_control_action(state, _context(store=store))

    assert "Presentations make me anxious" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_noops_in_incognito_mode() -> None:
    state = _state("What do you remember about me?")
    state["memory_control"]["action"] = {"type": "list"}

    result = await execute_memory_control_action(
        state,
        _context(memory_mode=MemoryMode.INCOGNITO),
    )

    assert "guest mode" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_unknown_action_returns_capability_reply() -> None:
    state = _state("memory help")
    state["memory_control"]["action"] = {"type": "unknown"}

    result = await execute_memory_control_action(state, _context())

    assert "I can show saved memory" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_set_recall_without_enabled_returns_capability_reply() -> None:
    """Malformed set_recall (missing required ``enabled``) must not silently disable.

    Discriminated-union validation rejects the action; the service falls through
    to the capability reply rather than defaulting ``enabled=False`` and turning
    off proactive recall behind the user's back. The result must not emit a
    ``procedural_profile`` update — that would propagate a fabricated value into
    graph state.
    """

    store = OpenCouchMemoryStore()
    state = _state("turn proactive recall")
    state["memory_control"]["action"] = {"type": "set_recall"}

    result = await execute_memory_control_action(state, _context(store=store))

    assert "I can show saved memory" in result.response_text
    assert result.procedural_profile is None
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_save_preference_without_rule_returns_capability_reply() -> None:
    """Malformed save_preference (missing ``rule_text``) must not save a blank rule."""

    store = OpenCouchMemoryStore()
    state = _state("remember preference")
    state["memory_control"]["action"] = {"type": "save_preference"}

    result = await execute_memory_control_action(state, _context(store=store))

    assert "I can show saved memory" in result.response_text
    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.rules == []


@pytest.mark.asyncio
async def test_service_confirm_pending_deletes_target() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(store)
    state = _state("yes, delete it")
    state["memory_control"] = {
        "action": {"type": "confirm_pending"},
        "pending_action": {
            "type": "delete",
            "target": {
                "kind": "fact",
                "namespace": ["user-1", "semantic"],
                "key": "fact-presentations",
                "rule_id": None,
                "preview": "trigger: presentations",
            },
        },
    }

    result = await execute_memory_control_action(state, _context(store=store))

    assert "Deleted that saved fact" in result.response_text
    assert result.memory_control == {"pending_action": None}
    assert await store.aget(("user-1", "semantic"), "fact-presentations") is None
