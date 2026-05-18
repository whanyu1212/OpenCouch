"""Tests for memory-control service behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.runtime import build_initial_state
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.gates.memory_control.service import (
    MemoryControlRequest,
    execute_memory_control_action,
    execute_memory_control_request,
)
from agent.models import AgentInput
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


class _FakePreferenceRuleLLM:
    def __init__(self, *, rule_text: str) -> None:
        self.rule_text = rule_text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by preference saving.")

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
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        if response_schema.__name__ != "PreferenceRuleDecision":
            if response_schema.__name__ == "ProceduralReconciliationDecision":
                return response_schema(
                    action="append",
                    replace_indexes=[],
                    reason="test preference reconciliation appends",
                    confidence="high",
                )
            raise AssertionError(f"Unexpected schema {response_schema.__name__!r}.")
        return response_schema(
            rule_text=self.rule_text,
            reasoning="The user explicitly stated a response preference.",
            confidence="high",
        )


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
    llm_client: Any | None = None,
) -> WorkflowContext:
    return WorkflowContext(
        llm_client=llm_client,
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
async def test_service_lists_saved_memory_from_neutral_request() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(store)

    result = await execute_memory_control_request(
        MemoryControlRequest(
            owner_id="user-1",
            current_user_message="What do you remember about me?",
            action={"type": "list"},
        ),
        _context(store=store),
    )

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
async def test_service_unknown_action_raises() -> None:
    state = _state("memory help")
    state["memory_control"]["action"] = {"type": "unknown"}

    with pytest.raises(ValueError, match="Invalid memory_control.action payload"):
        await execute_memory_control_action(state, _context())


@pytest.mark.asyncio
async def test_service_set_recall_without_enabled_raises() -> None:
    """Malformed set_recall (missing required ``enabled``) must not silently disable.

    Discriminated-union validation rejects the action rather than defaulting
    ``enabled=False`` and turning off proactive recall behind the user's back.
    """

    store = OpenCouchMemoryStore()
    state = _state("turn proactive recall")
    state["memory_control"]["action"] = {"type": "set_recall"}

    with pytest.raises(ValueError, match="Invalid memory_control.action payload"):
        await execute_memory_control_action(state, _context(store=store))

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.proactive_recall_enabled is False


@pytest.mark.asyncio
async def test_service_save_preference_without_text_raises() -> None:
    """Malformed save_preference must not save a blank rule."""

    store = OpenCouchMemoryStore()
    state = _state("remember preference")
    state["memory_control"]["action"] = {"type": "save_preference"}

    with pytest.raises(ValueError, match="Invalid memory_control.action payload"):
        await execute_memory_control_action(state, _context(store=store))

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.rules == []


@pytest.mark.asyncio
async def test_service_save_preference_writes_rule_with_llm() -> None:
    store = OpenCouchMemoryStore()
    state = _state("Remember that I prefer direct answers when I am spiraling.")
    state["memory_control"]["action"] = {
        "type": "save_preference",
        "preference_text": "direct answers when I am spiraling",
    }

    result = await execute_memory_control_action(
        state,
        _context(
            store=store,
            llm_client=_FakePreferenceRuleLLM(
                rule_text="You prefer direct answers when you are spiraling.",
            ),
        ),
    )

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert len(profile.rules) == 1
    assert profile.rules[0].rule == "You prefer direct answers when you are spiraling."
    assert "Saved:" in result.response_text
    assert "direct answers" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_save_preference_requires_llm() -> None:
    store = OpenCouchMemoryStore()
    state = _state("Remember that I prefer direct answers when I am spiraling.")
    state["memory_control"]["action"] = {
        "type": "save_preference",
        "preference_text": "direct answers when I am spiraling",
    }

    with pytest.raises(RuntimeError, match="save_preference requires an LLM client"):
        await execute_memory_control_action(state, _context(store=store))

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
