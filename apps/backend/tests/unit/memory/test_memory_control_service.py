"""Tests for memory-control service behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.control.service import (
    MemoryControlRequest,
    execute_memory_control_request,
)
from agent.runtime_context import WorkflowContext


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


def _request(
    message: str,
    action: dict[str, Any],
    *,
    owner_id: str | None = "user-1",
    pending_action: dict[str, Any] | None = None,
) -> MemoryControlRequest:
    return MemoryControlRequest(
        owner_id=owner_id,
        current_user_message=message,
        action=action,
        pending_action=pending_action,
        session_id="thread-1",
    )


async def _seed_fact(
    store: OpenCouchMemoryStore,
    *,
    owner_id: str = "user-1",
    fact_id: str = "fact-presentations",
    object_identifier: str = "presentations",
    evidence_quote: str = "Presentations make me anxious.",
) -> None:
    now = iso_now()
    fact = SemanticFact(
        id=fact_id,
        category="trigger",
        subject=EntityRef(type="User", identifier=owner_id),
        predicate="WORRIES_ABOUT",
        object=EntityRef(type="Event", identifier=object_identifier),
        evidence_quote=evidence_quote,
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

    result = await execute_memory_control_request(
        _request("What do you remember about me?", {"type": "list"}),
        _context(store=store),
    )

    assert "Presentations make me anxious" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_noops_in_incognito_mode() -> None:
    result = await execute_memory_control_request(
        _request("What do you remember about me?", {"type": "list"}),
        _context(memory_mode=MemoryMode.INCOGNITO),
    )

    assert "guest mode" in result.response_text
    assert result.memory_control == {"pending_action": None}


@pytest.mark.asyncio
async def test_service_unknown_action_raises() -> None:
    with pytest.raises(ValueError, match="Invalid memory action payload"):
        await execute_memory_control_request(
            _request("memory help", {"type": "unknown"}),
            _context(),
        )


@pytest.mark.asyncio
async def test_service_set_recall_without_enabled_raises() -> None:
    """Malformed set_recall (missing required ``enabled``) must not silently disable.

    Discriminated-union validation rejects the action rather than defaulting
    ``enabled=False`` and turning off proactive recall behind the user's back.
    """

    store = OpenCouchMemoryStore()

    with pytest.raises(ValueError, match="Invalid memory action payload"):
        await execute_memory_control_request(
            _request("turn proactive recall", {"type": "set_recall"}),
            _context(store=store),
        )

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.proactive_recall_enabled is False


@pytest.mark.asyncio
async def test_service_save_preference_without_text_raises() -> None:
    """Malformed save_preference must not save a blank rule."""

    store = OpenCouchMemoryStore()

    with pytest.raises(ValueError, match="Invalid memory action payload"):
        await execute_memory_control_request(
            _request("remember preference", {"type": "save_preference"}),
            _context(store=store),
        )

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.rules == []


@pytest.mark.asyncio
async def test_service_save_preference_writes_rule_with_llm() -> None:
    store = OpenCouchMemoryStore()

    result = await execute_memory_control_request(
        _request(
            "Remember that I prefer direct answers when I am spiraling.",
            {
                "type": "save_preference",
                "preference_text": "direct answers when I am spiraling",
            },
        ),
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

    with pytest.raises(RuntimeError, match="save_preference requires an LLM client"):
        await execute_memory_control_request(
            _request(
                "Remember that I prefer direct answers when I am spiraling.",
                {
                    "type": "save_preference",
                    "preference_text": "direct answers when I am spiraling",
                },
            ),
            _context(store=store),
        )

    profile = await aget_procedural_profile(store, user_id="user-1")
    assert profile.rules == []


@pytest.mark.asyncio
async def test_service_confirm_pending_deletes_target() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(store)

    result = await execute_memory_control_request(
        _request(
            "yes, delete it",
            {"type": "confirm_pending"},
            pending_action={
                "type": "delete",
                "target": {
                    "kind": "fact",
                    "namespace": ["user-1", "semantic"],
                    "key": "fact-presentations",
                    "rule_id": None,
                    "preview": "trigger: presentations",
                },
            },
        ),
        _context(store=store),
    )

    assert "Deleted that saved fact" in result.response_text
    assert result.memory_control == {"pending_action": None}
    assert await store.aget(("user-1", "semantic"), "fact-presentations") is None


@pytest.mark.asyncio
async def test_service_forget_by_query_keeps_multiple_matches_pending() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(
        store,
        fact_id="fact-presentations",
        evidence_quote="Presentations make me anxious.",
    )
    await _seed_fact(
        store,
        fact_id="fact-work-presentations",
        evidence_quote="Work presentations make my chest tight.",
    )

    result = await execute_memory_control_request(
        _request(
            "Forget the saved memory about presentations.",
            {"type": "forget_by_query", "query": "presentations"},
        ),
        _context(store=store),
    )

    pending_action = result.memory_control["pending_action"]
    assert pending_action["type"] == "delete_options"
    assert [target["key"] for target in pending_action["targets"]] == [
        "fact-presentations",
        "fact-work-presentations",
    ]
    assert "Which one should I delete?" in result.response_text


@pytest.mark.asyncio
async def test_service_forget_by_query_selects_pending_match_by_number() -> None:
    store = OpenCouchMemoryStore()
    await _seed_fact(
        store,
        fact_id="fact-presentations",
        evidence_quote="Presentations make me anxious.",
    )
    await _seed_fact(
        store,
        fact_id="fact-work-presentations",
        evidence_quote="Work presentations make my chest tight.",
    )
    ambiguous_result = await execute_memory_control_request(
        _request(
            "Forget the saved memory about presentations.",
            {"type": "forget_by_query", "query": "presentations"},
        ),
        _context(store=store),
    )

    result = await execute_memory_control_request(
        _request(
            "Delete number 2.",
            {"type": "forget_by_query", "query": "2"},
            pending_action=ambiguous_result.memory_control["pending_action"],
        ),
        _context(store=store),
    )

    pending_action = result.memory_control["pending_action"]
    assert pending_action["type"] == "delete"
    assert pending_action["target"]["key"] == "fact-work-presentations"
    assert "Do you want me to delete it?" in result.response_text
    assert (
        await store.aget(("user-1", "semantic"), "fact-work-presentations") is not None
    )


@pytest.mark.asyncio
async def test_service_forget_by_query_keeps_options_on_invalid_pending_selection() -> (
    None
):
    store = OpenCouchMemoryStore()
    await _seed_fact(
        store,
        fact_id="fact-presentations",
        evidence_quote="Presentations make me anxious.",
    )
    await _seed_fact(
        store,
        fact_id="fact-work-presentations",
        evidence_quote="Work presentations make my chest tight.",
    )
    ambiguous_result = await execute_memory_control_request(
        _request(
            "Forget the saved memory about presentations.",
            {"type": "forget_by_query", "query": "presentations"},
        ),
        _context(store=store),
    )
    pending_options = ambiguous_result.memory_control["pending_action"]

    result = await execute_memory_control_request(
        _request(
            "Delete number 9.",
            {"type": "forget_by_query", "query": "9"},
            pending_action=pending_options,
        ),
        _context(store=store),
    )

    assert result.memory_control["pending_action"] == pending_options
    assert "Please choose one of the listed memory options" in result.response_text
