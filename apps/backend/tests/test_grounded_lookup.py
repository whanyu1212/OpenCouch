"""Tests for grounded factual lookup helpers and nodes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput, ModeType, ResponseCategory
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.grounded_lookup_gate import (
    _detect_grounded_lookup_action,
    run_grounded_lookup_gate_node,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_lookup import answer_grounded_lookup
from services.llm.base import BaseLLMClient, StructuredResponseT


class _Runtime:
    """Minimal runtime wrapper for grounded lookup node tests."""

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


class _FakeSearchLLM(BaseLLMClient):
    """Fake text client that records whether search was requested."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        if not self.responses:
            raise AssertionError("No fake response configured.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

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
        raise AssertionError("Structured generation is not used by answer tests.")


class _FakeLookupClassifierLLM(BaseLLMClient):
    """Fake structured client for grounded-lookup routing tests."""

    def __init__(
        self,
        *,
        should_lookup: bool,
        query: str | None = None,
        confidence: str = "high",
    ) -> None:
        self.should_lookup = should_lookup
        self.query = query
        self.confidence = confidence
        self.structured_calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by lookup classification.")

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
        return response_schema(
            should_lookup=self.should_lookup,
            query=self.query,
            reasoning="fake lookup routing decision",
            confidence=self.confidence,
        )


def _state(message: str) -> AgentState:
    """Build a seeded state for grounded lookup tests."""

    state = build_initial_state(
        AgentInput(
            message=message,
            user_id="user-1",
            session_id="thread-1",
            history=[],
            working_memory=[],
        )
    )
    return cast(AgentState, dict(state))


def test_detect_grounded_lookup_requires_explicit_factual_signal() -> None:
    assert _detect_grounded_lookup_action(
        "Can you look up affordable counselling services in Singapore?"
    ) == {"query": "Can you look up affordable counselling services in Singapore?"}
    assert _detect_grounded_lookup_action(
        "Can you check if 988 works outside the US?"
    ) == {"query": "Can you check if 988 works outside the US?"}
    assert (
        _detect_grounded_lookup_action(
            "Can you check whether this wearable is evidence-based for anxiety?"
        )
        is None
    )
    assert (
        _detect_grounded_lookup_action("I'm overwhelmed about finding a therapist.")
        is None
    )
    assert (
        _detect_grounded_lookup_action("Can you check if I'm being unreasonable?")
        is None
    )
    assert _detect_grounded_lookup_action("Something else is on my mind today.") is None
    assert (
        _detect_grounded_lookup_action(
            "Before we wrap up, what's the main thing from today?"
        )
        is None
    )


@pytest.mark.asyncio
async def test_grounded_lookup_gate_routes_explicit_search_request() -> None:
    llm = _FakeLookupClassifierLLM(should_lookup=False)
    command = await run_grounded_lookup_gate_node(
        _state("Can you look up affordable counselling services in Singapore?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "grounded_answer_node"
    assert llm.structured_calls == []
    assert command.update["route"] == "grounded_lookup"
    assert (
        command.update["grounded_lookup_query"]
        == "Can you look up affordable counselling services in Singapore?"
    )
    assert command.update["grounded_lookup_status"] == "not_attempted"


@pytest.mark.asyncio
async def test_grounded_lookup_gate_passes_ordinary_support_to_memory_load() -> None:
    llm = _FakeLookupClassifierLLM(should_lookup=True)
    command = await run_grounded_lookup_gate_node(
        _state("I'm overwhelmed about finding a therapist."),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "load_memory_node"
    assert llm.structured_calls == []
    assert command.update["grounded_lookup_query"] == ""
    assert command.update["grounded_lookup_status"] == "not_attempted"


@pytest.mark.asyncio
async def test_grounded_lookup_gate_passes_subjective_check_without_llm_call() -> None:
    llm = _FakeLookupClassifierLLM(should_lookup=True)
    command = await run_grounded_lookup_gate_node(
        _state("Can you check if I'm being unreasonable?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "load_memory_node"
    assert llm.structured_calls == []
    assert command.update["grounded_lookup_query"] == ""


@pytest.mark.asyncio
async def test_grounded_lookup_gate_routes_ambiguous_factual_request_with_llm() -> None:
    llm = _FakeLookupClassifierLLM(
        should_lookup=True,
        query="wearable evidence base for anxiety",
    )
    command = await run_grounded_lookup_gate_node(
        _state("Can you check whether this wearable is evidence-based for anxiety?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "grounded_answer_node"
    assert len(llm.structured_calls) == 1
    assert command.update["route"] == "grounded_lookup"
    assert (
        command.update["grounded_lookup_query"] == "wearable evidence base for anxiety"
    )
    assert (
        command.update["diagnostics"]["grounded_lookup_classifier_path"]
        == "llm_primary"
    )


@pytest.mark.asyncio
async def test_grounded_lookup_gate_passes_ambiguous_low_confidence() -> None:
    llm = _FakeLookupClassifierLLM(
        should_lookup=True,
        query="unclear search",
        confidence="low",
    )
    command = await run_grounded_lookup_gate_node(
        _state("Is this advice evidence-based for people like me?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "load_memory_node"
    assert len(llm.structured_calls) == 1
    assert command.update["grounded_lookup_query"] == ""


@pytest.mark.asyncio
async def test_grounded_lookup_gate_passes_ambiguous_without_llm() -> None:
    command = await run_grounded_lookup_gate_node(
        _state("Is grounding proven to work?"),
        cast(Any, _Runtime(None)),
    )

    assert command.goto == "load_memory_node"
    assert command.update["grounded_lookup_query"] == ""
    assert (
        command.update["diagnostics"]["grounded_lookup_classifier_path"]
        == "deterministic"
    )


@pytest.mark.asyncio
async def test_answer_grounded_lookup_uses_search_grounding() -> None:
    llm = _FakeSearchLLM(
        [
            "988 is the US and Canada suicide and crisis line.\n\n"
            "Sources:\n- 988 Lifeline: https://988lifeline.org"
        ]
    )

    answer, status = await answer_grounded_lookup(
        _state("Can you check if 988 works outside the US?"),
        llm_client=llm,
        query="Can you check if 988 works outside the US?",
    )

    assert status == "answered"
    assert "988" in answer
    assert "Sources:" in answer
    assert [call["use_search"] for call in llm.calls] == [True]


@pytest.mark.asyncio
async def test_grounded_answer_node_returns_operational_response() -> None:
    llm = _FakeSearchLLM(["Official answer.\n\nSources:\n- Official source"])
    state = _state("Can you look up the current rule?")
    state["grounded_lookup_query"] = "Can you look up the current rule?"

    delta = await run_grounded_answer_node(state, cast(Any, _Runtime(llm)))

    assert delta["route"] == "grounded_lookup"
    assert delta["grounded_lookup_status"] == "answered"
    assert delta["response_style"] == "grounded_lookup"
    assert delta["response_style_source"] == "grounded_lookup_gate"
    assert delta["response_style_type"] == ModeType.OPERATIONAL
    assert delta["response_kind"] == ResponseCategory.THERAPEUTIC
    assert delta["response_text"] == "Official answer.\n\nSources:\n- Official source"


@pytest.mark.asyncio
async def test_grounded_answer_node_does_not_guess_without_llm() -> None:
    state = _state("Can you look up the current rule?")
    state["grounded_lookup_query"] = "Can you look up the current rule?"

    delta = await run_grounded_answer_node(state, cast(Any, _Runtime(None)))

    assert delta["grounded_lookup_status"] == "search_unavailable"
    assert "don't want to guess" in delta["response_text"]


@pytest.mark.asyncio
async def test_answer_grounded_lookup_marks_unverified_answers() -> None:
    llm = _FakeSearchLLM(["I couldn't verify that from reliable sources."])

    answer, status = await answer_grounded_lookup(
        _state("Can you verify whether this is current?"),
        llm_client=llm,
        query="Can you verify whether this is current?",
    )

    assert status == "no_verified_answer"
    assert "couldn't verify" in answer.lower()
