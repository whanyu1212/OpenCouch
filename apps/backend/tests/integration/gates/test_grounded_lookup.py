"""Tests for grounded factual lookup helpers and nodes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.turn_dispatch import run_turn_dispatch_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.turn_dispatch import build_turn_dispatch_prompt
from agent.tools.grounded_search import answer_factual_lookup
from llm.base import BaseLLMClient, StructuredResponseT


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
    """Fake structured client that records whether search was requested."""

    def __init__(self, structured_responses: list[dict[str, Any] | Exception]) -> None:
        self.structured_responses = list(structured_responses)
        self.structured_calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by grounded lookup.")

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
        use_search: bool = False,
    ) -> StructuredResponseT:
        self.structured_calls.append(
            {
                "prompt": prompt,
                "response_schema": response_schema.__name__,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        if not self.structured_responses:
            raise AssertionError("No fake structured response configured.")
        response = self.structured_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_schema(**response)


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
        use_search: bool = False,
    ) -> StructuredResponseT:
        self.structured_calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        if isinstance(self.decision, Exception):
            raise self.decision
        return response_schema(**self.decision)


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


@pytest.mark.asyncio
async def test_turn_dispatch_routes_grounded_lookup_request() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "grounded_lookup",
            "query": "affordable counselling services in Singapore",
            "reasoning": "The user asked for external service information.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    command = await run_turn_dispatch_node(
        _state("Can you look up affordable counselling services in Singapore?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "grounded_answer_node"
    assert len(llm.structured_calls) == 1
    update = cast(dict[str, Any], command.update)
    assert update["route"] == "grounded_lookup"
    assert update["grounded_lookup"] == {
        "query": "affordable counselling services in Singapore",
        "status": "not_attempted",
    }
    assert update["memory_control"]["action"] == {}
    assert update["diagnostics"]["turn_dispatch_classifier_path"] == "llm_primary"
    trace = update["diagnostics"]["routing_trace"]
    assert trace[-1]["stage"] == "turn_dispatch"
    assert trace[-1]["decision"] == "grounded_lookup"
    assert trace[-1]["source"] == "llm_primary"


@pytest.mark.asyncio
async def test_turn_dispatch_routes_subjective_check_to_therapeutic() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "therapeutic",
            "reasoning": "The user is asking for subjective support.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    command = await run_turn_dispatch_node(
        _state("Can you check if I'm being unreasonable?"),
        cast(Any, _Runtime(llm)),
    )

    assert command.goto == "load_memory_node"
    update = cast(dict[str, Any], command.update)
    assert update["grounded_lookup"] == {"query": "", "status": "not_attempted"}


def test_turn_dispatch_prompt_names_resource_seeking_boundary() -> None:
    """Resource-seeking mental-health asks should be a first-class lookup boundary."""

    prompt = build_turn_dispatch_prompt(
        _state("Can you find credible anxiety worksheets from an official source?")
    )

    assert "look up worksheets" in prompt
    assert "credible articles" in prompt
    assert "Do not route these to therapeutic" in prompt
    assert "in-the-moment help" in prompt


@pytest.mark.asyncio
async def test_turn_dispatch_requires_grounded_lookup_query() -> None:
    llm = _FakeTurnDispatchLLM(
        {
            "route": "grounded_lookup",
            "reasoning": "The model selected lookup but omitted the query.",
            "confidence": "high",
            "active_flow_action": "none",
        }
    )

    with pytest.raises(ValueError, match="without query"):
        await run_turn_dispatch_node(
            _state("Is grounding proven to work?"),
            cast(Any, _Runtime(llm)),
        )


@pytest.mark.asyncio
async def test_answer_factual_lookup_uses_search_grounding() -> None:
    llm = _FakeSearchLLM(
        [
            {
                "status": "search",
                "search_query": "988 outside the United States",
                "answer": "",
                "reasoning": "Verifiable current factual lookup.",
            },
            {
                "status": "answered",
                "answer": (
                    "988 is the US and Canada suicide and crisis line.\n\n"
                    "Sources:\n- 988 Lifeline: https://988lifeline.org"
                ),
                "sources": ["988 Lifeline: https://988lifeline.org"],
                "source_quality": "official",
                "reasoning": "Official source found.",
            },
        ]
    )

    answer, status = await answer_factual_lookup(
        _state("Can you check if 988 works outside the US?"),
        llm_client=llm,
        query="Can you check if 988 works outside the US?",
    )

    assert status == "answered"
    assert "988" in answer
    assert "Sources:" in answer
    assert [call["response_schema"] for call in llm.structured_calls] == [
        "LookupPreflightDecision",
        "GroundedLookupResult",
    ]
    assert [call["use_search"] for call in llm.structured_calls] == [False, True]


@pytest.mark.asyncio
async def test_grounded_answer_node_returns_operational_response() -> None:
    llm = _FakeSearchLLM(
        [
            {
                "status": "search",
                "search_query": "current rule",
                "answer": "",
                "reasoning": "Specific factual lookup.",
            },
            {
                "status": "answered",
                "answer": "Official answer.\n\nSources:\n- Official source",
                "sources": ["Official source"],
                "source_quality": "official",
                "reasoning": "Official source found.",
            },
        ]
    )
    state = _state("Can you look up the current rule?")
    state["grounded_lookup"] = {"query": "Can you look up the current rule?"}

    delta = await run_grounded_answer_node(state, cast(Any, _Runtime(llm)))

    assert delta["route"] == "grounded_lookup"
    assert delta["grounded_lookup"]["status"] == "answered"
    assert delta["response_style"] == "grounded_lookup"
    assert delta["response_text"] == "Official answer.\n\nSources:\n- Official source"


@pytest.mark.asyncio
async def test_answer_factual_lookup_appends_structured_sources() -> None:
    llm = _FakeSearchLLM(
        [
            {
                "status": "search",
                "search_query": "current rule",
                "answer": "",
                "reasoning": "Specific factual lookup.",
            },
            {
                "status": "answered",
                "answer": "Official answer.",
                "sources": ["Official source: https://example.org"],
                "source_quality": "official",
                "reasoning": "Official source found.",
            },
        ]
    )

    answer, status = await answer_factual_lookup(
        _state("Can you look up the current rule?"),
        llm_client=llm,
        query="Can you look up the current rule?",
    )

    assert status == "answered"
    assert (
        answer == "Official answer.\n\nSources:\n- Official source: https://example.org"
    )


@pytest.mark.asyncio
async def test_grounded_answer_node_requires_llm() -> None:
    state = _state("Can you look up the current rule?")
    state["grounded_lookup"] = {"query": "Can you look up the current rule?"}

    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await run_grounded_answer_node(state, cast(Any, _Runtime(None)))


@pytest.mark.asyncio
async def test_answer_factual_lookup_surfaces_preflight_provider_failure() -> None:
    llm = _FakeSearchLLM([RuntimeError("scripted preflight failure")])

    with pytest.raises(RuntimeError, match="scripted preflight failure"):
        await answer_factual_lookup(
            _state("Can you verify whether this is current?"),
            llm_client=llm,
            query="Can you verify whether this is current?",
        )


@pytest.mark.asyncio
async def test_answer_factual_lookup_surfaces_search_provider_failure() -> None:
    llm = _FakeSearchLLM(
        [
            {
                "status": "search",
                "search_query": "current rule",
                "answer": "",
                "reasoning": "Specific factual lookup.",
            },
            RuntimeError("scripted search failure"),
        ]
    )

    with pytest.raises(RuntimeError, match="scripted search failure"):
        await answer_factual_lookup(
            _state("Can you look up the current rule?"),
            llm_client=llm,
            query="Can you look up the current rule?",
        )


@pytest.mark.asyncio
async def test_answer_factual_lookup_marks_unverified_answers() -> None:
    llm = _FakeSearchLLM(
        [
            {
                "status": "no_verified_answer",
                "search_query": "",
                "answer": "I couldn’t verify that from reliable sources.",
                "reasoning": "Not specific enough to verify.",
            },
            {
                "status": "no_verified_answer",
                "search_query": "",
                "answer": "That isn’t something I can verify as an external fact.",
                "reasoning": "Subjective claim.",
            },
        ]
    )

    answer, status = await answer_factual_lookup(
        _state("Can you verify whether this is current?"),
        llm_client=llm,
        query="Can you verify whether this is current?",
    )

    assert status == "no_verified_answer"
    assert "couldn’t verify" in answer.lower()

    answer, status = await answer_factual_lookup(
        _state("Can you check if I'm overreacting?"),
        llm_client=llm,
        query="whether user is overreacting",
    )

    assert status == "no_verified_answer"
    assert "isn’t something" in answer.lower()
