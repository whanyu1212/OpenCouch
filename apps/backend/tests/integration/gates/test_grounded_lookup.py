"""Tests for grounded factual lookup helpers and nodes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.runtime import build_initial_state
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded import build_grounded_lookup_delta
from agent.tools.grounded_search import (
    GroundedLookupRequest,
    answer_factual_lookup,
    answer_factual_lookup_request,
)
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
async def test_answer_factual_lookup_accepts_neutral_request() -> None:
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
                "sources": ["Official source"],
                "source_quality": "official",
                "reasoning": "Official source found.",
            },
        ]
    )

    answer, status = await answer_factual_lookup_request(
        GroundedLookupRequest(
            query="Can you look up the current rule?",
            current_user_message="Can you look up the current rule?",
            transcript=(),
        ),
        llm_client=llm,
    )

    assert status == "answered"
    assert answer == "Official answer.\n\nSources:\n- Official source"


@pytest.mark.asyncio
async def test_grounded_lookup_returns_operational_response() -> None:
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

    delta = await build_grounded_lookup_delta(state, _Runtime(llm).context)

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
async def test_grounded_lookup_requires_llm() -> None:
    state = _state("Can you look up the current rule?")
    state["grounded_lookup"] = {"query": "Can you look up the current rule?"}

    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await build_grounded_lookup_delta(state, _Runtime(None).context)


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
