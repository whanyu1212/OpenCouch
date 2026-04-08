"""Hybrid tests for the LLM-backed therapeutic mode classifier fallback."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent.graph import build_initial_state
from agent.models import AgentInput, Message, MessageRole
from agent.subgraphs.therapeutic import (
    select_therapeutic_mode,
)
from services.llm.base import BaseLLMClient


class FakeClassifierResponse(BaseModel):
    """Controlled response for the fake LLM client."""

    mode: str
    confidence: str
    reason: str


class FakeModeLLMClient(BaseLLMClient):
    """Fake provider client for hybrid mode-routing tests."""

    def __init__(
        self,
        response: FakeClassifierResponse,
        *,
        raise_on_structured: bool = False,
    ) -> None:
        self.response = response
        self.raise_on_structured = raise_on_structured
        self.structured_calls = 0
        self.last_prompt: str | None = None
        self.last_system_instruction: str | None = None

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        raise NotImplementedError("Text generation not used in mode classifier tests.")

    async def generate_text_stream(
        self, *, prompt, system_instruction=None, temperature=0
    ):
        yield ""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> BaseModel:
        self.structured_calls += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.raise_on_structured:
            raise RuntimeError("Simulated structured generation failure")
        return response_schema(**self.response.model_dump())


@pytest.mark.asyncio
async def test_llm_classifier_fires_when_no_keyword_and_no_intent_match() -> None:
    """LLM classifier should be called when keyword and intent layers miss."""

    state = build_initial_state(
        AgentInput(message="I don't know what I need right now.")
    )

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="guided_exercise",
            confidence="medium",
            reason="User seems uncertain, exercise might help ground them.",
        )
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 1
    assert mode == "guided_exercise"
    assert source == "llm"


@pytest.mark.asyncio
async def test_llm_classifier_bypassed_when_keyword_matches() -> None:
    """LLM should not be called when a keyword pattern matches."""

    state = build_initial_state(
        AgentInput(message="Can you give me a grounding exercise?")
    )

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="psychoeducation",
            confidence="high",
            reason="Should not be used.",
        )
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 0
    assert mode == "guided_exercise"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_llm_classifier_bypassed_when_session_intent_provides_route() -> None:
    """LLM should not be called when session intent provides a route."""

    state = build_initial_state(
        AgentInput(
            message="Okay let's keep going.",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="I want to do CBT for this.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="We can work through that.",
                ),
            ],
        )
    )
    state["session_intent"] = "guided_cbt_work"
    state["session_intent_source"] = "explicit"

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="psychoeducation",
            confidence="high",
            reason="Should not be used.",
        )
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 0
    assert mode == "guided_exercise"
    assert source == "session_intent"


@pytest.mark.asyncio
async def test_llm_classifier_clamps_invalid_mode_to_default() -> None:
    """LLM returning an invalid mode should fall back to supportive_conversation."""

    state = build_initial_state(
        AgentInput(message="I don't know what I need right now.")
    )

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="crisis_response",
            confidence="high",
            reason="Attempted to return a safety-critical mode.",
        )
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 1
    assert mode == "supportive_conversation"
    assert source == "llm"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_default() -> None:
    """LLM call failure should fall back to supportive_conversation."""

    state = build_initial_state(
        AgentInput(message="I don't know what I need right now.")
    )

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="guided_exercise",
            confidence="high",
            reason="Should not be used.",
        ),
        raise_on_structured=True,
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 1
    assert mode == "supportive_conversation"
    assert source == "default"


@pytest.mark.asyncio
async def test_no_llm_client_falls_back_to_default() -> None:
    """No LLM client should fall back to supportive_conversation."""

    state = build_initial_state(
        AgentInput(message="I don't know what I need right now.")
    )

    mode, source = await select_therapeutic_mode(state, llm_client=None)

    assert mode == "supportive_conversation"
    assert source == "default"


@pytest.mark.asyncio
async def test_safety_critical_modes_never_overridden_by_llm() -> None:
    """Out-of-scope keyword match should bypass LLM even if LLM would disagree."""

    state = build_initial_state(AgentInput(message="Can you diagnose me?"))

    llm_client = FakeModeLLMClient(
        FakeClassifierResponse(
            mode="supportive_conversation",
            confidence="high",
            reason="Should not be used.",
        )
    )

    mode, source = await select_therapeutic_mode(state, llm_client=llm_client)

    assert llm_client.structured_calls == 0
    assert mode == "out_of_scope"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_mode_source_set_correctly_in_subgraph() -> None:
    """mode_source should be populated in state after run_therapeutic_subgraph."""

    from agent.subgraphs.therapeutic import run_therapeutic_subgraph

    state = build_initial_state(
        AgentInput(message="Can you give me a grounding exercise?")
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "guided_exercise"
    assert state["mode_source"] == "keyword"
