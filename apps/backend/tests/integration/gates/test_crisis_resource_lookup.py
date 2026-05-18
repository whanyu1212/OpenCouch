"""Tests for the crisis resource lookup graph node."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.runtime import build_initial_state
from agent.tools.crisis import build_crisis_resource_lookup_delta
from agent.memory.modes import MemoryMode
from agent.models import AgentInput, CrisisAssessment
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeRuntime:
    """Minimal runtime wrapper exposing ``runtime.context``."""

    def __init__(self, *, llm_client: BaseLLMClient | None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=cast(Any, object()),
            crisis_log_backend=cast(Any, object()),
            memory_mode=MemoryMode.INCOGNITO,
        )


class _FakeLookupLLM(BaseLLMClient):
    """Fake client for the structured location + resource lookup chain."""

    def __init__(
        self,
        *,
        structured_responses: list[dict[str, Any]],
    ) -> None:
        self.structured_responses = list(structured_responses)
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
        raise AssertionError("Crisis resource lookup should use structured output.")

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
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
                "response_schema": response_schema.__name__,
            }
        )
        if not self.structured_responses:
            raise AssertionError("No fake structured response configured.")
        return response_schema(**self.structured_responses.pop(0))


def _state(
    message: str = "I'm in Singapore and I might end my life tonight.",
) -> AgentState:
    state = build_initial_state(
        AgentInput(message=message),
        include_input_history=True,
    )
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )
    return cast(AgentState, dict(state))


@pytest.mark.asyncio
async def test_crisis_resource_lookup_node_requires_llm() -> None:
    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await build_crisis_resource_lookup_delta(
            _state(),
            _FakeRuntime(llm_client=None).context,
        )


@pytest.mark.asyncio
async def test_crisis_resource_lookup_node_writes_verified_singapore_resource() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "provided",
                "location": "Singapore",
                "reasoning": "User stated location.",
            },
            {
                "status": "found",
                "resources": [
                    {
                        "name": "Samaritans of Singapore",
                        "phone": "1767",
                        "url": "https://www.sos.org.sg",
                        "region": "Singapore",
                    }
                ],
                "reasoning": "Verified official resource.",
            },
        ],
    )

    delta = await build_crisis_resource_lookup_delta(
        _state(),
        _FakeRuntime(llm_client=llm).context,
    )

    assert delta == {
        "inferred_location": "Singapore",
        "found_resources": [
            {
                "name": "Samaritans of Singapore",
                "phone": "1767",
                "url": "https://www.sos.org.sg",
                "region": "Singapore",
            }
        ],
        "resource_lookup_status": "found",
    }
    assert [call["use_search"] for call in llm.calls] == [False, True]


@pytest.mark.asyncio
async def test_crisis_resource_lookup_node_records_location_refusal() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "refused",
                "location": "",
                "reasoning": "User declined to share location.",
            }
        ]
    )

    delta = await build_crisis_resource_lookup_delta(
        _state("I might hurt myself, but I don't want to share where I am."),
        _FakeRuntime(llm_client=llm).context,
    )

    assert delta == {
        "inferred_location": "",
        "found_resources": [],
        "resource_lookup_status": "location_refused",
    }
    assert [call["use_search"] for call in llm.calls] == [False]


@pytest.mark.asyncio
async def test_crisis_resource_lookup_node_surfaces_lookup_failure() -> None:
    llm = _FakeLookupLLM(structured_responses=[])

    with pytest.raises(AssertionError, match="No fake structured response configured"):
        await build_crisis_resource_lookup_delta(
            _state(),
            _FakeRuntime(llm_client=llm).context,
        )
