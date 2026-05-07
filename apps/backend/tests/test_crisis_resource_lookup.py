"""Tests for the crisis resource lookup graph node."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

import agent.nodes.crisis_resource_lookup as lookup_node
from agent.graph import build_initial_state
from agent.memory.modes import MemoryMode
from agent.models import AgentInput, CrisisAssessment
from agent.nodes.crisis_resource_lookup import run_crisis_resource_lookup_node
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
    """Fake text client for the two-call resource lookup chain."""

    def __init__(self, responses: list[str]) -> None:
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
        return self.responses.pop(0)

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
        raise AssertionError("Structured generation is not used by lookup node tests.")


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
async def test_crisis_resource_lookup_node_skips_without_llm() -> None:
    delta = await run_crisis_resource_lookup_node(
        _state(),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert delta == {
        "inferred_location": "",
        "found_resources": [],
        "resource_lookup_status": "not_attempted",
    }


@pytest.mark.asyncio
async def test_crisis_resource_lookup_node_writes_verified_singapore_resource() -> None:
    llm = _FakeLookupLLM(
        [
            "Singapore",
            "Samaritans of Singapore | 1767 | https://www.sos.org.sg",
        ]
    )

    delta = await run_crisis_resource_lookup_node(
        _state(),
        cast(Any, _FakeRuntime(llm_client=llm)),
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
async def test_crisis_resource_lookup_node_converts_unexpected_failure_to_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_lookup(
        *args: Any, **kwargs: Any
    ) -> tuple[str, list[dict[str, str]], str]:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(lookup_node, "find_local_crisis_resources", _raise_lookup)

    delta = await run_crisis_resource_lookup_node(
        _state(),
        cast(Any, _FakeRuntime(llm_client=_FakeLookupLLM([]))),
    )

    assert delta == {
        "inferred_location": "",
        "found_resources": [],
        "resource_lookup_status": "search_failed",
    }
