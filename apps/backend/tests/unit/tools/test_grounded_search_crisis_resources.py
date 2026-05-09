"""Tests for crisis resource lookup status handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.state import AgentState
from agent.tools.grounded_search import (
    _normalize_extracted_location,
    find_crisis_resources,
)
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeLookupLLM(BaseLLMClient):
    """Fake client for the structured location + resource lookup chain."""

    def __init__(
        self,
        *,
        structured_responses: list[dict[str, Any] | Exception],
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
        response = self.structured_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_schema(**response)


def _state(message: str = "I'm scared and I'm in Singapore.") -> AgentState:
    """Build the minimal state read by the lookup helper."""

    return cast(
        AgentState,
        {
            "message": message,
            "history": [{"role": "user", "content": message}],
        },
    )


@pytest.mark.asyncio
async def test_lookup_returns_no_location_without_search_call() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "not_provided",
                "location": "",
                "reasoning": "No location stated.",
            }
        ]
    )

    location, resources, status = await find_crisis_resources(
        _state("I don't feel safe right now."),
        llm_client=llm,
    )

    assert location == ""
    assert resources == []
    assert status == "no_location"
    assert [call["use_search"] for call in llm.calls] == [False]
    assert [call["response_schema"] for call in llm.calls] == ["CrisisLocationDecision"]


@pytest.mark.asyncio
async def test_lookup_returns_location_refused_without_search_call() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "refused",
                "location": "",
                "reasoning": "User declined to share location.",
            }
        ]
    )

    location, resources, status = await find_crisis_resources(
        _state("I don't want to share where I am."),
        llm_client=llm,
    )

    assert location == ""
    assert resources == []
    assert status == "location_refused"
    assert [call["use_search"] for call in llm.calls] == [False]


@pytest.mark.asyncio
async def test_lookup_returns_found_for_verified_singapore_resource() -> None:
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

    location, resources, status = await find_crisis_resources(
        _state(),
        llm_client=llm,
    )

    assert location == "Singapore"
    assert status == "found"
    assert resources == [
        {
            "name": "Samaritans of Singapore",
            "phone": "1767",
            "url": "https://www.sos.org.sg",
            "region": "Singapore",
        }
    ]
    assert [call["use_search"] for call in llm.calls] == [False, True]


@pytest.mark.asyncio
async def test_lookup_surfaces_resource_search_failure() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "provided",
                "location": "Singapore",
                "reasoning": "User stated location.",
            },
            RuntimeError("search unavailable"),
        ],
    )

    with pytest.raises(RuntimeError, match="search unavailable"):
        await find_crisis_resources(
            _state(),
            llm_client=llm,
        )


@pytest.mark.asyncio
async def test_lookup_returns_no_verified_results_when_none_found() -> None:
    llm = _FakeLookupLLM(
        structured_responses=[
            {
                "status": "provided",
                "location": "Singapore",
                "reasoning": "User stated location.",
            },
            {
                "status": "no_verified_results",
                "resources": [],
                "reasoning": "No verified actionable resource found.",
            },
        ],
    )

    location, resources, status = await find_crisis_resources(
        _state(),
        llm_client=llm,
    )

    assert location == "Singapore"
    assert resources == []
    assert status == "no_verified_results"


def test_normalize_extracted_location_rejects_placeholder_text() -> None:
    assert _normalize_extracted_location("No location mentioned.") == ""
    assert _normalize_extracted_location("  `Singapore`  ") == "Singapore"
