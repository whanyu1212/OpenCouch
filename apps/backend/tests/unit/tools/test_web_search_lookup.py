"""Tests for crisis resource lookup status handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.state import AgentState
from agent.tools.web_search import (
    _normalize_extracted_location,
    find_local_crisis_resources,
)
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeLookupLLM(BaseLLMClient):
    """Fake text client for the two-call resource lookup chain."""

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
        raise AssertionError("Structured generation is not used by web search lookup.")


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
    llm = _FakeLookupLLM([""])

    location, resources, status = await find_local_crisis_resources(
        _state("I don't feel safe right now."),
        llm_client=llm,
    )

    assert location == ""
    assert resources == []
    assert status == "no_location"
    assert [call["use_search"] for call in llm.calls] == [False]


@pytest.mark.asyncio
async def test_lookup_returns_found_for_verified_singapore_resource() -> None:
    llm = _FakeLookupLLM(
        [
            "Singapore",
            "Samaritans of Singapore | 1767 | https://www.sos.org.sg",
        ]
    )

    location, resources, status = await find_local_crisis_resources(
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
async def test_lookup_returns_search_failed_when_search_call_raises() -> None:
    llm = _FakeLookupLLM(["Singapore", RuntimeError("search unavailable")])

    location, resources, status = await find_local_crisis_resources(
        _state(),
        llm_client=llm,
    )

    assert location == "Singapore"
    assert resources == []
    assert status == "search_failed"


@pytest.mark.asyncio
async def test_lookup_returns_no_verified_results_for_unusable_search_text() -> None:
    llm = _FakeLookupLLM(
        [
            "Singapore",
            "I could not verify an official crisis phone number from sources.",
        ]
    )

    location, resources, status = await find_local_crisis_resources(
        _state(),
        llm_client=llm,
    )

    assert location == "Singapore"
    assert resources == []
    assert status == "no_verified_results"


def test_normalize_extracted_location_rejects_placeholder_text() -> None:
    assert _normalize_extracted_location("No location mentioned.") == ""
    assert _normalize_extracted_location("  `Singapore`  ") == "Singapore"
