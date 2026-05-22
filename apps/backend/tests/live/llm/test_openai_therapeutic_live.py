"""Live-API tests for OpenAI-backed therapeutic response selection.

Mirror of ``test_gemini_therapeutic_live.py`` — same ambiguous cases,
same assertions, different provider. Gated behind ``RUN_LIVE_OPENAI_TESTS=1``
+ an OpenAI API key. In normal pytest runs these tests are skipped.
"""

from __future__ import annotations

import os

import pytest

from agent.runtime import run_agent
from agent.models import AgentInput, ResponseCategory
from llm.factory import create_llm_client


def _has_live_openai_env() -> bool:
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    wants_live = os.getenv("RUN_LIVE_OPENAI_TESTS") == "1"
    return has_key and wants_live


pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(
        not _has_live_openai_env(),
        reason="Set OPENAI_API_KEY and RUN_LIVE_OPENAI_TESTS=1 to run.",
    ),
]


@pytest.mark.asyncio
async def test_live_openai_picks_reflective_for_implicit_pattern() -> None:
    """Live OpenAI should recognize implicit pattern language as reflective."""

    llm_client = create_llm_client(provider="openai")

    result = await run_agent(
        AgentInput(
            message=(
                "It's like I can never say what I actually mean without it "
                "turning into a fight."
            ),
            history=[],
        ),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "reflective", (
        f"Expected reflective but dispatcher picked {result.response_style}. "
        "The LLM should recognize this as an implicit pattern recognition."
    )


@pytest.mark.asyncio
async def test_live_openai_picks_supportive_for_venting() -> None:
    """Live OpenAI should recognize venting signals as supportive."""

    llm_client = create_llm_client(provider="openai")

    result = await run_agent(
        AgentInput(message="I just needed to get that off my chest."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "supportive", (
        f"Expected supportive but dispatcher picked {result.response_style}."
    )


@pytest.mark.asyncio
async def test_live_openai_picks_clarifying_for_ambiguous_pronoun() -> None:
    """Live OpenAI should pick clarifying for a message with an unresolved pronoun."""

    llm_client = create_llm_client(provider="openai")

    result = await run_agent(
        AgentInput(message="It just doesn't make sense to me."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "clarifying", (
        f"Expected clarifying but dispatcher picked {result.response_style}."
    )


@pytest.mark.asyncio
async def test_live_openai_supportive_default_for_neutral_self_report() -> None:
    """Live OpenAI should keep a neutral self-report in supportive mode."""

    llm_client = create_llm_client(provider="openai")

    result = await run_agent(
        AgentInput(message="I had a rough day at work today."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "supportive", (
        f"Expected supportive but dispatcher picked {result.response_style}."
    )
