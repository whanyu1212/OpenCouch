"""Live-API tests for the Gemini-backed therapeutic dispatcher.

These tests exercise the dispatcher's LLM path against the real Gemini
provider. They use messages that require semantic judgment rather than the
minimal no-LLM dispatch fallback.

Gated behind ``RUN_LIVE_GEMINI_TESTS=1`` + a Gemini API key. In normal
pytest runs these tests are skipped.
"""

from __future__ import annotations

import os

import pytest

from agent.graph import run_agent
from agent.models import AgentInput, ResponseCategory
from llm.factory import create_llm_client


def _has_live_gemini_env() -> bool:
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    wants_live = os.getenv("RUN_LIVE_GEMINI_TESTS") == "1"
    return has_key and wants_live


pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(
        not _has_live_gemini_env(),
        reason="Set GEMINI_API_KEY or GOOGLE_API_KEY and RUN_LIVE_GEMINI_TESTS=1 to run.",
    ),
]


@pytest.mark.asyncio
async def test_live_gemini_picks_reflective_for_implicit_pattern() -> None:
    """Live Gemini should recognize implicit pattern language as reflective.

    The user describes a recurring dynamic without using any of the
    obvious keywords ('keep', 'always', 'every time'). The LLM dispatcher
    should pick reflective from the described dynamic.
    """

    llm_client = create_llm_client(provider="gemini")

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
async def test_live_gemini_picks_supportive_for_venting() -> None:
    """Live Gemini should recognize venting signals as supportive.

    The user wants to be heard, not probed. No pattern language, no
    confusion markers. A good dispatcher picks supportive.
    """

    llm_client = create_llm_client(provider="gemini")

    result = await run_agent(
        AgentInput(message="I just needed to get that off my chest."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "supportive", (
        f"Expected supportive but dispatcher picked {result.response_style}."
    )


@pytest.mark.asyncio
async def test_live_gemini_picks_clarifying_for_ambiguous_pronoun() -> None:
    """Live Gemini should pick clarifying for a message with an unresolved pronoun.

    'It just doesn't make sense to me' has an ambiguous 'it' that lacks
    an antecedent. A good dispatcher asks 'what doesn't make sense?'.
    """

    llm_client = create_llm_client(provider="gemini")

    result = await run_agent(
        AgentInput(message="It just doesn't make sense to me."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "clarifying", (
        f"Expected clarifying but dispatcher picked {result.response_style}."
    )


@pytest.mark.asyncio
async def test_live_gemini_supportive_default_for_neutral_self_report() -> None:
    """Live Gemini should keep a neutral self-report in supportive mode.

    Regression guard: the LLM shouldn't pull a neutral self-report into
    reflective or clarifying just because it 'sounds interesting'.
    """

    llm_client = create_llm_client(provider="gemini")

    result = await run_agent(
        AgentInput(message="I had a rough day at work today."),
        llm_client=llm_client,
    )

    assert result.response_type == ResponseCategory.THERAPEUTIC
    assert result.response_style == "supportive", (
        f"Expected supportive but dispatcher picked {result.response_style}."
    )
