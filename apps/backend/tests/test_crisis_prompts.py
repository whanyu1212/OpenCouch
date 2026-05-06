"""Tests for crisis prompt resource-lookup status wording."""

from __future__ import annotations

from typing import Any, cast

from agent.models import CrisisAssessment
from agent.safety.prompts import build_crisis_response_prompt
from agent.state import AgentState


def _crisis_state(**overrides: Any) -> AgentState:
    """Build minimal crisis prompt state.

    Args:
        **overrides: State fields to override for the prompt case.

    Returns:
        A minimal ``AgentState`` for ``build_crisis_response_prompt``.
    """

    state: dict[str, Any] = {
        "message": "I might hurt myself tonight.",
        "history": [],
        "crisis": CrisisAssessment(
            level=3,
            reason="imminent risk",
            needs_crisis_response=True,
        ),
        "found_resources": [],
        "inferred_location": "",
        "resource_lookup_status": "not_attempted",
    }
    state.update(overrides)
    return cast(AgentState, state)


def test_crisis_prompt_no_location_asks_once_without_pressure() -> None:
    prompt = build_crisis_response_prompt(
        _crisis_state(resource_lookup_status="no_location")
    )

    assert "has not stated their location" in prompt
    assert "Ask once, optionally" in prompt
    assert "Do not pressure them for location" in prompt
    assert "Do not invent phone numbers" in prompt


def test_crisis_prompt_search_failed_uses_general_safety_guidance() -> None:
    prompt = build_crisis_response_prompt(
        _crisis_state(
            inferred_location="Singapore",
            resource_lookup_status="search_failed",
        )
    )

    assert "could not be verified right now" in prompt
    assert "cannot verify local lines right now" in prompt
    assert "nearest emergency department" in prompt
    assert "Do not invent phone numbers" in prompt


def test_crisis_prompt_no_verified_results_names_location_without_numbers() -> None:
    prompt = build_crisis_response_prompt(
        _crisis_state(
            inferred_location="Singapore",
            resource_lookup_status="no_verified_results",
        )
    )

    assert "The user gave this location: Singapore" in prompt
    assert "No verified, actionable local crisis line was found" in prompt
    assert "Do not invent phone numbers" in prompt


def test_crisis_prompt_found_resources_includes_verified_resource_block() -> None:
    prompt = build_crisis_response_prompt(
        _crisis_state(
            inferred_location="Singapore",
            resource_lookup_status="found",
            found_resources=[
                {
                    "name": "Samaritans of Singapore",
                    "phone": "1767",
                    "url": "https://www.sos.org.sg",
                    "region": "Singapore",
                }
            ],
        )
    )

    assert "Verified local crisis resources for Singapore" in prompt
    assert "Samaritans of Singapore: 1767" in prompt
    assert "Do not modify phone numbers" in prompt
