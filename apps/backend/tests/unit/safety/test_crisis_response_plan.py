"""Contract tests for the shared crisis-response policy plan."""

from __future__ import annotations

import pytest

from agent.guardrails.crisis_response import (
    CRISIS_RESPONSE_AVOID,
    build_crisis_response_plan,
)


@pytest.mark.parametrize(
    ("crisis_level", "requested_risk_level", "expected_risk"),
    [
        (2, "imminent", "moderate"),
        (3, "moderate", "imminent"),
        (None, "moderate", "moderate"),
        (None, "high", "high"),
    ],
)
def test_plan_uses_trusted_assessment_over_requested_risk(
    crisis_level: int | None,
    requested_risk_level: str,
    expected_risk: str,
) -> None:
    plan = build_crisis_response_plan(
        crisis_level=crisis_level,
        requested_risk_level=requested_risk_level,
    )

    assert plan.risk_level == expected_risk
    assert plan.max_follow_up_questions == 1
    assert plan.avoid == CRISIS_RESPONSE_AVOID


@pytest.mark.parametrize(
    ("status", "crisis_level", "allows_location_question"),
    [
        ("no_location", 2, True),
        ("no_location", 3, False),
        ("location_refused", 2, False),
        ("no_verified_results", 2, False),
        ("lookup_error", 2, False),
    ],
)
def test_plan_keeps_location_behavior_explicit(
    status: str,
    crisis_level: int,
    allows_location_question: bool,
) -> None:
    plan = build_crisis_response_plan(
        crisis_level=crisis_level,
        resource_lookup_status=status,  # type: ignore[arg-type]
    )

    assert plan.location_question_permitted is allows_location_question
    assert "Do not invent phone numbers" in plan.resource_guidance


def test_plan_requires_verified_resources_without_allowing_number_invention() -> None:
    plan = build_crisis_response_plan(
        crisis_level=3,
        inferred_location="Singapore",
        found_resources=[
            {
                "name": "Samaritans of Singapore",
                "phone": "1767",
                "url": "https://www.sos.org.sg",
                "region": "Singapore",
            }
        ],
        resource_lookup_status="found",
    )

    assert "Samaritans of Singapore: 1767" in plan.resource_guidance
    assert "Only include phone numbers that appear" in plan.text_resource_guidance
    assert any("Do not invent phone numbers" in item for item in plan.avoid)
