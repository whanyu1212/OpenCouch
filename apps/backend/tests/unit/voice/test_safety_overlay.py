from __future__ import annotations

from dataclasses import asdict

import pytest

from agent.models import CrisisAssessment
from agent.voice.concurrent_safety import VoiceConcurrentSafetyResult
from agent.voice.safety_overlay import VoiceSafetyOverlayService


def _result(
    *,
    status: str = "completed",
    assessment: CrisisAssessment | None = None,
) -> VoiceConcurrentSafetyResult:
    return VoiceConcurrentSafetyResult(
        status=status,  # type: ignore[arg-type]
        reason=None,
        assessment=assessment,
        duration_ms=1.0,
    )


def _assessment(
    *,
    level: int = 2,
    confidence: str = "high",
    needs_crisis_response: bool = True,
) -> CrisisAssessment:
    return CrisisAssessment(
        level=level,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        reason="private classifier reason",
        needs_crisis_response=needs_crisis_response,
    )


@pytest.mark.parametrize("status", ["skipped", "timeout", "failed"])
def test_non_completed_statuses_always_continue(status: str) -> None:
    decision = VoiceSafetyOverlayService().decide(
        _result(status=status, assessment=_assessment(level=3))
    )

    assert asdict(decision) == {
        "action": "continue",
        "risk_level": None,
        "support": None,
    }


@pytest.mark.parametrize(
    "assessment",
    [
        None,
        _assessment(level=1),
        _assessment(confidence="medium"),
        _assessment(needs_crisis_response=False),
    ],
)
def test_completed_non_interrupt_matrix_continues(
    assessment: CrisisAssessment | None,
) -> None:
    decision = VoiceSafetyOverlayService().decide(_result(assessment=assessment))

    assert decision.action == "continue"
    assert decision.risk_level is None
    assert decision.support is None


@pytest.mark.parametrize("level", [2, 3])
def test_high_confidence_crisis_interrupts_with_public_support(level: int) -> None:
    decision = VoiceSafetyOverlayService().decide(
        _result(assessment=_assessment(level=level))
    )

    assert decision.action == "interrupt"
    assert decision.risk_level == level
    assert decision.support is not None
    support = asdict(decision.support)
    assert set(support) == {"headline", "validation", "immediate_step"}
    assert all(support.values())
    assert "private classifier reason" not in str(support)


@pytest.mark.parametrize(
    ("status", "expected_fragment"),
    [
        ("no_location", "country or region"),
        ("location_refused", "not requested"),
        ("no_verified_results", "could be verified"),
        ("lookup_error", "could not be checked"),
    ],
)
def test_resource_resolution_has_clean_fallback_messages(
    status: str,
    expected_fragment: str,
) -> None:
    result = VoiceSafetyOverlayService().resource_resolution(
        inferred_location="",
        resources=[],
        status=status,  # type: ignore[arg-type]
    )

    assert result.status == status
    assert result.resources == []
    assert expected_fragment in result.message
    assert "Do not" not in result.message


def test_resource_resolution_rejects_contacts_without_https_source() -> None:
    result = VoiceSafetyOverlayService().resource_resolution(
        inferred_location="Example",
        resources=[
            {
                "name": "Unverified contact",
                "phone": "12345",
                "url": "http://lookalike.example/contact",
                "region": "Example",
            }
        ],
        status="found",
    )

    assert result.status == "no_verified_results"
    assert result.resources == []
