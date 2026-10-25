from datetime import UTC, datetime

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from opencouch_agent.safety import FailureReason, RiskLevel, run_gate
from opencouch_agent.safety.risk import RiskState
from tests.fake_jev import ScriptedJev

AT = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
CRISIS_READING = {"intent": 0.9, "ideation": 0.8, "latest_adds_risk": 0.9}
IN_CRISIS = RiskState(
    level=RiskLevel.CRISIS, peak=RiskLevel.CRISIS, score=0.9, last_risk_at=AT
)


def _said(text: str) -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


@pytest.mark.parametrize(
    ("reading", "level", "path"),
    [
        ({}, RiskLevel.NONE, "companion"),
        ({"ideation": 0.2, "latest_adds_risk": 0.3}, RiskLevel.WATCH, "companion"),
        ({"ideation": 0.5, "latest_adds_risk": 0.6}, RiskLevel.ELEVATED, "companion"),
        (CRISIS_READING, RiskLevel.CRISIS, "crisis"),
    ],
)
async def test_each_level_takes_its_path(
    reading: dict[str, float], level: RiskLevel, path: str
) -> None:
    jev = ScriptedJev({"message": reading})

    decision = await run_gate(jev, RiskState(), [], _said("message"), at=AT)

    assert decision.risk.level is level
    assert decision.path == path
    assert decision.screening_failure is None


async def test_a_calm_message_in_a_crisis_still_goes_to_the_crisis_responder() -> None:
    decision = await run_gate(ScriptedJev(), IN_CRISIS, [], _said("thanks"), at=AT)

    assert decision.risk.level is RiskLevel.CRISIS
    assert decision.path == "crisis"


async def test_a_crisis_reading_without_self_harm_context_goes_to_the_companion() -> (
    None
):
    # Imminence alone needs self-harm context to make CRISIS; it's held at ELEVATED.
    jev = ScriptedJev({"tonight": {"imminent": 0.5, "latest_adds_risk": 0.9}})

    decision = await run_gate(jev, RiskState(), [], _said("tonight"), at=AT)

    assert decision.risk.level is RiskLevel.ELEVATED
    assert decision.path == "companion"


async def test_a_failed_check_at_elevated_goes_to_the_crisis_responder() -> None:
    elevated = RiskState(level=RiskLevel.ELEVATED, peak=RiskLevel.ELEVATED)
    jev = ScriptedJev(failing=frozenset({"hello"}))

    decision = await run_gate(jev, elevated, [], _said("hello"), at=AT)

    assert jev.screened == ["hello"]
    assert decision.path == "crisis"
    assert decision.risk.level is RiskLevel.ELEVATED
    assert decision.screening_failure is FailureReason.ERROR


@pytest.mark.parametrize(
    ("reading", "imminent"),
    [
        (CRISIS_READING, False),
        ({"imminent": 0.5, "ideation": 0.8, "latest_adds_risk": 0.9}, True),
        (
            {"plan": 0.6, "means": 0.6, "ideation": 0.8, "latest_adds_risk": 0.9},
            True,
        ),
        ({"preparation": 0.6, "ideation": 0.8, "latest_adds_risk": 0.9}, True),
    ],
    ids=["intent alone", "imminent", "plan and means", "preparation"],
)
async def test_a_crisis_is_imminent_only_on_a_timeline_plan_or_preparation(
    reading: dict[str, float], imminent: bool
) -> None:
    jev = ScriptedJev({"message": reading})

    decision = await run_gate(jev, RiskState(), [], _said("message"), at=AT)

    assert decision.path == "crisis"
    assert decision.imminent is imminent


async def test_a_failed_check_is_not_treated_as_imminent() -> None:
    jev = ScriptedJev(failing=frozenset({"hello"}))

    decision = await run_gate(jev, RiskState(), [], _said("hello"), at=AT)

    assert decision.imminent is False
