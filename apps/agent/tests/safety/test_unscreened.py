"""The gate when Jev doesn't answer: route from the stored level, recheck later."""

import logging
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from opencouch_agent.safety import FailureReason, RiskLevel, run_gate
from opencouch_agent.safety.gate import MAX_RECHECKS, MAX_UNSCREENED
from opencouch_agent.safety.risk import RiskState, UnscreenedTurn
from tests.fake_jev import ScriptedJev

AT = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
CRISIS_READING = {"intent": 0.9, "ideation": 0.8, "latest_adds_risk": 0.9}
# A first unscreened turn, then the companion's reply to it.
MISSED = "i want to end it tonight"
HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=MISSED)]),
    ModelResponse(parts=[TextPart(content="I'm here. Are you safe right now?")]),
]
MISSED_TURN = UnscreenedTurn(start=0, count=1)


def _said(text: str) -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


def _at(level: RiskLevel) -> RiskState:
    return RiskState(level=level, peak=level)


def _jev(failure: str) -> ScriptedJev:
    if failure == "error":
        return ScriptedJev(failing=frozenset({"hello"}))
    if failure == "timeout":
        return ScriptedJev(timing_out=frozenset({"hello"}))
    # An answer that isn't a probability can't be trusted as "no risk".
    return ScriptedJev({"hello": {"intent": 2.0}})


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("error", FailureReason.ERROR),
        ("timeout", FailureReason.TIMEOUT),
        ("garbage", FailureReason.ERROR),
    ],
)
@pytest.mark.parametrize(
    ("level", "path"),
    [
        (RiskLevel.NONE, "companion"),
        (RiskLevel.WATCH, "companion"),
        (RiskLevel.ELEVATED, "crisis"),
        (RiskLevel.CRISIS, "crisis"),
    ],
)
async def test_a_failed_check_routes_from_the_stored_level_and_marks_the_turn(
    failure: str, reason: FailureReason, level: RiskLevel, path: str
) -> None:
    decision = await run_gate(_jev(failure), _at(level), HISTORY, _said("hello"), at=AT)

    assert decision.path == path
    assert decision.screening_failure is reason
    assert decision.imminent is False
    assert decision.risk == replace(
        _at(level), unscreened=(UnscreenedTurn(start=2, count=1),)
    )


async def test_a_recheck_that_finds_a_crisis_raises_the_level_from_this_turn() -> None:
    jev = ScriptedJev({MISSED: CRISIS_READING})
    risk = RiskState(unscreened=(MISSED_TURN,))

    decision = await run_gate(jev, risk, HISTORY, _said("ok"), at=AT)

    assert jev.screened == [MISSED, "ok"]
    assert decision.path == "crisis"
    assert decision.risk.level is RiskLevel.CRISIS
    assert decision.risk.unscreened == ()
    assert decision.risk.changes[0].reason.startswith("recheck of a missed turn: ")


async def test_a_calm_recheck_clears_the_mark_and_keeps_the_level() -> None:
    risk = RiskState(unscreened=(MISSED_TURN,))

    decision = await run_gate(ScriptedJev(), risk, HISTORY, _said("ok"), at=AT)

    assert decision.path == "companion"
    assert decision.risk.level is RiskLevel.NONE
    assert decision.risk.unscreened == ()
    assert decision.risk.turns == 2


async def test_a_recheck_that_fails_again_stays_marked_and_counts_the_failure() -> None:
    jev = ScriptedJev(failing=frozenset({MISSED}))
    risk = RiskState(unscreened=(MISSED_TURN,))

    decision = await run_gate(jev, risk, HISTORY, _said("ok"), at=AT)

    assert decision.screening_failure is None
    assert decision.risk.unscreened == (replace(MISSED_TURN, failed_rechecks=1),)
    assert decision.abandoned == 0


async def test_a_turn_jev_keeps_rejecting_is_given_up_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    jev = ScriptedJev(failing=frozenset({MISSED}))
    last_try = replace(MISSED_TURN, failed_rechecks=MAX_RECHECKS - 1)

    with caplog.at_level(logging.WARNING, logger="opencouch_agent.safety.gate"):
        decision = await run_gate(
            jev, RiskState(unscreened=(last_try,)), HISTORY, _said("ok"), at=AT
        )

    assert decision.risk.unscreened == ()
    assert decision.abandoned == 1
    assert "Giving up on a missed turn" in caplog.text


async def test_a_turn_after_a_failed_recheck_waits_for_it() -> None:
    history = [*HISTORY, *_said("second")]
    second = UnscreenedTurn(start=2, count=1)
    jev = ScriptedJev({"second": CRISIS_READING}, failing=frozenset({MISSED}))
    risk = RiskState(unscreened=(MISSED_TURN, second))

    decision = await run_gate(jev, risk, history, _said("ok"), at=AT)

    assert decision.risk.unscreened == (
        replace(MISSED_TURN, failed_rechecks=1),
        second,
    )
    assert decision.risk.level is RiskLevel.NONE


async def test_when_jev_is_still_down_both_turns_stay_marked() -> None:
    jev = ScriptedJev(failing=frozenset({MISSED, "hello"}))
    risk = RiskState(unscreened=(MISSED_TURN,))

    decision = await run_gate(jev, risk, HISTORY, _said("hello"), at=AT)

    assert decision.risk.unscreened == (
        replace(MISSED_TURN, failed_rechecks=1),
        UnscreenedTurn(2, 1),
    )


async def test_only_the_newest_unscreened_turns_are_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marks = tuple(UnscreenedTurn(start=index, count=1) for index in range(4))
    history = [message for text in "abcd" for message in _said(text)]
    jev = ScriptedJev(failing=frozenset({*"abcd", "hello"}))

    with caplog.at_level(logging.WARNING, logger="opencouch_agent.safety.gate"):
        decision = await run_gate(
            jev, RiskState(unscreened=marks), history, _said("hello"), at=AT
        )

    assert len(decision.risk.unscreened) == MAX_UNSCREENED
    assert decision.risk.unscreened[0].start == 1
    assert decision.risk.unscreened[-1] == UnscreenedTurn(start=4, count=1)
    assert decision.abandoned == 1
    assert "giving up on the oldest" in caplog.text


@pytest.mark.parametrize(
    "mark",
    [UnscreenedTurn(start=5, count=1), UnscreenedTurn(start=1, count=1)],
    ids=["outside the history", "on the companion's reply"],
)
async def test_a_mark_that_points_elsewhere_is_given_up_on_not_screened(
    mark: UnscreenedTurn, caplog: pytest.LogCaptureFixture
) -> None:
    jev = ScriptedJev()

    with caplog.at_level(logging.ERROR, logger="opencouch_agent.safety.gate"):
        decision = await run_gate(
            jev, RiskState(unscreened=(mark,)), HISTORY, _said("ok"), at=AT
        )

    assert jev.screened == ["ok"]
    assert decision.risk.unscreened == ()
    assert decision.abandoned == 1
    assert "points elsewhere" in caplog.text


async def test_a_rechecked_turn_that_was_imminent_puts_the_emergency_number_first() -> (
    None
):
    imminent = {"imminent": 0.5, "ideation": 0.8, "latest_adds_risk": 0.9}
    risk = RiskState(unscreened=(MISSED_TURN,))

    calm = await run_gate(
        ScriptedJev({MISSED: imminent}), risk, HISTORY, _said("ok"), at=AT
    )
    jev_down = ScriptedJev({MISSED: imminent}, failing=frozenset({"ok"}))
    unscreened = await run_gate(jev_down, risk, HISTORY, _said("ok"), at=AT)

    assert calm.path == unscreened.path == "crisis"
    assert calm.imminent is unscreened.imminent is True
