from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta

import pytest

from opencouch_agent.safety.floor import initial_risk, risk_floor
from opencouch_agent.safety.policy import RiskLevel, crisis_routes
from opencouch_agent.safety.questions import UserSignals
from opencouch_agent.safety.risk import (
    CheckInAnswer,
    RiskConfig,
    RiskState,
    check_in_due,
    read_turn,
    update_risk,
)

QUIET = UserSignals(**{field.name: 0.0 for field in fields(UserSignals)})
START = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def signals(**probabilities: float) -> UserSignals:
    return replace(QUIET, **probabilities)


CALM = QUIET
CRISIS_TURN = signals(intent=0.9, ideation=0.8, latest_adds_risk=0.9)
ELEVATED_TURN = signals(ideation=0.5, latest_adds_risk=0.6)

Turn = tuple[UserSignals, float] | tuple[UserSignals, float, CheckInAnswer | None]


def replay(
    turns: Sequence[Turn],
    state: RiskState | None = None,
    config: RiskConfig | None = None,
) -> list[RiskState]:
    """Run each (signals, minutes from START[, check-in]) turn; every state after."""
    current = state or RiskState()
    states = []
    for signals_at, minutes, *check_in in turns:
        current = update_risk(
            current,
            signals_at,
            at=START + timedelta(minutes=minutes),
            check_in=check_in[0] if check_in else None,
            config=config or RiskConfig(),
        )
        states.append(current)
    return states


def levels(states: Sequence[RiskState]) -> list[RiskLevel]:
    return [state.level for state in states]


def in_crisis(minutes: float = 0) -> RiskState:
    """A conversation that reached CRISIS, then went calm."""
    return replay([(CRISIS_TURN, minutes)])[-1]


def calm_turns(start: float, count: int) -> list[Turn]:
    return [(CALM, start + n) for n in range(count)]


# Reading one turn (FirstRead's evidence-backed CRISIS)


def test_stated_intent_is_crisis_on_its_own() -> None:
    level, reason = read_turn(signals(intent=0.9), RiskState())
    assert level is RiskLevel.CRISIS
    assert "intent" in reason


def test_imminence_without_self_harm_context_is_held_at_elevated() -> None:
    level, reason = read_turn(signals(imminent=0.9, latest_adds_risk=0.9), RiskState())
    assert level is RiskLevel.ELEVATED
    assert "no self-harm context" in reason


def test_plan_and_means_echoing_earlier_messages_is_held_at_elevated() -> None:
    echo = signals(plan=0.5, means=0.5, ideation=0.6, latest_adds_risk=0.1)
    assert crisis_routes(echo) == ["plan_means"]
    level, reason = read_turn(echo, RiskState())
    assert level is RiskLevel.ELEVATED
    assert "echoes" in reason


def test_imminence_with_context_and_new_evidence_is_crisis() -> None:
    acute = signals(imminent=0.9, latest_adds_risk=0.9)
    level, _ = read_turn(acute, RiskState(level=RiskLevel.ELEVATED))
    assert level is RiskLevel.CRISIS


def test_a_turn_below_crisis_reads_as_its_assessed_level() -> None:
    assert read_turn(ELEVATED_TURN, RiskState())[0] is RiskLevel.ELEVATED


# Raising and settling


def test_weak_evidence_builds_to_watch_then_elevated_across_turns() -> None:
    weak = signals(latest_adds_risk=0.7)
    states = replay([(weak, n) for n in range(4)])
    assert levels(states) == [
        RiskLevel.NONE,
        RiskLevel.NONE,
        RiskLevel.WATCH,
        RiskLevel.ELEVATED,
    ]
    assert states[-1].changes[-1].reason == "evidence building across turns"


def test_one_message_cannot_build_the_score_past_watch() -> None:
    (state,) = replay([(signals(latest_adds_risk=1.0), 0)])
    assert state.level is RiskLevel.NONE


def test_first_time_rise_counts_even_on_a_calm_turn() -> None:
    (state,) = replay([(signals(ideation=0.5), 0)])
    assert state.level is RiskLevel.ELEVATED


def test_elevated_settles_to_watch_after_calm_turns_and_never_to_none() -> None:
    states = replay([(ELEVATED_TURN, 0), *calm_turns(1, 10)])
    assert levels(states)[3:5] == [RiskLevel.ELEVATED, RiskLevel.WATCH]
    assert states[-1].level is RiskLevel.WATCH


def test_a_calm_echo_does_not_lift_a_settled_conversation() -> None:
    settled = replay([(ELEVATED_TURN, 0), *calm_turns(1, 4)])[-1]
    (state,) = replay([(signals(ideation=0.5), 5)], state=settled)
    assert state.level is RiskLevel.WATCH


def test_elevated_does_not_settle_while_the_score_is_high() -> None:
    config = RiskConfig(settle_after=1, score_elevated=0.1)
    states = replay([(ELEVATED_TURN, 0), (CALM, 1)], config=config)
    assert levels(states) == [RiskLevel.ELEVATED, RiskLevel.ELEVATED]


# Stepping down from CRISIS


def offered() -> RiskState:
    """CRISIS, then ten calm minutes: the reply offers the check-in."""
    return replay(calm_turns(1, 10), state=in_crisis())[-1]


def stepped_down() -> RiskState:
    """A conversation that stepped down from CRISIS on a "safe" tap."""
    return replay([(CALM, 11, "safe")], state=offered())[-1]


def test_crisis_steps_down_when_the_user_taps_safe_on_an_offered_check_in() -> None:
    assert check_in_due(offered())
    state = stepped_down()
    assert state.level is RiskLevel.ELEVATED
    assert state.changes[-1].reason == "the user tapped safe on the check-in"


def test_a_tap_before_the_check_in_was_offered_does_not_count() -> None:
    early = replay(calm_turns(1, 4), state=in_crisis())[-1]
    assert not check_in_due(early)
    (state,) = replay([(CALM, 12, "safe")], state=early)
    assert state.level is RiskLevel.CRISIS


def test_a_tap_while_the_score_is_high_does_not_count() -> None:
    (state,) = replay([(CALM, 30, "safe")], state=in_crisis())
    assert state.level is RiskLevel.CRISIS


def test_typed_reassurance_never_steps_crisis_down() -> None:
    # Typed text, however reassuring, is never a tap.
    (state,) = replay([(CALM, 11)], state=offered())
    assert state.level is RiskLevel.CRISIS


def test_a_safe_tap_on_a_turn_that_adds_risk_does_not_count() -> None:
    (state,) = replay([(signals(latest_adds_risk=0.5), 11, "safe")], state=offered())
    assert state.level is RiskLevel.CRISIS


def test_tapping_not_sure_keeps_crisis() -> None:
    (state,) = replay([(CALM, 11, "not_sure")], state=offered())
    assert state.level is RiskLevel.CRISIS


def test_tapping_need_help_keeps_crisis_and_restarts_the_clock() -> None:
    states = replay([(CALM, 11, "need_help"), (CALM, 12, "safe")], state=offered())
    assert levels(states) == [RiskLevel.CRISIS, RiskLevel.CRISIS]
    assert states[0].last_risk_at == START + timedelta(minutes=11)


def test_the_entry_latch_makes_a_crisis_wait_for_more_calm_turns() -> None:
    turns: list[Turn] = [(CRISIS_TURN, 0), *calm_turns(1, 4), (CALM, 5, "safe")]
    no_wait = RiskConfig(min_calm_minutes=0, crisis_exit_score=0.15)
    unlatched = replace(no_wait, crisis_entry_score=0.0)
    latched = replace(no_wait, crisis_entry_score=2.0)
    assert replay(turns, config=unlatched)[-1].level is RiskLevel.ELEVATED
    held = replay(turns, config=latched)
    assert held[0].score == 2.0
    assert held[-1].level is RiskLevel.CRISIS


def test_renewed_risk_during_a_crisis_restarts_the_clock_and_delays_the_check_in() -> (
    None
):
    still_wants_to = signals(intent=0.9, latest_adds_risk=0.3)
    states = replay(
        [(still_wants_to, 11), (CALM, 12), (CALM, 13, "safe")], state=offered()
    )
    assert states[0].last_risk_at == START + timedelta(minutes=11)
    assert not check_in_due(states[0])
    assert levels(states) == [RiskLevel.CRISIS] * 3


def test_the_check_in_is_only_offered_after_a_calm_turn() -> None:
    restless = signals(latest_adds_risk=0.18)
    (state,) = replay([(restless, 11)], state=offered())
    assert state.level is RiskLevel.CRISIS
    assert state.score < RiskConfig().crisis_exit_score
    assert not check_in_due(state)


def test_only_risk_above_the_reset_cutoff_restarts_the_calm_clock() -> None:
    turns: list[Turn] = [(CRISIS_TURN, 0), (signals(latest_adds_risk=0.3), 4)]
    restarted = replay(turns, config=RiskConfig(clock_reset_adds=0.25))[-1]
    kept = replay(turns, config=RiskConfig(clock_reset_adds=0.35))[-1]
    assert restarted.last_risk_at == START + timedelta(minutes=4)
    assert kept.last_risk_at == START


def test_new_risk_after_a_step_down_goes_straight_back_to_crisis() -> None:
    (state,) = replay([(ELEVATED_TURN, 13)], state=stepped_down())
    assert state.level is RiskLevel.CRISIS
    assert state.changes[-1].reason.startswith("new risk after CRISIS")


def test_a_calm_echo_after_a_step_down_does_not_relapse() -> None:
    (state,) = replay([(signals(intent=0.9), 13)], state=stepped_down())
    assert state.level is RiskLevel.ELEVATED


def test_a_conversation_that_was_in_crisis_never_settles_below_elevated() -> None:
    states = replay(calm_turns(13, 20), state=stepped_down())
    assert {state.level for state in states} == {RiskLevel.ELEVATED}


def test_the_check_in_is_offered_once_only_the_answer_is_missing() -> None:
    assert not check_in_due(replay(calm_turns(1, 4), state=in_crisis())[-1])
    assert check_in_due(offered())
    assert not check_in_due(in_crisis(minutes=30))
    assert not check_in_due(RiskState())


def test_a_state_without_a_risk_time_is_never_calm_for_long_enough() -> None:
    malformed = RiskState(level=RiskLevel.CRISIS, calm_turns=1, last_turn_at=START)
    assert not check_in_due(malformed)


def test_a_turn_time_without_a_timezone_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        update_risk(RiskState(), CALM, at=datetime(2026, 10, 23, 12, 0))


def test_every_change_of_level_is_recorded() -> None:
    states = replay([(ELEVATED_TURN, 0), (CRISIS_TURN, 1)])
    assert [
        (change.turn, change.from_level, change.to_level)
        for change in states[-1].changes
    ] == [
        (1, RiskLevel.NONE, RiskLevel.ELEVATED),
        (2, RiskLevel.ELEVATED, RiskLevel.CRISIS),
    ]


# The per-user floor


def test_crisis_carries_into_a_new_thread_until_the_floor_expires() -> None:
    floor = risk_floor(in_crisis())
    assert floor is not None
    carried = initial_risk(floor, START + timedelta(hours=1))
    assert (carried.level, carried.peak, carried.score) == (
        RiskLevel.CRISIS,
        RiskLevel.CRISIS,
        in_crisis().score,
    )
    assert initial_risk(floor, START + timedelta(hours=72)) == RiskState()


def test_a_carried_crisis_is_not_offered_the_check_in_straight_away() -> None:
    floor = risk_floor(in_crisis())
    state = initial_risk(floor, START + timedelta(minutes=30))
    assert not check_in_due(state)
    (after,) = replay([(CALM, 31, "safe")], state=state)
    assert after.level is RiskLevel.CRISIS


def test_nothing_below_elevated_carries_into_a_new_thread() -> None:
    watch = replay([(signals(ideation=0.2), 0)])[-1]
    assert watch.level is RiskLevel.WATCH
    assert risk_floor(watch) is None
    assert initial_risk(None, START) == RiskState()
