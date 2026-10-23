"""Rules that hold for any sequence of turns, not just the cases we wrote.

The tuned numbers can change; these must not.
"""

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from opencouch_agent.safety.floor import initial_risk, risk_floor
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.questions import UserSignals
from opencouch_agent.safety.risk import (
    CheckInAnswer,
    RiskConfig,
    RiskState,
    check_in_due,
    read_turn,
    update_risk,
)

CONFIG = RiskConfig()
START = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)

# Mostly values well clear of the cutoffs, so runs of calm turns and full
# crisis-and-recovery arcs come up often; any float sometimes.
probability = st.one_of(
    st.sampled_from([0.0, 0.0, 0.0, 0.1, 0.6, 0.95]),
    st.floats(min_value=0.0, max_value=1.0),
)
any_signals = st.builds(
    UserSignals, **{field.name: probability for field in fields(UserSignals)}
)
check_ins = st.one_of(st.none(), st.sampled_from(["safe", "not_sure", "need_help"]))
# Minutes since the previous turn: bursts of messages and long silences.
gaps = st.floats(min_value=0.0, max_value=24 * 60)
turns = st.lists(st.tuples(any_signals, gaps, check_ins), min_size=1, max_size=30)

Turns = list[tuple[UserSignals, float, CheckInAnswer | None]]


def replay(turn_list: Turns, state: RiskState | None = None) -> list[RiskState]:
    """Every state from the starting one onwards, one per turn after it."""
    states = [state or RiskState()]
    at = START
    for signals, gap, check_in in turn_list:
        at += timedelta(minutes=gap)
        states.append(update_risk(states[-1], signals, at=at, check_in=check_in))
    return states


@given(turns)
def test_crisis_never_drops_below_elevated(turn_list: Turns) -> None:
    states = replay(turn_list)
    for state in states:
        if state.peak is RiskLevel.CRISIS:
            assert state.level.at_least(RiskLevel.ELEVATED)


@given(turns)
def test_the_level_only_comes_down_one_step_at_a_time(turn_list: Turns) -> None:
    states = replay(turn_list)
    for before, after in pairwise(states):
        if after.level.rank < before.level.rank:
            assert (before.level, after.level) in {
                (RiskLevel.CRISIS, RiskLevel.ELEVATED),
                (RiskLevel.ELEVATED, RiskLevel.WATCH),
            }


@given(turns)
def test_crisis_steps_down_only_on_a_calm_safe_tap_to_an_offered_check_in(
    turn_list: Turns,
) -> None:
    states = replay(turn_list)
    for (signals, _, check_in), before, after in zip(
        turn_list, states[:-1], states[1:], strict=True
    ):
        if before.level is RiskLevel.CRISIS and after.level is not RiskLevel.CRISIS:
            assert check_in_due(before)
            # Spelled out too, so the rule holds even if check_in_due changes.
            assert before.score < CONFIG.crisis_exit_score
            assert before.last_risk_at is not None
            assert before.last_turn_at is not None
            calm_for = before.last_turn_at - before.last_risk_at
            assert calm_for >= timedelta(minutes=CONFIG.min_calm_minutes)
            assert check_in == "safe"
            assert signals.latest_adds_risk < CONFIG.calm_evidence


@given(turns)
def test_a_crisis_turn_that_adds_risk_always_lands_in_crisis(
    turn_list: Turns,
) -> None:
    states = replay(turn_list)
    for (signals, _, _), before, after in zip(
        turn_list, states[:-1], states[1:], strict=True
    ):
        reading, _ = read_turn(signals, before)
        adds_risk = signals.latest_adds_risk >= CONFIG.calm_evidence
        if reading is RiskLevel.CRISIS and adds_risk:
            assert after.level is RiskLevel.CRISIS


def stepped_down() -> RiskState:
    """A conversation that was in CRISIS and has stepped down to ELEVATED."""
    quiet = UserSignals(**{field.name: 0.0 for field in fields(UserSignals)})
    crisis = replace(quiet, intent=0.9, latest_adds_risk=0.9)
    calm: Turns = [(quiet, 1.0, None)] * 10
    state = replay([(crisis, 0.0, None), *calm, (quiet, 1.0, "safe")])[-1]
    assert (state.level, state.peak) == (RiskLevel.ELEVATED, RiskLevel.CRISIS)
    return state


@given(any_signals, gaps)
def test_new_risk_after_a_step_down_always_lands_in_crisis(
    signals: UserSignals, gap: float
) -> None:
    before = stepped_down()
    assert before.last_turn_at is not None
    after = update_risk(
        before, signals, at=before.last_turn_at + timedelta(minutes=gap)
    )
    reading, _ = read_turn(signals, before)
    adds_risk = signals.latest_adds_risk >= CONFIG.calm_evidence
    if adds_risk and reading.at_least(RiskLevel.ELEVATED):
        assert after.level is RiskLevel.CRISIS


@given(turns)
def test_renewed_risk_after_crisis_always_restarts_the_calm_clock(
    turn_list: Turns,
) -> None:
    states = replay(turn_list)
    for (signals, _, _), before, after in zip(
        turn_list, states[:-1], states[1:], strict=True
    ):
        reading, _ = read_turn(signals, before)
        renewed = (
            before.peak is RiskLevel.CRISIS
            and signals.latest_adds_risk >= CONFIG.acute_adds
            and reading.at_least(RiskLevel.ELEVATED)
        )
        if renewed:
            assert after.last_risk_at == after.last_turn_at


@given(turns)
def test_asking_for_help_always_lands_in_crisis(turn_list: Turns) -> None:
    states = replay(turn_list)
    for (_, _, check_in), after in zip(turn_list, states[1:], strict=True):
        if check_in == "need_help":
            assert after.level is RiskLevel.CRISIS


@given(turns)
def test_the_peak_never_falls_and_is_never_below_the_level(turn_list: Turns) -> None:
    states = replay(turn_list)
    for before, after in pairwise(states):
        assert after.peak.rank >= before.peak.rank
        assert after.peak.rank >= after.level.rank


@settings(max_examples=50)
@given(turns, turns)
def test_a_new_thread_starts_where_the_last_one_left_off(
    first: Turns, second: Turns
) -> None:
    ended = replay(first)[-1]
    floor = risk_floor(ended)
    if floor is None:
        assert not ended.level.at_least(RiskLevel.ELEVATED)
        return
    assert ended.last_turn_at is not None
    # The new thread starts before the floor expires, however long ago the
    # last risk was.
    starts_at = min(ended.last_turn_at, floor.expires_at - timedelta(minutes=1))
    started = initial_risk(floor, starts_at)
    assert started.level is ended.level
    for state in replay(second, state=started):
        if ended.peak is RiskLevel.CRISIS:
            assert state.level.at_least(RiskLevel.ELEVATED)
