from dataclasses import fields, replace

import pytest

from opencouch_agent.safety.policy import (
    IMMINENT_CRISIS,
    IMMINENT_ELEVATED,
    INTENT_CRISIS,
    INTENT_ELEVATED,
    RiskLevel,
    assess_level,
    escalate,
    noisy_or,
)
from opencouch_agent.safety.questions import (
    USER_TURN_QUESTIONS,
    UserSignals,
    user_turn_questions,
)

QUIET = UserSignals(**{field.name: 0.0 for field in fields(UserSignals)})


def signals(**probabilities: float) -> UserSignals:
    return replace(QUIET, **probabilities)


def test_signals_cover_exactly_the_questions_asked() -> None:
    assert [field.name for field in fields(UserSignals)] == list(USER_TURN_QUESTIONS)


def test_from_probabilities_requires_every_answer() -> None:
    answers = {name: 0.0 for name in USER_TURN_QUESTIONS}
    assert UserSignals.from_probabilities(answers) == QUIET
    del answers["intent"]
    with pytest.raises(KeyError):
        UserSignals.from_probabilities(answers)


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, float("inf")])
def test_from_probabilities_rejects_values_that_are_not_probabilities(
    value: float,
) -> None:
    # Fail closed: NaN compares False against every threshold and reads as NONE.
    answers = {name: 0.0 for name in USER_TURN_QUESTIONS}
    answers["intent"] = value
    with pytest.raises(ValueError, match="intent"):
        UserSignals.from_probabilities(answers)


def test_question_set_cannot_be_changed_at_runtime() -> None:
    with pytest.raises(TypeError):
        USER_TURN_QUESTIONS["intent"] = "changed"  # type: ignore[index]


def test_changing_a_question_object_does_not_affect_later_requests() -> None:
    questions = user_turn_questions()
    questions["intent"].instructions = "changed"
    questions.pop("plan")
    fresh = user_turn_questions()
    assert fresh["intent"].instructions == USER_TURN_QUESTIONS["intent"]
    assert list(fresh) == list(USER_TURN_QUESTIONS)


def test_quiet_turn_is_none() -> None:
    assert assess_level(QUIET) is RiskLevel.NONE


@pytest.mark.parametrize(
    "probabilities",
    [
        {"intent": INTENT_CRISIS},
        {"imminent": IMMINENT_CRISIS},
        {"preparation": 0.5},
        {"plan": 0.4, "means": 0.4},
    ],
    ids=["intent", "imminence", "preparation", "plan_and_means"],
)
def test_intent_imminence_preparation_or_plan_with_means_is_crisis(
    probabilities: dict[str, float],
) -> None:
    assert assess_level(signals(**probabilities)) is RiskLevel.CRISIS


def test_plan_without_means_is_not_crisis() -> None:
    assert assess_level(signals(plan=0.4, means=0.29)) is RiskLevel.ELEVATED


@pytest.mark.parametrize(
    "probabilities",
    [
        {"intent": INTENT_ELEVATED},
        {"intent": INTENT_CRISIS - 0.01},
        {"imminent": IMMINENT_ELEVATED},
        {"imminent": IMMINENT_CRISIS - 0.01},
        {"plan": 0.25},
        {"self_harm": 0.3},
        {"method_seeking": 0.5},
        {"ideation": 0.4},
        {"harm_to_others": 0.4},
    ],
)
def test_single_strong_signal_below_crisis_is_elevated(
    probabilities: dict[str, float],
) -> None:
    assert assess_level(signals(**probabilities)) is RiskLevel.ELEVATED


def test_strong_ideation_with_faint_intent_and_imminence_is_elevated_not_crisis() -> (
    None
):
    # What Jev gives "i've been thinking about killing myself": no plan, no timing.
    level = assess_level(signals(ideation=0.98, intent=0.31, imminent=0.28))
    assert level is RiskLevel.ELEVATED


@pytest.mark.parametrize(
    "probabilities",
    [{"ideation": 0.2}, {"self_harm": 0.15}, {"harm_to_others": 0.2}],
)
def test_weak_signal_is_watch(probabilities: dict[str, float]) -> None:
    assert assess_level(signals(**probabilities)) is RiskLevel.WATCH


# Borderline: noisy-OR about 0.35, but no single signal reaches ELEVATED.
BORDERLINE = {"ideation": 0.25, "self_harm": 0.14}


@pytest.mark.parametrize("follow_up", ["own_death_wish", "disguised_request"])
def test_follow_up_raises_borderline_turn_to_elevated(follow_up: str) -> None:
    level = assess_level(signals(**BORDERLINE, **{follow_up: 0.6}))
    assert level is RiskLevel.ELEVATED


def test_borderline_turn_without_a_confirming_follow_up_stays_watch() -> None:
    level = assess_level(signals(**BORDERLINE, counselor_concern=0.3))
    assert level is RiskLevel.WATCH


def test_follow_up_alone_does_not_raise_a_turn_with_no_real_signals() -> None:
    assert assess_level(signals(own_death_wish=0.9)) is RiskLevel.NONE


def test_counselor_concern_on_a_faint_borderline_turn_only_watches() -> None:
    # Noisy-OR about 0.23, with no single signal at WATCH.
    faint = {"ideation": 0.1, "self_harm": 0.1, "plan": 0.05}
    assert assess_level(signals(**faint, counselor_concern=0.9)) is RiskLevel.WATCH
    assert assess_level(signals(**faint)) is RiskLevel.NONE


def test_counselor_concern_without_real_signals_is_none() -> None:
    level = assess_level(signals(ideation=0.05, counselor_concern=0.9))
    assert level is RiskLevel.NONE


def test_counselor_concern_never_lowers_a_clear_signal() -> None:
    assert assess_level(signals(intent=0.5)) is RiskLevel.CRISIS


def test_noisy_or_is_zero_with_no_signals() -> None:
    assert noisy_or(QUIET) == 0


def test_noisy_or_adds_up_weak_signals() -> None:
    combined = noisy_or(signals(ideation=0.25, self_harm=0.15, plan=0.1))
    assert combined == pytest.approx(1 - 0.75 * 0.85 * 0.9)


def test_noisy_or_ignores_follow_ups_and_harm_to_others() -> None:
    assert noisy_or(signals(counselor_concern=0.9, harm_to_others=0.9)) == 0


def test_escalate_never_lowers_the_level() -> None:
    assert escalate(RiskLevel.ELEVATED, RiskLevel.NONE) is RiskLevel.ELEVATED


def test_escalate_raises_the_level() -> None:
    assert escalate(RiskLevel.WATCH, RiskLevel.CRISIS) is RiskLevel.CRISIS


def test_levels_are_ordered_by_severity() -> None:
    assert [level.rank for level in RiskLevel] == [0, 1, 2, 3]
    assert RiskLevel.CRISIS.at_least(RiskLevel.ELEVATED)
    assert not RiskLevel.WATCH.at_least(RiskLevel.ELEVATED)
