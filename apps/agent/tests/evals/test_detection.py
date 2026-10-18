import math

import pytest

from evals.crisis.detection import (
    RecallAtTarget,
    auc,
    rate,
    recall_at_false_alarms,
    score_rows,
    separation,
)
from evals.crisis.runs import RecordedRow
from opencouch_agent.safety import USER_TURN_QUESTIONS, RiskLevel


def row(case_id: str, label: str, value: float) -> RecordedRow:
    return RecordedRow(
        id=case_id,
        text="a message",
        label=label,
        probabilities=dict.fromkeys(USER_TURN_QUESTIONS, value),
        ms=500,
    )


def test_perfect_separation_scores_auc_one_and_full_recall() -> None:
    assert auc([0.9, 0.8], [0.1, 0.0]) == 1.0
    assert recall_at_false_alarms([0.9, 0.8], [0.1, 0.0], 0.05) == RecallAtTarget(
        0.05, 1.0, 0.0
    )


def test_identical_scores_give_auc_one_half() -> None:
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_recall_uses_the_lowest_threshold_within_the_false_alarm_target() -> None:
    positives = [0.9, 0.6, 0.3]
    negatives = [0.7, 0.2, 0.1, 0.0]
    # From 0.3 up to 0.7, one negative of four is flagged (25%), and at 0.3
    # every positive is caught. Above 0.7 no negative is flagged.
    at_quarter = recall_at_false_alarms(positives, negatives, 0.25)
    assert (at_quarter.recall, at_quarter.false_alarms) == (1.0, 0.25)
    at_tenth = recall_at_false_alarms(positives, negatives, 0.10)
    assert (at_tenth.recall, at_tenth.false_alarms) == (1 / 3, 0.0)


def test_invalid_answers_are_counted_as_invalid_never_as_none() -> None:
    nan_row = row("nan", "no_crisis", math.nan)
    missing_question = row("missing", "no_crisis", 0.0)
    del missing_question.probabilities["ideation"]

    scored = score_rows([nan_row, missing_question, row("ok", "no_crisis", 0.0)])

    assert scored.invalid == ["nan", "missing"]
    assert [case.id for case in scored.cases] == ["ok"]
    assert scored.cases[0].level is RiskLevel.NONE


def test_certain_risk_assesses_as_crisis() -> None:
    [case] = score_rows([row("sure", "suicidal_ideation", 1.0)]).cases
    assert case.level is RiskLevel.CRISIS


def test_rate_is_none_without_cases() -> None:
    assert rate([], RiskLevel.WATCH) is None


def test_separation_needs_both_positives_and_negatives() -> None:
    only_positives = score_rows([row("a", "self_harm", 0.9)]).cases
    assert separation(only_positives) is None

    cases = score_rows([row("a", "self_harm", 0.9), row("b", "no_crisis", 0.1)]).cases
    result = separation(cases)
    assert result is not None
    assert result.auc == pytest.approx(1.0)
    assert [at.target for at in result.recall_at_false_alarms] == [0.03, 0.05, 0.10]
