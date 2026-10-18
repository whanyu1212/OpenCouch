from typing import Any

import pytest
from pydantic import ValidationError

from evals.crisis.conversations import (
    LabelledConversation,
    exchange_cases,
    load_conversations,
    score_conversation,
)
from opencouch_agent.safety import RiskLevel, Turn

NONE, WATCH, ELEVATED, CRISIS = (
    RiskLevel.NONE,
    RiskLevel.WATCH,
    RiskLevel.ELEVATED,
    RiskLevel.CRISIS,
)


def conversation(**labels: Any) -> LabelledConversation:
    fields: dict[str, Any] = {
        "id": "c",
        "category": "slow_build",
        "split": "dev",
        "exchanges": [{"user": f"u{n}", "assistant": f"a{n}"} for n in range(1, 5)],
        "target": "ELEVATED",
        "detect_by": 2,
        "ceiling": "ELEVATED",
        "note": "test",
    } | labels
    return LabelledConversation.model_validate(fields)


def test_every_committed_conversation_passes_the_label_checks() -> None:
    conversations = load_conversations()
    assert len(conversations) == 58
    assert {c.split for c in conversations} == {"dev", "holdout"}


@pytest.mark.parametrize(
    ("labels", "problem"),
    [
        ({"ceiling": ["NONE", "ELEVATED"]}, "one ceiling per exchange"),
        ({"ceiling": ["ELEVATED", "WATCH", "WATCH", "WATCH"]}, "ceiling decreases"),
        ({"detect_by": None}, "go together"),
        ({"detect_by": 9}, "out of range"),
        ({"target": "CRISIS"}, "below target"),
    ],
)
def test_inconsistent_labels_are_rejected(labels: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        conversation(**labels)


def test_each_exchange_carries_everything_said_before_it() -> None:
    cases = exchange_cases(conversation())

    assert [case.id for case in cases] == ["c#1", "c#2", "c#3", "c#4"]
    assert cases[0].history == ()
    assert cases[2].history == (
        Turn("user", "u1"),
        Turn("assistant", "a1"),
        Turn("user", "u2"),
        Turn("assistant", "a2"),
    )
    assert cases[2].text == "u3"


def test_reaching_the_target_early_is_a_negative_delay() -> None:
    score = score_conversation(conversation(), [ELEVATED, ELEVATED, ELEVATED, ELEVATED])
    assert (score.reached_at, score.delay, score.missed) == (1, -1, False)


def test_never_reaching_the_target_is_a_miss() -> None:
    score = score_conversation(conversation(), [NONE, WATCH, WATCH, WATCH])
    assert score.missed
    assert score.delay is None


def test_levels_above_the_ceiling_are_listed_by_exchange() -> None:
    score = score_conversation(conversation(), [NONE, ELEVATED, CRISIS, CRISIS])
    assert score.over_ceiling == [3, 4]


def test_exchanges_still_flagged_after_recovery_are_counted() -> None:
    labelled = conversation(detect_by=1, recovery_from=3)
    score = score_conversation(labelled, [ELEVATED, ELEVATED, ELEVATED, WATCH])
    assert score.flagged_after_recovery == 1
    assert labelled.depends_on_exit_rule


@pytest.mark.parametrize(
    ("levels", "settled"),
    [
        ([ELEVATED, ELEVATED, WATCH, WATCH], 3),
        ([ELEVATED, ELEVATED, ELEVATED, WATCH], "late"),
        ([ELEVATED, ELEVATED, ELEVATED, ELEVATED], "never"),
    ],
)
def test_settling_is_judged_against_settle_by(
    levels: list[RiskLevel], settled: int | str
) -> None:
    labelled = conversation(detect_by=1, settle_by=3, settle_to="WATCH")
    assert score_conversation(labelled, levels).settled == settled
