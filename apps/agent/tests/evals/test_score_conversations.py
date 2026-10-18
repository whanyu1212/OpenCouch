"""The conversation scorer, and a CI check against the committed recording.

The recording holds Jev's answers for every exchange of the 58 labelled
conversations. CI re-scores the dev split with today's rule: a change that
alters which conversations are missed or over-escalated fails here until the
baseline file is updated with it.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest

from evals.crisis.runs import ERRORS_FILE, RECORDED_FILE, RunCheckError
from evals.crisis.score_conversations import (
    per_message_escalation,
    print_results,
    score_run,
)
from opencouch_agent.safety import USER_TURN_QUESTIONS, RiskLevel, UserSignals

CRISIS_DIR = Path(__file__).parents[2] / "evals" / "crisis"
RECORDING = CRISIS_DIR / "conversations" / "recorded"
DEV_BASELINE = json.loads(
    (CRISIS_DIR / "baselines" / "conversations-dev.json").read_text()
)


def signals(value: float) -> UserSignals:
    return UserSignals.from_probabilities(dict.fromkeys(USER_TURN_QUESTIONS, value))


def test_todays_rule_never_lowers_the_level() -> None:
    levels = per_message_escalation([signals(0.0), signals(1.0), signals(0.0)])
    assert levels == [RiskLevel.NONE, RiskLevel.CRISIS, RiskLevel.CRISIS]


def test_dev_results_match_the_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    results, totals = score_run(RECORDING, split="dev")

    assert totals.unscorable == []
    # Every metric, per conversation and in total: any change to levels,
    # delays, ceilings, recovery or settling must update the baseline too.
    assert asdict(totals) == DEV_BASELINE["totals"]
    assert [asdict(result) for result in results] == DEV_BASELINE["conversations"]

    print_results(results, totals)
    assert "missed CRISIS" in capsys.readouterr().out


def test_a_rule_must_return_one_level_per_exchange() -> None:
    def too_short(signals: Sequence[UserSignals]) -> list[RiskLevel]:
        return [RiskLevel.NONE]

    with pytest.raises(ValueError, match="one per exchange"):
        score_run(RECORDING, split="dev", tracker=too_short)


def test_a_conversation_with_an_errored_exchange_is_unscorable(
    tmp_path: Path,
) -> None:
    for name in ["meta.json", ERRORS_FILE]:
        (tmp_path / name).write_text((RECORDING / name).read_text())
    rows = (RECORDING / RECORDED_FILE).read_text().splitlines()
    dropped = json.loads(rows[0])["id"]
    (tmp_path / RECORDED_FILE).write_text("\n".join(rows[1:]) + "\n")
    (tmp_path / ERRORS_FILE).write_text(
        json.dumps({"id": dropped, "error": "TimeoutError"}) + "\n"
    )

    _, totals = score_run(tmp_path, split="all")

    assert totals.unscorable == [dropped.rsplit("#", 1)[0]]


def test_a_run_of_single_messages_is_refused() -> None:
    sample = CRISIS_DIR / "samples" / "single_turn_recorded"
    with pytest.raises(RunCheckError, match="not a conversations run"):
        score_run(sample)
