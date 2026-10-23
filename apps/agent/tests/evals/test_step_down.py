"""The step-down cases, how they're recorded, and how a setting is scored."""

import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.crisis import record
from evals.crisis.conversations import LabelledConversation
from evals.crisis.record import prepare_run
from evals.crisis.runs import (
    ERRORS_FILE,
    RECORDED_FILE,
    ErrorRow,
    RecordedRow,
    RunCheckError,
    load_meta,
)
from evals.crisis.score_conversations import LoadedSplit
from evals.crisis.step_down import (
    AXES,
    Setting,
    grid_search,
    load_step_down_run,
    replay,
    score_step_down,
)
from evals.crisis.step_down_cases import (
    CHECK_IN_TEXT,
    STEP_DOWN_PATH,
    StepDownConversation,
    load_step_down,
    step_down_cases,
)
from opencouch_agent.safety import USER_TURN_QUESTIONS, RiskLevel, Turn, UserSignals
from opencouch_agent.safety.risk import DEFAULT_RISK_CONFIG

C, E = "CRISIS", "ELEVATED"


def exchange(user: str, minutes: float, **extra: Any) -> dict[str, Any]:
    return {"user": user, "assistant": f"reply to {user}", "minutes": minutes, **extra}


def conversation(**labels: Any) -> StepDownConversation:
    fields: dict[str, Any] = {
        "id": "case",
        "category": "reset_split",
        "split": "dev",
        "exchanges": [exchange("one", 0), exchange("two", 1, new_thread=True)],
        "floor": [C, C],
        "ceiling": [C, C],
        "note": "",
    }
    return StepDownConversation.model_validate(fields | labels)


def signals(**values: float) -> UserSignals:
    answers = dict.fromkeys(USER_TURN_QUESTIONS, 0.0) | values
    return UserSignals.from_probabilities(answers)


CRISIS_TURN = signals(intent=0.9, latest_adds_risk=0.9)
CALM = signals()


def test_every_committed_step_down_conversation_passes_the_label_checks() -> None:
    conversations = load_step_down(STEP_DOWN_PATH)
    categories = {conversation.category for conversation in conversations}
    assert categories == {
        "crisis_recovery",
        "crisis_masking",
        "crisis_relapse",
        "reset_split",
    }
    assert {conversation.split for conversation in conversations} == {"dev", "holdout"}


@pytest.mark.parametrize(
    ("labels", "problem"),
    [
        ({"floor": [C]}, "one floor and one ceiling"),
        ({"floor": [C, C], "ceiling": [E, C]}, "floor above ceiling"),
        (
            {"exchanges": [exchange("one", 5), exchange("two", 1)]},
            "minutes go backwards",
        ),
        (
            {"exchanges": [exchange("one", 0, new_thread=True), exchange("two", 1)]},
            "can't start a new thread",
        ),
        (
            {"exchanges": [exchange("one", 0), exchange("ok", 1, check_in="safe")]},
            "the tap's text",
        ),
    ],
)
def test_inconsistent_step_down_labels_are_rejected(
    labels: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValidationError, match=problem):
        conversation(**labels)


def test_duplicate_conversation_ids_are_rejected(tmp_path: Path) -> None:
    line = conversation().model_dump_json() + "\n"
    path = tmp_path / "step_down.jsonl"
    path.write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_step_down(path)


def test_a_new_thread_starts_with_no_history_for_jev() -> None:
    first, second = step_down_cases(conversation())
    assert (first.id, second.id) == ("case#1", "case#2")
    assert first.history == ()
    assert second.history == ()
    later = conversation(
        exchanges=[exchange("one", 0), exchange("two", 1)], floor=[C, C]
    )
    assert step_down_cases(later)[1].history == (
        Turn("user", "one"),
        Turn("assistant", "reply to one"),
    )


def test_replay_carries_the_level_into_a_new_thread() -> None:
    levels = replay(conversation(), {1: CRISIS_TURN, 2: CALM}, DEFAULT_RISK_CONFIG)
    assert levels == [RiskLevel.CRISIS, RiskLevel.CRISIS]


def test_levels_below_the_floor_and_above_the_ceiling_are_listed() -> None:
    labelled = conversation(floor=[C, E], ceiling=[C, E])
    result = score_step_down(labelled, [RiskLevel.ELEVATED, RiskLevel.CRISIS])
    assert (result.below_floor, result.above_ceiling) == ([1], [2])


def test_a_setting_knows_its_place_in_the_grid() -> None:
    first = Setting(*(axis[0] for axis in AXES))
    assert first.position() == (0, 0, 0, 0)
    assert first.config().crisis_exit_score == AXES[0][0]


def test_the_grid_fails_early_step_downs_and_ranks_passing_settings_first() -> None:
    # A "safe" tap four minutes after a crisis, labelled too soon: only
    # settings with no time floor and a high exit score step down.
    too_soon = conversation(
        category="crisis_masking",
        exchanges=[
            exchange("crisis", 0),
            exchange("calm", 1),
            exchange("calm", 2),
            exchange("calm", 3),
            exchange(CHECK_IN_TEXT["safe"], 4, check_in="safe"),
        ],
        floor=[C] * 5,
        ceiling=[C] * 5,
    )
    signals_by_exchange = {1: CRISIS_TURN, 2: CALM, 3: CALM, 4: CALM, 5: CALM}

    rows = grid_search([(too_soon, signals_by_exchange)], None)

    assert len(rows) == math.prod(len(axis) for axis in AXES)
    passes = [row.passes for row in rows]
    assert True in passes and False in passes
    assert passes == sorted(passes, reverse=True)
    failing = [row.setting for row in rows if not row.passes]
    assert all(setting.calm_minutes == 0 for setting in failing)


def test_a_setting_fails_if_it_misses_what_the_per_message_rule_catches() -> None:
    # Imminence with no self-harm context: the per-message rule says CRISIS,
    # the conversation rule holds it at ELEVATED, so it misses the target.
    imminent = LabelledConversation.model_validate(
        {
            "id": "imminent",
            "category": "turns_serious",
            "split": "dev",
            "exchanges": [{"user": "tonight", "assistant": "..."}],
            "target": "CRISIS",
            "detect_by": 1,
            "ceiling": "CRISIS",
            "note": "",
        }
    )
    detection = LoadedSplit(
        scorable=[(imminent, [signals(imminent=0.9, latest_adds_risk=0.9)])],
        unscorable=[],
    )
    calm = conversation(floor=[C, C])

    rows = grid_search([(calm, {1: CRISIS_TURN, 2: CALM})], detection)

    assert not any(row.passes for row in rows)
    assert {row.missed for row in rows} == {1}


def new_step_down_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "step_down.jsonl"
    path.write_text(conversation().model_dump_json() + "\n")
    monkeypatch.setattr(record, "RESULTS_DIR", tmp_path / "results")
    run_dir, cases = prepare_run(
        split=None, cases_path=None, resume=None, step_down_path=path
    )
    assert load_meta(run_dir).source == "step_down"
    assert [case.id for case in cases] == ["case#1", "case#2"]
    return run_dir


def recorded_row(case_id: str) -> str:
    row = RecordedRow(
        id=case_id,
        text="",
        label="case",
        probabilities=dict.fromkeys(USER_TURN_QUESTIONS, 0.0),
        ms=1,
    )
    return row.model_dump_json() + "\n"


def test_a_step_down_run_is_recorded_one_case_per_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = new_step_down_run(tmp_path, monkeypatch)
    (run_dir / RECORDED_FILE).write_text(
        recorded_row("case#1") + recorded_row("case#2")
    )

    [(loaded, exchanges)] = load_step_down_run(run_dir, "all")

    assert loaded.id == "case"
    assert sorted(exchanges) == [1, 2]
    assert load_step_down_run(run_dir, "holdout") == []


def test_a_limited_run_loads_only_the_conversations_it_fully_covers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "step_down.jsonl"
    path.write_text(conversation().model_dump_json() + "\n")
    monkeypatch.setattr(record, "RESULTS_DIR", tmp_path / "results")
    run_dir, cases = prepare_run(
        split=None, cases_path=None, resume=None, step_down_path=path, limit=1
    )
    assert [case.id for case in cases] == ["case#1"]
    (run_dir / RECORDED_FILE).write_text(recorded_row("case#1"))

    assert load_step_down_run(run_dir, "all") == []


def test_a_conversation_with_an_errored_exchange_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = new_step_down_run(tmp_path, monkeypatch)
    (run_dir / RECORDED_FILE).write_text(recorded_row("case#1"))
    error = ErrorRow(id="case#2", error="TimeoutError")
    (run_dir / ERRORS_FILE).write_text(error.model_dump_json() + "\n")

    with pytest.raises(RunCheckError, match="no valid answer"):
        load_step_down_run(run_dir, "dev")


def test_a_conversations_run_is_not_a_step_down_run() -> None:
    recorded = Path(__file__).parents[2] / "evals/crisis/conversations/recorded"
    with pytest.raises(RunCheckError, match="not a step-down run"):
        load_step_down_run(recorded, "dev")
