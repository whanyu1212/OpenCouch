from pathlib import Path

import pytest

from evals.crisis import runs
from evals.crisis.record import build_meta
from evals.crisis.runs import (
    RECORDED_FILE,
    InputVersionMismatchError,
    RecordedRow,
    jev_input_version,
    write_meta,
)
from evals.crisis.score import load_run, print_summary, summarize
from opencouch_agent.safety import USER_TURN_QUESTIONS


def write_run(run_dir: Path, rows: list[RecordedRow]) -> None:
    write_meta(run_dir, build_meta("test", run_dir / "cases.jsonl", is_bhh=False))
    (run_dir / RECORDED_FILE).write_text(
        "".join(row.model_dump_json() + "\n" for row in rows)
    )


def recorded(case_id: str, label: str, value: float) -> RecordedRow:
    probabilities = dict.fromkeys(USER_TURN_QUESTIONS, value)
    return RecordedRow(
        id=case_id, text="text", label=label, probabilities=probabilities, ms=500
    )


def test_changing_the_transcript_framing_changes_the_input_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = jev_input_version()
    monkeypatch.setattr(
        runs, "user_check_state", lambda history, latest: f"NEW FRAMING {latest}"
    )
    assert jev_input_version() != before


def test_a_run_recorded_with_another_input_version_is_refused(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, [])
    meta = runs.load_meta(tmp_path)
    write_meta(tmp_path, meta.model_copy(update={"jev_input_version": "stale"}))

    with pytest.raises(InputVersionMismatchError, match="Record a new run"):
        load_run(tmp_path)


def test_summary_reports_rates_headline_and_separation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_run(
        tmp_path,
        [
            recorded("risk", "suicidal_ideation", 1.0),
            recorded("calm", "no_crisis", 0.0),
        ],
    )

    summary = summarize(tmp_path)

    assert (summary.counts.scored, summary.counts.invalid) == (2, 0)
    assert [rates.label for rates in summary.by_label] == [
        "suicidal_ideation",
        "no_crisis",
    ]
    assert summary.headline.suicide_self_harm_recall_elevated == 1.0
    assert summary.headline.false_alarms_watch == 0.0
    assert summary.headline.violent_thoughts_watch is None
    assert summary.separation is not None
    assert summary.separation.auc == 1.0

    print_summary(summary)
    assert "AUC 1.000" in capsys.readouterr().out
