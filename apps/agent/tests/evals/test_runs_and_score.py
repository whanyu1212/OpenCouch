import json
from dataclasses import asdict
from pathlib import Path

import pytest

from evals.crisis import runs
from evals.crisis.cases import Case
from evals.crisis.record import build_meta
from evals.crisis.runs import (
    ERRORS_FILE,
    RECORDED_FILE,
    CasesChangedError,
    ErrorRow,
    IncompleteRunError,
    InputVersionMismatchError,
    RecordedRow,
    jev_input_version,
    write_meta,
)
from evals.crisis.score import batch_times, load_run, print_summary, summarize
from opencouch_agent.safety import USER_TURN_QUESTIONS


def recorded(case: Case, value: float, ms: int = 500) -> RecordedRow:
    probabilities = dict.fromkeys(USER_TURN_QUESTIONS, value)
    return RecordedRow(
        id=case.id, text=case.text, label=case.label, probabilities=probabilities, ms=ms
    )


def write_run(
    run_dir: Path,
    cases: list[Case],
    rows: list[RecordedRow],
    errors: list[ErrorRow] | None = None,
) -> Path:
    cases_file = run_dir / "cases.jsonl"
    cases_file.write_text("".join(json.dumps(asdict(case)) + "\n" for case in cases))
    write_meta(run_dir, build_meta("test", cases_file, cases, is_bhh=False))
    (run_dir / RECORDED_FILE).write_text(
        "".join(row.model_dump_json() + "\n" for row in rows)
    )
    (run_dir / ERRORS_FILE).write_text(
        "".join(row.model_dump_json() + "\n" for row in errors or [])
    )
    return cases_file


RISK = Case("risk", "I want to end it", "suicidal_ideation")
CALM = Case("calm", "Lovely walk today", "no_crisis")


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
    write_run(tmp_path, [CALM], [recorded(CALM, 0.0)])
    meta = runs.load_meta(tmp_path)
    write_meta(tmp_path, meta.model_copy(update={"jev_input_version": "stale"}))

    with pytest.raises(InputVersionMismatchError, match="Record a new run"):
        load_run(tmp_path)


def test_a_run_whose_cases_were_edited_is_refused(tmp_path: Path) -> None:
    cases_file = write_run(tmp_path, [CALM], [recorded(CALM, 0.0)])
    edited = Case(CALM.id, "an edited message", CALM.label)
    cases_file.write_text(json.dumps(asdict(edited)) + "\n")

    with pytest.raises(CasesChangedError, match="changed since"):
        load_run(tmp_path)


def test_an_interrupted_run_is_refused_until_finished(tmp_path: Path) -> None:
    # CALM has neither an answer nor an error: the run stopped part-way.
    write_run(tmp_path, [RISK, CALM], [recorded(RISK, 1.0)])

    with pytest.raises(IncompleteRunError, match="1 of 2 cases"):
        load_run(tmp_path)


def test_errored_cases_count_as_accounted_for(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        [RISK, CALM],
        [recorded(RISK, 1.0)],
        errors=[ErrorRow(id=CALM.id, error="TimeoutError")],
    )

    _, rows, errors = load_run(tmp_path)

    assert ([row.id for row in rows], errors) == ([RISK.id], 1)


def test_p90_stays_within_the_observed_times() -> None:
    times = batch_times([recorded(CALM, 0.0, ms=100), recorded(RISK, 0.0, ms=200)])
    assert times.p90_ms is not None
    assert 100 <= times.p90_ms <= 200
    assert batch_times([recorded(CALM, 0.0, ms=150)]).p90_ms == 150
    assert batch_times([]).p90_ms is None


def test_summary_reports_rates_headline_and_separation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_run(tmp_path, [RISK, CALM], [recorded(RISK, 1.0), recorded(CALM, 0.0)])

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
