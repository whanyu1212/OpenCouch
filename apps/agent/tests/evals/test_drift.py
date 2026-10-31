"""Regression alerts are measured only on complete, comparable Jev recordings."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.crisis import detection, drift, score
from evals.crisis.cases import Case
from evals.crisis.drift import Snapshot, check_scope, compare, snapshot
from evals.crisis.runs import ErrorRow, RunCheckError, load_meta, write_meta
from opencouch_agent.safety import RiskLevel, policy, questions
from tests.evals.test_runs_and_score import recorded, write_run

CASES = [
    Case("risk", "risk", "suicidal_ideation"),
    Case("violence", "violence", "violent_thoughts"),
    Case("calm", "calm", "no_crisis"),
]


@pytest.fixture
def reference(tmp_path: Path) -> Snapshot:
    write_run(
        tmp_path,
        CASES,
        [recorded(CASES[0], 1), recorded(CASES[1], 1), recorded(CASES[2], 0)],
    )
    return snapshot(tmp_path)


def test_identical_recording_passes(reference: Snapshot) -> None:
    report = compare(reference, reference)
    assert report["status"] == "pass"
    assert report["changed_cases"] == []


@pytest.mark.parametrize("metric", list(drift.TOLERANCES))
def test_each_metric_trips_beyond_its_predeclared_band(
    reference: Snapshot, metric: str
) -> None:
    rates = dict(reference.rates)
    change = drift.TOLERANCES[metric] + 0.001
    rates[metric] += change if metric.startswith("false_alarms") else -change
    report = compare(reference.model_copy(update={"rates": rates}), reference)
    assert report["status"] == "drift"


@pytest.mark.parametrize("metric", ["false_alarms_watch", "false_alarms_elevated"])
def test_exact_false_alarm_boundary_passes(reference: Snapshot, metric: str) -> None:
    rates = dict(reference.rates)
    rates[metric] += drift.TOLERANCES[metric]
    assert (
        compare(reference.model_copy(update={"rates": rates}), reference)["status"]
        == "pass"
    )


def test_changed_case_ids_are_reported_without_text(reference: Snapshot) -> None:
    current = reference.model_copy(
        update={"levels": reference.levels | {"risk": RiskLevel.NONE}}
    )
    report = compare(current, reference)
    assert report["changed_cases"] == [
        {"id": "risk", "before": "crisis", "after": "none"}
    ]
    assert "text" not in json.dumps(report)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("cases_digest", "changed"),
        ("jev_input_version", "changed"),
        ("limit", 1),
        ("dataset", None),
        ("requested_model", "different/model"),
    ],
)
def test_changed_input_cannot_be_called_drift(
    reference: Snapshot, attribute: str, value: object
) -> None:
    if attribute == "dataset":
        value = {"dataset": "different", "revision": "new", "license": "test"}
    meta = reference.meta.model_copy(update={attribute: value})
    with pytest.raises(RunCheckError):
        check_scope(meta, reference)


def test_policy_edit_requires_explicit_baseline_review(reference: Snapshot) -> None:
    with pytest.raises(RunCheckError, match="Scoring differs"):
        check_scope(
            reference.meta, reference.model_copy(update={"scoring_digest": "old"})
        )


@pytest.mark.parametrize("module", [policy, questions, detection, score])
def test_changed_scoring_module_requires_baseline_review(
    tmp_path: Path, reference: Snapshot, monkeypatch: pytest.MonkeyPatch, module: object
) -> None:
    changed = tmp_path / "changed.py"
    changed.write_text("changed scoring semantics")
    monkeypatch.setattr(module, "__file__", str(changed))
    with pytest.raises(RunCheckError, match="Scoring differs"):
        check_scope(reference.meta, reference)


@pytest.mark.parametrize(
    "problem", ["error", "missing", "duplicate", "extra", "invalid", "label", "text"]
)
def test_bad_measurements_are_not_scored(tmp_path: Path, problem: str) -> None:
    rows = [recorded(case, 0.0) for case in CASES]
    errors = []
    if problem == "error":
        rows.pop()
        errors = [ErrorRow(id="calm", error="TimeoutError")]
    elif problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "extra":
        rows.append(recorded(Case("extra", "extra", "no_crisis"), 0))
    elif problem == "invalid":
        rows[0].probabilities["intent"] = 1.1
    else:
        rows[0] = rows[0].model_copy(update={problem: "different"})
    write_run(tmp_path, CASES, rows, errors)
    with pytest.raises(RunCheckError):
        snapshot(tmp_path)


def test_reference_rejects_nonfinite_rate_at_file_boundary(reference: Snapshot) -> None:
    data = reference.model_dump()
    data["rates"]["false_alarms_watch"] = float("nan")
    with pytest.raises(ValidationError):
        Snapshot.model_validate(data)


@pytest.mark.parametrize("failure,exit_code", [(False, 0), (True, 2)])
def test_cli_writes_report_and_distinguishes_measurement_errors(
    tmp_path: Path,
    reference: Snapshot,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    exit_code: int,
) -> None:
    baseline = tmp_path / "reference.json"
    baseline.write_text(reference.model_dump_json())
    if failure:
        (tmp_path / "errors.jsonl").write_text(
            ErrorRow(id="calm", error="TimeoutError").model_dump_json()
        )
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift",
            "--run-dir",
            str(tmp_path),
            "--reference",
            str(baseline),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as error:
        drift.main()
    assert error.value.code == exit_code
    assert json.loads(report.read_text())["status"] == (
        "measurement_error" if failure else "pass"
    )


async def test_live_run_checks_compatibility_before_contacting_jev(
    tmp_path: Path, reference: Snapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = load_meta(tmp_path)
    write_meta(tmp_path, meta.model_copy(update={"cases_digest": "wrong"}))
    monkeypatch.setattr(drift, "prepare_run", lambda **_: (tmp_path, CASES))
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(RunCheckError, match="differs"):
        await drift.record_live(reference)


def test_simulated_missed_risk_fails_the_real_cli(
    tmp_path: Path, reference: Snapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "reference.json"
    baseline.write_text(reference.model_dump_json())
    write_run(
        tmp_path,
        CASES,
        [recorded(CASES[0], 0), recorded(CASES[1], 1), recorded(CASES[2], 0)],
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift",
            "--run-dir",
            str(tmp_path),
            "--reference",
            str(baseline),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as error:
        drift.main()
    assert error.value.code == 1
    result = json.loads(report.read_text())
    assert result["status"] == "drift"
    assert result["changed_cases"] == [
        {"id": "risk", "before": "crisis", "after": "none"}
    ]


def test_malformed_recording_report_does_not_echo_benchmark_text(
    tmp_path: Path,
    reference: Snapshot,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "reference.json"
    baseline.write_text(reference.model_dump_json())
    (tmp_path / "recorded.jsonl").write_text('{"text":"PRIVATE BENCHMARK TEXT"}')
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift",
            "--run-dir",
            str(tmp_path),
            "--reference",
            str(baseline),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as error:
        drift.main()
    assert error.value.code == 2
    assert "PRIVATE BENCHMARK TEXT" not in capsys.readouterr().out + report.read_text()
