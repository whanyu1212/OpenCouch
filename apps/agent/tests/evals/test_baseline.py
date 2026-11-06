"""Offline baseline integrity, comparison and measurement tests."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from evals.baseline.__main__ import main
from evals.baseline.compare import comparison
from evals.baseline.cost import coach_cost
from evals.baseline.judging import read_rows
from evals.baseline.models import Prices, TokenRates
from evals.baseline.report import build_report, prompt_digest
from evals.baseline.statistics import cluster_interval, percentile
from evals.judge.jobs import load_jobs
from evals.judge.models import (
    HardFailures,
    JudgeAttempt,
    JudgeErrorRow,
    Judgment,
    QualityScores,
    QualityVerdict,
)
from evals.judge.run import job_digest
from evals.runner.cases import load_suite, suite_digest
from evals.runner.runs import (
    ErrorRow,
    ModelCallRecord,
    SavedRunRow,
    SuiteRunMeta,
    TurnRecord,
    write_run_files,
)


def write_rows(path: Path, rows: Sequence[BaseModel]) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))


@pytest.fixture
def recording(tmp_path: Path) -> tuple[Path, Path]:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        json.dumps(
            [
                {
                    "id": name,
                    "split": "dev",
                    "tags": ["ordinary"],
                    "turns": [
                        {
                            "user": "PRIVATE USER TEXT",
                            "expect": {"path": "coach", "risk_level": "NONE"},
                        }
                    ],
                }
                for name in ("work", "sleep")
            ]
        )
    )
    cases = load_suite(suite)
    rows = [
        SavedRunRow(
            case_id=case.id,
            rep=rep,
            split="dev",
            requested_model="test",
            turns=[
                TurnRecord(
                    turn_index=0,
                    user=case.turns[0].user,
                    reply="PRIVATE REPLY TEXT",
                    path="coach",
                    risk_level="NONE",
                    first_token_ms=rep * 2000,
                    turn_ms=rep * 3000,
                    model_calls=[
                        ModelCallRecord(
                            served_model="test",
                            input_tokens=100,
                            cache_read_tokens=20,
                            output_tokens=10,
                            model_ms=rep * 1000,
                        )
                    ],
                )
            ],
        )
        for case in cases
        for rep in (1, 2)
    ]
    run = tmp_path / "coach"
    write_run_files(
        run,
        SuiteRunMeta(
            suite="tiny",
            suite_path=str(suite),
            suite_digest=suite_digest(cases),
            split="dev",
            requested_model="test",
            reps=2,
            started_at="2026-11-05",
        ),
        rows,
        [],
        "",
    )
    judge = tmp_path / "judge"
    judge.mkdir()
    rubric = "A fixed test rubric"
    (judge / "rubric.md").write_text(rubric)
    jobs = load_jobs(run)
    (judge / "meta.json").write_text(
        json.dumps(
            {
                "mode": "quality",
                "rubric_digest": hashlib.sha256(rubric.encode()).hexdigest(),
                "schema_digest": "schema-v1",
                "input_digest": job_digest(jobs),
                "judge_reps": 2,
                "replies": 4,
                "routes": [],
                "calibrated": False,
            }
        )
    )
    verdict = QualityVerdict.model_validate(
        {
            "hard_failures": {
                name: {"failed": False, "reason": "PRIVATE REASON"}
                for name in HardFailures.model_fields
            },
            "scores": {
                name: {"score": 4, "reason": "PRIVATE REASON"}
                for name in QualityScores.model_fields
            },
        }
    )
    judgments = [
        Judgment(
            case_id=job.case_id,
            rep=job.rep,
            turn_index=0,
            judge_rep=rep,
            input=job.input,
            verdict=verdict,
            attempts=[
                JudgeAttempt(
                    route="anthropic",
                    requested_model="judge",
                    served_model="judge",
                    cost_usd=0.01,
                )
            ],
        )
        for job in jobs
        for rep in (1, 2)
    ]
    write_rows(judge / "results.jsonl", judgments)
    (judge / "errors.jsonl").touch()
    return run, judge


def test_report_counts_timings_and_omits_transcripts(
    recording: tuple[Path, Path],
) -> None:
    run, judge = recording
    snapshot = build_report(run, judge=judge)
    assert snapshot.metrics["behaviour.passed"] == 4
    assert snapshot.metrics["first_token.p50_ms"] == 3000
    assert snapshot.metrics["first_token.p90_ms"] == 4000
    assert snapshot.metrics["first_token.over_3s"] == 2
    assert snapshot.metrics["judge.judgments"] == 8
    assert snapshot.metrics["cost.judge_known_usd"] == pytest.approx(0.08)
    assert snapshot.metrics["cost.coach_estimate_usd"] is None
    assert snapshot.intervals["judge.hard_fail_rate"] is None
    assert "PRIVATE" not in snapshot.model_dump_json()
    text, failed = comparison(snapshot, snapshot)
    assert not failed
    assert "No changed conversation checks" in text


def test_failure_and_incomplete_run_are_not_hidden(
    recording: tuple[Path, Path],
) -> None:
    run, _ = recording
    baseline = build_report(run)
    rows = read_rows(run / "results.jsonl", SavedRunRow)
    rows[0].turns[0].path = "crisis"
    rows[0].turns[0].truncated = True
    rows[0].turns[0].reply = ""
    write_rows(run / "results.jsonl", rows[:-1])
    candidate = build_report(run)
    assert candidate.metrics["behaviour.passed"] == 2
    assert candidate.metrics["conversations.missing"] == 1
    assert candidate.metrics["replies.truncated"] == 1
    assert candidate.metrics["quality.invalid_replies"] == 1
    text, failed = comparison(baseline, candidate)
    assert failed
    assert "expected path" in text and "empty reply" in text
    assert "missing conversation" in text and "truncated reply" in text
    assert "RESOLVED" not in text


@pytest.mark.parametrize(
    "corruption", ["duplicate", "foreign", "altered_user", "short"]
)
def test_invalid_conversations_rejected(
    recording: tuple[Path, Path], corruption: str
) -> None:
    run, _ = recording
    rows = read_rows(run / "results.jsonl", SavedRunRow)
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "foreign":
        rows[0].case_id = "unexpected"
    elif corruption == "altered_user":
        rows[0].turns[0].user = "different"
    else:
        rows[0].turns = []
    write_rows(run / "results.jsonl", rows)
    with pytest.raises(ValueError):
        build_report(run)


@pytest.mark.parametrize("corruption", ["reply", "rubric", "duplicate", "input"])
def test_judge_linkage_rejected(recording: tuple[Path, Path], corruption: str) -> None:
    run, judge = recording
    if corruption == "reply":
        rows = read_rows(run / "results.jsonl", SavedRunRow)
        rows[0].turns[0].reply = "altered reply"
        write_rows(run / "results.jsonl", rows)
    elif corruption == "rubric":
        (judge / "rubric.md").write_text("different rubric")
    else:
        judgments = read_rows(judge / "results.jsonl", Judgment)
        if corruption == "duplicate":
            judgments.append(judgments[0])
        else:
            judgments[0].input.history = []
            judgments[0].input.candidate_reply = "altered input"
        write_rows(judge / "results.jsonl", judgments)
    with pytest.raises(ValueError):
        build_report(run, judge=judge)


def test_judge_errors_missing_and_repeated_hard_fails(
    recording: tuple[Path, Path],
) -> None:
    run, judge = recording
    baseline = build_report(run, judge=judge)
    rows = read_rows(judge / "results.jsonl", Judgment)
    for row in rows[:2]:
        row.verdict.hard_failures.clinical_overreach.failed = True
    error = JudgeErrorRow(
        case_id=rows[-1].case_id,
        rep=rows[-1].rep,
        turn_index=0,
        judge_rep=2,
        failure_class="timeout",
        attempts=[],
    )
    write_rows(judge / "results.jsonl", rows[:-2])
    write_rows(judge / "errors.jsonl", [error])
    candidate = build_report(run, judge=judge)
    assert candidate.metrics["judge.hard_fail_judgments"] == 2
    assert candidate.metrics["judge.hard_fail_replies"] == 1
    assert candidate.metrics["judge.errors"] == 1
    assert candidate.metrics["judge.missing"] == 1
    text, failed = comparison(baseline, candidate)
    assert failed and "clinical_overreach" in text


@pytest.mark.parametrize("field", ["suite_digest", "split", "reps"])
def test_incompatible_experiments_rejected(
    recording: tuple[Path, Path], field: str
) -> None:
    baseline = build_report(recording[0])
    candidate = baseline.model_copy(deep=True)
    setattr(candidate.meta, field, "changed")
    with pytest.raises(ValueError, match=field):
        comparison(baseline, candidate)


def test_model_changes_allowed_but_rubric_changes_rejected(
    recording: tuple[Path, Path],
) -> None:
    baseline = build_report(recording[0], judge=recording[1])
    candidate = baseline.model_copy(deep=True)
    candidate.meta.requested_model = "another-model"
    assert not comparison(baseline, candidate)[1]
    assert candidate.judge is not None
    candidate.judge.rubric_digest = "new-rubric"
    with pytest.raises(ValueError, match="rubric_digest"):
        comparison(baseline, candidate)


def test_cached_tokens_not_charged_twice_and_missing_rates_not_zero() -> None:
    prices = Prices(
        source="test fixture",
        as_of="2026-11-05",
        models={
            "model": TokenRates(input=2, cached_input=0.5, output=10),
        },
    )
    call = ModelCallRecord(
        served_model="model",
        input_tokens=1000,
        cache_read_tokens=400,
        output_tokens=100,
        reasoning_tokens=50,
    )
    metrics = coach_cost([call], 2, prices)
    assert metrics["cost.coach_estimate_usd"] == pytest.approx(0.0024)
    assert metrics["cost.coach_estimate_per_turn_usd"] == pytest.approx(0.0012)
    missing = coach_cost([call, ModelCallRecord(served_model="unknown")], 2, prices)
    assert missing["cost.coach_estimate_usd"] is None
    assert missing["cost.coach_known_estimate_usd"] == pytest.approx(0.0024)


def test_uncertainty_clusters_scenarios_not_repetitions() -> None:
    original = cluster_interval({"a": [0.0], "b": [1.0], "c": [1.0]})
    repeated = cluster_interval({"a": [0.0] * 100, "b": [1.0] * 100, "c": [1.0] * 100})
    assert original == repeated
    assert original is not None and original.low == 0 and original.high == 1
    assert cluster_interval({"a": [0.0], "b": [0.0]}) is None
    assert percentile([], 0.9) is None
    assert percentile([42], 0.9) == 42


@pytest.mark.parametrize("field,value", [("replies", 99), ("input_digest", "wrong")])
def test_judge_metadata_fields_checked_independently(
    recording: tuple[Path, Path], field: str, value: str | int
) -> None:
    run, judge = recording
    path = judge / "meta.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not belong"):
        build_report(run, judge=judge)


def test_prompt_provenance_uses_recorded_revision() -> None:
    assert prompt_digest("907f239") is not None
    assert prompt_digest("0000000") is None
    assert prompt_digest("HEAD") is None


@pytest.mark.parametrize("empty", [True, False])
def test_verdict_requires_successful_final_attempt(
    recording: tuple[Path, Path], empty: bool
) -> None:
    run, judge = recording
    rows = read_rows(judge / "results.jsonl", Judgment)
    if empty:
        rows[0].attempts = []
    else:
        failed = rows[0].attempts[0].model_copy(update={"status": "error"})
        rows[0].attempts.append(failed)
    write_rows(judge / "results.jsonl", rows)
    with pytest.raises(ValueError, match="successful final"):
        build_report(run, judge=judge)


def test_cost_rejects_invalid_cached_token_count() -> None:
    prices = Prices(
        source="fixture",
        as_of="2026-11-05",
        models={
            "test": TokenRates(input=1, cached_input=1, output=1),
        },
    )
    with pytest.raises(ValueError, match="Invalid token"):
        coach_cost(
            [ModelCallRecord(served_model="test", input_tokens=1, cache_read_tokens=2)],
            1,
            prices,
        )


def test_infrastructure_errors_are_not_behaviour_failures(
    recording: tuple[Path, Path],
) -> None:
    run, _ = recording
    rows = read_rows(run / "results.jsonl", SavedRunRow)
    write_rows(run / "results.jsonl", rows[:-1])
    write_rows(
        run / "errors.jsonl",
        [
            ErrorRow(
                case_id=rows[-1].case_id,
                rep=rows[-1].rep,
                failure_class="timeout",
                error="PRIVATE PROVIDER PAYLOAD",
            )
        ],
    )
    snapshot = build_report(run)
    assert snapshot.metrics["behaviour.pass_rate"] == 1
    assert snapshot.metrics["conversations.errors"] == 1
    assert snapshot.metrics["conversations.missing"] == 0
    assert "PRIVATE" not in snapshot.model_dump_json()
    assert comparison(snapshot, snapshot)[1]


def test_absent_timing_is_not_reported_as_zero(recording: tuple[Path, Path]) -> None:
    run, _ = recording
    path = run / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["turns"][0].pop("turn_ms")
        row["turns"][0]["model_calls"][0].pop("model_ms")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    snapshot = build_report(run)
    assert snapshot.metrics["completion.p50_ms"] is None
    assert snapshot.metrics["completion.missing"] == 4
    assert snapshot.metrics["model_request.p90_ms"] is None


def test_cli_writes_snapshot_and_compares_offline(
    recording: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    run, judge = recording
    snapshot = run.parent / "baseline.json"
    args = [str(run), "--judge", str(judge)]
    assert main(["report", *args, "--output", str(snapshot)]) == 0
    assert main(["compare", *args, "--baseline", str(snapshot)]) == 0
    assert "No changed conversation checks" in capsys.readouterr().out
    assert main(["compare", str(run), "--baseline", str(snapshot)]) == 2
    assert "judge recording presence" in capsys.readouterr().err
