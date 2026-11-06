"""Human confirmation, criterion boundaries and saved-input integrity."""

import hashlib
import json
from pathlib import Path

import pytest

from evals.calibration.__main__ import main
from evals.calibration.agreement import agreement
from evals.calibration.models import CalibrationItem, CalibrationSet
from evals.calibration.prepare import prepare
from evals.judge.models import (
    HardFailures,
    JudgeAttempt,
    JudgeErrorRow,
    JudgeInput,
    Judgment,
    QualityScores,
    QualityVerdict,
)
from evals.judge.run import job_digest
from evals.runner.cases import load_suite, suite_digest
from evals.runner.runs import SavedRunRow, SuiteRunMeta, TurnRecord, write_run_files


def verdict(failed: bool = False, score: int = 3) -> QualityVerdict:
    return QualityVerdict.model_validate(
        {
            "hard_failures": {
                name: {"failed": failed, "reason": "Test evidence"}
                for name in HardFailures.model_fields
            },
            "scores": {
                name: {"score": score, "reason": "Test evidence"}
                for name in QualityScores.model_fields
            },
        }
    )


@pytest.fixture
def labels() -> CalibrationSet:
    return CalibrationSet(
        rubric_digest=hashlib.sha256(b"Test rubric").hexdigest(),
        labeller="test",
        source_run="synthetic-dev",
        items=[
            CalibrationItem(
                id=str(index),
                source_case_id=str(index),
                source_rep=1,
                source_turn_index=0,
                kind="control",
                tags=["test"],
                input=JudgeInput(user="A synthetic worry", candidate_reply=str(index)),
                verdict=verdict(index == 0),
                confirmed_by="Wu Hanyu",
            )
            for index in range(10)
        ],
    )


def save_judge(
    directory: Path, labels: CalibrationSet, verdicts: list[QualityVerdict]
) -> None:
    jobs = labels.jobs()
    (directory / "rubric.md").write_text("Test rubric")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "mode": "quality",
                "rubric_digest": labels.rubric_digest,
                "schema_digest": "test",
                "input_digest": job_digest(jobs),
                "judge_reps": 1,
                "replies": len(jobs),
                "routes": [],
                "calibrated": False,
            }
        )
    )
    rows = [
        Judgment(
            case_id=job.case_id,
            rep=1,
            turn_index=0,
            judge_rep=1,
            input=job.input,
            verdict=prediction,
            attempts=[JudgeAttempt(route="anthropic", requested_model="test")],
        )
        for job, prediction in zip(jobs, verdicts, strict=False)
    ]
    (directory / "results.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows)
    )
    (directory / "errors.jsonl").write_text("")


def test_unconfirmed_labels_never_enter_denominator(
    tmp_path: Path, labels: CalibrationSet
) -> None:
    save_judge(tmp_path, labels, [verdict()] * 10)
    for item in labels.items:
        item.confirmed_by = None
    empty = agreement(labels, tmp_path)
    assert empty.hard_failures["enables_harm"].rate is None
    assert empty.confirmed_items == 0
    assert not empty.meets_target
    labels.items[0].confirmed_by = "Wu Hanyu"
    partial = agreement(labels, tmp_path)
    assert partial.hard_failures["enables_harm"].total == 1
    assert partial.hard_failures["enables_harm"].matched == 0
    assert partial.disagreements
    assert not partial.meets_target


@pytest.mark.parametrize(("mismatches", "passes"), [(0, True), (1, True), (2, False)])
def test_hard_fail_threshold_per_criterion(
    tmp_path: Path, labels: CalibrationSet, mismatches: int, passes: bool
) -> None:
    predictions = [verdict(index == 0) for index in range(10)]
    for prediction in predictions[1 : mismatches + 1]:
        prediction.hard_failures.false_identity.failed = True
    save_judge(tmp_path, labels, predictions)
    report = agreement(labels, tmp_path)
    assert report.meets_target is passes
    assert report.hard_failures["false_identity"].matched == 10 - mismatches
    assert report.hard_failures["false_identity"].human_positive == 1
    assert report.hard_failures["false_identity"].human_negative == 9
    assert report.advisory


@pytest.mark.parametrize(("score", "matches"), [(1, 0), (2, 10), (4, 10), (5, 0)])
def test_score_agreement_is_within_one(
    tmp_path: Path, labels: CalibrationSet, score: int, matches: int
) -> None:
    save_judge(tmp_path, labels, [verdict(index == 0, score) for index in range(10)])
    report = agreement(labels, tmp_path)
    assert report.scores["attunement"].matched == matches
    assert report.scores["attunement"].total == 10
    assert report.model_dump()["scores"]["attunement"]["rate"] == matches / 10


def test_incomplete_judging_cannot_pass(tmp_path: Path, labels: CalibrationSet) -> None:
    save_judge(tmp_path, labels, [verdict(True)])
    report = agreement(labels, tmp_path)
    assert report.missing_judgments == 9
    assert report.judged_confirmed_items == 1
    assert not report.meets_target


def test_absent_positive_controls_cannot_pass(
    tmp_path: Path, labels: CalibrationSet
) -> None:
    for item in labels.items:
        item.verdict = verdict()
    save_judge(tmp_path, labels, [verdict()] * 10)
    report = agreement(labels, tmp_path)
    assert report.hard_failures["enables_harm"].rate == 1
    assert not report.meets_target


@pytest.mark.parametrize("alteration", ["input", "rubric", "duplicate"])
def test_rejects_stale_or_duplicate_judgments(
    tmp_path: Path, labels: CalibrationSet, alteration: str
) -> None:
    save_judge(tmp_path, labels, [verdict()] * 10)
    if alteration == "input":
        labels.items[0].input.candidate_reply = "Changed since judgment"
    elif alteration == "rubric":
        labels.rubric_digest = "another rubric"
    else:
        results = tmp_path / "results.jsonl"
        results.write_text(results.read_text() + results.read_text().splitlines()[0])
    with pytest.raises(ValueError):
        agreement(labels, tmp_path)


async def test_paid_judging_requires_blind_labels(
    tmp_path: Path, labels: CalibrationSet
) -> None:
    labels.items[0].verdict = None
    labels.items[0].confirmed_by = None
    path = tmp_path / "labels.yaml"
    path.write_text(labels.model_dump_json())
    with pytest.raises(SystemExit, match="2"):
        await main([str(path), "--judge", str(tmp_path / "paid")])
    assert not (tmp_path / "paid").exists()


def test_confirmation_requires_label_and_known_human(labels: CalibrationSet) -> None:
    item = labels.items[0].model_dump()
    item["verdict"] = None
    with pytest.raises(ValueError, match="no verdict"):
        CalibrationItem.model_validate(item)
    item["verdict"] = verdict().model_dump()
    item["confirmed_by"] = "agent"
    with pytest.raises(ValueError):
        CalibrationItem.model_validate(item)


def test_duplicate_label_ids_rejected(labels: CalibrationSet) -> None:
    payload = labels.model_dump()
    payload["items"].append(payload["items"][0])
    with pytest.raises(ValueError, match="unique"):
        CalibrationSet.model_validate(payload)


@pytest.mark.parametrize("split", ["dev", "holdout"])
def test_selection_is_blind_deterministic_and_dev_only(
    tmp_path: Path, split: str
) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        json.dumps(
            [
                {
                    "id": name,
                    "split": split,
                    "tags": [name],
                    "turns": [{"user": "Synthetic worry"}],
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
            split=case.split,
            requested_model="test",
            turns=[
                TurnRecord(
                    turn_index=0,
                    user="Synthetic worry",
                    reply=f"Reply {rep}",
                    path="coach",
                    risk_level="NONE",
                )
            ],
        )
        for case in cases
        for rep in range(1, 16)
    ]
    run = tmp_path / "recording"
    write_run_files(
        run,
        SuiteRunMeta(
            suite="calibration-test",
            suite_path=str(suite),
            suite_digest=suite_digest(cases),
            split=cases[0].split,
            requested_model="test",
            reps=15,
            started_at="2026-11-06",
        ),
        rows,
        [],
        "",
    )
    if split == "holdout":
        with pytest.raises(ValueError, match="dev split"):
            prepare(run)
        return
    packet = prepare(run)
    assert packet == prepare(run)
    assert len(packet.items) == 40
    assert {item.source_case_id for item in packet.items[:2]} == {"work", "sleep"}
    assert all(
        item.verdict is None and item.confirmed_by is None for item in packet.items
    )
    assert sum(item.kind == "control" for item in packet.items) == 10


async def test_report_exit_tracks_review_completeness(
    tmp_path: Path, labels: CalibrationSet
) -> None:
    save_judge(tmp_path, labels, [verdict(index == 0) for index in range(10)])
    path = tmp_path / "labels.yaml"
    path.write_text(labels.model_dump_json())
    assert await main([str(path), "--report", str(tmp_path)]) == 0
    labels.items[0].confirmed_by = None
    path.write_text(labels.model_dump_json())
    assert await main([str(path), "--report", str(tmp_path)]) == 1


@pytest.mark.parametrize("incomplete", ["review", "missing", "error"])
def test_perfect_observed_agreement_still_requires_complete_review_and_judging(
    tmp_path: Path, labels: CalibrationSet, incomplete: str
) -> None:
    predictions = [verdict(index == 0) for index in range(10)]
    save_judge(
        tmp_path, labels, predictions if incomplete == "review" else predictions[:9]
    )
    if incomplete == "review":
        labels.items[-1].confirmed_by = None
    elif incomplete == "error":
        error = JudgeErrorRow(
            case_id="9",
            rep=1,
            turn_index=0,
            judge_rep=1,
            failure_class="TimeoutError",
            attempts=[],
        )
        (tmp_path / "errors.jsonl").write_text(error.model_dump_json() + "\n")
    report = agreement(labels, tmp_path)
    assert report.hard_failures["enables_harm"].rate == 1
    assert not report.meets_target
    if incomplete == "error":
        assert report.judge_errors == 1
        assert report.missing_judgments == 0


@pytest.mark.parametrize("negative_count", [0, 1])
def test_negative_control_coverage(
    tmp_path: Path, labels: CalibrationSet, negative_count: int
) -> None:
    predictions = [verdict(index >= negative_count) for index in range(10)]
    for item, prediction in zip(labels.items, predictions, strict=True):
        item.verdict = prediction
    save_judge(tmp_path, labels, predictions)
    report = agreement(labels, tmp_path)
    assert report.hard_failures["false_identity"].human_negative == negative_count
    assert report.meets_target is (negative_count == 1)
