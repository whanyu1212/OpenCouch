"""Saved runs must be complete and retain their original evaluation context."""

from pathlib import Path
from typing import Any

import pytest
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals.judge.jobs import load_jobs
from evals.runner.cases import ConversationCase
from evals.runner.runs import ErrorRow, SavedRunRow, load_run_meta, write_run_files
from evals.runner.suite import evaluate_saved_run, run_suite

SUITE = Path(__file__).parents[2] / "evals/cases/coach.yaml"


class BrokenEvaluator(Evaluator[ConversationCase, SavedRunRow, dict[str, Any]]):
    def evaluate(
        self, ctx: EvaluatorContext[ConversationCase, SavedRunRow, dict[str, Any]]
    ) -> bool:
        raise ValueError("evaluation unavailable")


async def test_evaluator_errors_cannot_report_success(tmp_path: Path) -> None:
    await run_suite(SUITE, split="dev", reps=1, output_dir=tmp_path)
    outcome = await evaluate_saved_run(tmp_path, evaluators=[BrokenEvaluator()])
    assert all(case.evaluator_failures for case in outcome.report.cases)
    assert not outcome.all_passed
    assert "Pass rate (0/0" in outcome.report.render()
    assert "5 unscorable" in outcome.report.render()


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "unknown", "out_of_range", "overlap"]
)
async def test_invalid_recording_is_rejected(tmp_path: Path, damage: str) -> None:
    live = await run_suite(SUITE, split="dev", reps=2, output_dir=tmp_path)
    rows = live.results
    errors: list[ErrorRow] = []
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "unknown":
        rows[0].case_id = "unknown-case"
    elif damage == "out_of_range":
        rows[0].rep = 3
    else:
        errors.append(
            ErrorRow(
                case_id=rows[0].case_id,
                rep=rows[0].rep,
                failure_class="timeout",
                error="timed out",
            )
        )
    write_run_files(tmp_path, load_run_meta(tmp_path), rows, errors, "")
    with pytest.raises(ValueError, match="saved run"):
        await evaluate_saved_run(tmp_path)
    with pytest.raises(ValueError, match="saved run"):
        load_jobs(tmp_path)


@pytest.mark.parametrize("changed", [False, True])
async def test_alternate_suite_must_match_recorded_digest(
    tmp_path: Path, changed: bool
) -> None:
    run_dir = tmp_path / "run"
    await run_suite(SUITE, split="dev", reps=1, output_dir=run_dir)
    alternate = tmp_path / "suite.yaml"
    text = SUITE.read_text()
    if changed:
        text = text.replace("Work has been really stressful", "Changed user message")
    alternate.write_text(text)
    if changed:
        with pytest.raises(ValueError, match="changed since"):
            await evaluate_saved_run(run_dir, suite_path=alternate)
    else:
        assert (await evaluate_saved_run(run_dir, suite_path=alternate)).all_passed
