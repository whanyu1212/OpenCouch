"""Run a multi-turn YAML suite with `pydantic-evals` or re-score offline (HY-239)."""

import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case as EvalCase
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, ReportEvaluator
from pydantic_evals.reporting import EvaluationReport

from evals.runner.cases import ConversationCase, Split, load_suite, suite_digest
from evals.runner.evaluators import (
    BehaviourEvaluator,
    NonEmptyReplyEvaluator,
    NotTruncatedEvaluator,
    RepPassRateReportEvaluator,
)
from evals.runner.execute import DEFAULT_TURN_TIMEOUT, EvalInfraError, play_case
from evals.runner.runs import (
    ErrorRow,
    SavedRunRow,
    SuiteRunMeta,
    load_meta_cases,
    load_run_meta,
    portable_path,
    read_error_rows,
    read_saved_runs,
    validate_run_rows,
    write_run_files,
)
from opencouch_agent.coach import CoachDeps
from opencouch_agent.safety import JevClient

RESULTS_ROOT: Final = Path(__file__).parents[1] / "results"
DEFAULT_TEST_COACH_REPLY: Final = "Let's look at that worry together."
DEFAULT_TEST_CRISIS_REPLY: Final = (
    "I'm really glad you told me. Please call SOS on 1767 or 995 if you are in "
    "immediate danger. Are you safe right now?"
)


@dataclass(frozen=True)
class SuiteRunOutcome:
    """What running or re-scoring a multi-turn suite produced."""

    report: EvaluationReport[ConversationCase, SavedRunRow, dict[str, Any]]
    results: list[SavedRunRow]
    errors: list[ErrorRow]
    run_dir: Path | None = None

    @property
    def all_passed(self) -> bool:
        """Whether every run passed all assertions with no infra errors."""
        if (
            self.errors
            or not self.report.cases
            or self.report.failures
            or self.report.report_evaluator_failures
        ):
            return False
        return all(
            not case.evaluator_failures
            and all(res.value is True for res in case.assertions.values())
            for case in self.report.cases
        )


def default_evaluators() -> list[
    Evaluator[ConversationCase, SavedRunRow, dict[str, Any]]
]:
    """Default deterministic evaluators for a multi-turn suite."""
    return [
        NonEmptyReplyEvaluator(),
        NotTruncatedEvaluator(),
        BehaviourEvaluator(),
    ]


def default_report_evaluators() -> list[
    ReportEvaluator[ConversationCase, SavedRunRow, dict[str, Any]]
]:
    """Default report-level evaluators."""
    return [RepPassRateReportEvaluator()]


def build_dataset(
    name: str,
    cases: Sequence[ConversationCase],
    *,
    evaluators: Sequence[Evaluator[ConversationCase, SavedRunRow, dict[str, Any]]]
    | None = None,
    report_evaluators: Sequence[
        ReportEvaluator[ConversationCase, SavedRunRow, dict[str, Any]]
    ]
    | None = None,
) -> Dataset[ConversationCase, SavedRunRow, dict[str, Any]]:
    """Build a `pydantic-evals` `Dataset` from `ConversationCase` items."""
    eval_cases = [
        EvalCase(
            name=case.id,
            inputs=case,
            metadata={"split": case.split, "tags": list(case.tags)},
        )
        for case in cases
    ]
    return Dataset(
        name=name,
        cases=eval_cases,
        evaluators=list(evaluators) if evaluators is not None else default_evaluators(),
        report_evaluators=(
            list(report_evaluators)
            if report_evaluators is not None
            else default_report_evaluators()
        ),
    )


async def run_suite(
    suite_path: Path,
    *,
    split: Split | None = None,
    requested_model: str = "test",
    coach_model: Model | str | None = None,
    crisis_model: Model | str | None = None,
    coach_agent: Agent[CoachDeps, str] | None = None,
    jev: JevClient | None = None,
    reps: int = 3,
    max_concurrency: int = 4,
    turn_timeout: timedelta = DEFAULT_TURN_TIMEOUT,
    output_dir: Path | None = None,
    save: bool = True,
    progress: bool = False,
) -> SuiteRunOutcome:
    """Run a YAML suite through `pydantic-evals` and save the run files.

    Infra errors (`EvalInfraError`) are recorded in `errors.jsonl` and excluded
    from `report.cases` so they never count as model quality/behaviour failures.
    """
    cases = load_suite(suite_path, split=split)
    dataset = build_dataset(suite_path.stem, cases)
    resolved_coach, resolved_crisis = _resolve_models(
        requested_model, coach_model=coach_model, crisis_model=crisis_model
    )

    rep_counters: dict[str, int] = defaultdict(int)
    results: list[SavedRunRow] = []
    errors: list[ErrorRow] = []

    async def task(case: ConversationCase) -> SavedRunRow:
        rep_counters[case.id] += 1
        rep = rep_counters[case.id]
        try:
            row = await play_case(
                case,
                rep=rep,
                requested_model=requested_model,
                coach_model=resolved_coach,
                crisis_model=resolved_crisis,
                coach_agent=coach_agent,
                jev=jev,
                turn_timeout=turn_timeout,
            )
        except EvalInfraError as error:
            errors.append(
                ErrorRow(
                    case_id=case.id,
                    rep=rep,
                    turn_index=error.turn_index,
                    failure_class=error.failure_class,
                    error=error.message,
                )
            )
            raise
        results.append(row)
        return row

    report = await dataset.evaluate(
        task,
        name=f"{suite_path.stem}-{requested_model}",
        max_concurrency=max_concurrency,
        repeat=reps,
        progress=progress,
    )
    results.sort(key=lambda row: (row.case_id, row.rep))
    errors.sort(key=lambda row: (row.case_id, row.rep))

    run_dir: Path | None = None
    if save:
        run_dir = output_dir or new_suite_run_dir(suite_path.stem, split)
        served = sorted({m for row in results for m in row.served_models})
        meta = SuiteRunMeta(
            suite=suite_path.stem,
            suite_path=portable_path(suite_path),
            suite_digest=suite_digest(cases),
            split=split,
            requested_model=requested_model,
            served_models=served,
            reps=reps,
            started_at=datetime.now(UTC).isoformat(),
            git_sha=_current_git_sha(),
        )
        rendered = report.render(include_reasons=True)
        write_run_files(run_dir, meta, results, errors, rendered)
    return SuiteRunOutcome(
        report=report, results=results, errors=errors, run_dir=run_dir
    )


async def evaluate_saved_run(
    run_dir: Path,
    *,
    suite_path: Path | None = None,
    evaluators: Sequence[Evaluator[ConversationCase, SavedRunRow, dict[str, Any]]]
    | None = None,
    progress: bool = False,
) -> SuiteRunOutcome:
    """Re-score a saved `run_dir` offline through `pydantic-evals` without model calls.

    Raises:
        ValueError: If the suite file changed since `run_dir` was recorded, or
            if a saved row has no matching case in the suite.
    """
    meta = load_run_meta(run_dir)
    cases = load_meta_cases(meta, run_dir, suite_path=suite_path)
    saved_rows = read_saved_runs(run_dir)
    saved_errors = read_error_rows(run_dir)
    validate_run_rows(meta, cases, saved_rows, saved_errors)

    by_case: dict[str, list[SavedRunRow]] = defaultdict(list)
    for row in sorted(saved_rows, key=lambda item: (item.case_id, item.rep)):
        by_case[row.case_id].append(row)

    present_cases = [case for case in cases if case.id in by_case]
    dataset = build_dataset(meta.suite, present_cases, evaluators=evaluators)
    # Schedule only recorded repetitions, before report-level statistics run.
    dataset.cases = [
        EvalCase(
            name=(f"{case.id} [{row.rep}/{meta.reps}]" if meta.reps > 1 else case.id),
            inputs=case,
            metadata={"split": case.split, "tags": list(case.tags)},
        )
        for case in present_cases
        for row in by_case[case.id]
    ]
    cursors: dict[str, int] = defaultdict(int)

    async def replay_task(case: ConversationCase) -> SavedRunRow:
        rows = by_case[case.id]
        idx = cursors[case.id]
        cursors[case.id] = idx + 1
        return rows[idx]

    report = await dataset.evaluate(
        replay_task,
        name=f"{meta.suite}-{meta.requested_model}-rescore",
        progress=progress,
    )
    if meta.reps > 1:
        for report_case in report.cases:
            report_case.source_case_name = report_case.output.case_id
    return SuiteRunOutcome(
        report=report,
        results=saved_rows,
        errors=saved_errors,
        run_dir=run_dir,
    )


def _resolve_models(
    requested_model: str,
    *,
    coach_model: Model | str | None,
    crisis_model: Model | str | None,
) -> tuple[Model | str, Model | str]:
    if coach_model is not None or requested_model != "test":
        coach = coach_model or requested_model
        crisis = crisis_model or coach
        return coach, crisis
    return (
        TestModel(custom_output_text=DEFAULT_TEST_COACH_REPLY, call_tools=[]),
        TestModel(custom_output_text=DEFAULT_TEST_CRISIS_REPLY, call_tools=[]),
    )


def new_suite_run_dir(suite: str, split: Split | None) -> Path:
    """Build a timestamped run directory path under `evals/results/`."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = _current_git_sha() or "nosha"
    split_tag = split or "all"
    return RESULTS_ROOT / f"{suite}-{split_tag}-{sha}-{stamp}"


def _current_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


__all__ = [
    "DEFAULT_TEST_COACH_REPLY",
    "DEFAULT_TEST_CRISIS_REPLY",
    "RESULTS_ROOT",
    "SuiteRunOutcome",
    "build_dataset",
    "default_evaluators",
    "default_report_evaluators",
    "evaluate_saved_run",
    "new_suite_run_dir",
    "run_suite",
]
