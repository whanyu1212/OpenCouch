"""Create deterministic baseline summaries from saved synthetic runs."""

import hashlib
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from evals.baseline.cost import coach_cost
from evals.baseline.judging import load_judgments, read_rows
from evals.baseline.models import Outcome, Prices, Snapshot
from evals.baseline.statistics import cluster_interval, timing_metrics
from evals.judge.jobs import JudgeJob, row_jobs
from evals.judge.models import JudgeErrorRow, Judgment, QualityScores
from evals.runner import evaluators
from evals.runner.evaluators import check_turn_expectation
from evals.runner.runs import (
    ErrorRow,
    SavedRunRow,
    load_meta_cases,
    load_run_meta,
)

AGENT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILES = (
    "src/opencouch_agent/coach/coach_prompt.md",
    "src/opencouch_agent/coach/instructions.py",
    "src/opencouch_agent/safety/crisis_prompt.md",
    "src/opencouch_agent/safety/crisis.py",
)
NOTES = [
    "Synthetic scenarios are the sampling units; repetitions are not independent.",
    "Intervals: exploratory 95% whole-scenario bootstrap (2,000 draws, seed 241). "
    "Null means too few clusters or constant cluster means, not zero uncertainty.",
    "Eight scenarios cannot establish small quality improvements. Investigate any "
    "new behavioural failure; do not gate on small score movements.",
    "Judge scores are advisory pending human calibration (HY-242).",
    "model_request timing includes the wrapped request/stream and possible retries; "
    "final-attempt-only timing is unavailable. Retry counts are SDK-reported only.",
    "Coach costs, if rates are supplied, estimate recorded successful calls only; "
    "failed attempts and Jev charges are unavailable. Reasoning is part of output.",
    "Prompt digest identifies source at the recorded git revision, not captured "
    "runtime instructions; dirty-tree and environment overrides were not recorded.",
]


def prompt_digest(revision: str | None) -> str | None:
    """Hash prompt sources at the recorded commit, never today's working tree."""
    if revision is None or not re.fullmatch(r"[a-fA-F0-9]{7,40}", revision):
        return None
    digest = hashlib.sha256()
    for name in PROMPT_FILES:
        result = subprocess.run(
            ["git", "show", f"{revision}:apps/agent/{name}"],
            cwd=AGENT_ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            return None
        digest.update(name.encode() + b"\0" + result.stdout)
    return digest.hexdigest()


def build_report(
    directory: Path,
    *,
    judge: Path | None = None,
    suite: Path | None = None,
    prices: Prices | None = None,
) -> Snapshot:
    """Validate recordings and produce a transcript-free report without API calls."""
    meta = load_run_meta(directory)
    cases = load_meta_cases(meta, directory, suite_path=suite)
    if not cases:
        raise ValueError("Cannot report an empty suite split")
    results = read_rows(directory / "results.jsonl", SavedRunRow)
    errors = read_rows(directory / "errors.jsonl", ErrorRow)
    expected = {(case.id, rep) for case in cases for rep in range(1, meta.reps + 1)}
    outcomes = _outcomes(results, errors, expected)
    jobs: list[JudgeJob] = []
    by_id = {case.id: case for case in cases}
    for row in results:
        if (
            row.split != by_id[row.case_id].split
            or row.requested_model != meta.requested_model
        ):
            raise ValueError(f"Run metadata mismatch: {row.case_id}")
        jobs.extend(row_jobs(row, by_id[row.case_id]))
        outcome = outcomes[(row.case_id, row.rep)]
        for turn, script in zip(row.turns, by_id[row.case_id].turns, strict=True):
            outcome.issues.extend(check_turn_expectation(turn, script.expect))
            if not turn.reply.strip():
                outcome.issues.append(f"turn {turn.turn_index + 1}: empty reply")
        if row.status == "truncated" or any(turn.truncated for turn in row.turns):
            outcome.issues.append("truncated reply")
    ordered = [outcomes[key] for key in sorted(outcomes)]
    groups: dict[str, list[float]] = defaultdict(list)
    for outcome in ordered:
        if outcome.status == "complete":
            groups[outcome.case_id].append(float(not outcome.issues))
    snapshot = Snapshot(
        run=directory.name,
        meta=meta,
        evaluator_digest=hashlib.sha256(
            Path(evaluators.__file__).read_bytes()
        ).hexdigest(),
        prompt_digest=prompt_digest(meta.git_sha),
        judge=None,
        judge_models=[],
        prices=prices,
        outcomes=ordered,
        metrics=_metrics(results, ordered, prices),
        intervals={"behaviour.pass_rate": cluster_interval(groups)},
        notes=list(NOTES),
    )
    snapshot.metrics["scenarios"] = len(cases)
    snapshot.metrics["one_scenario_share"] = 1 / len(cases)
    if judge is not None:
        _add_judging(snapshot, judge, jobs)
    else:
        snapshot.notes.append("No judge recording supplied; quality is unmeasured.")
    return snapshot


def _outcomes(
    results: list[SavedRunRow], errors: list[ErrorRow], expected: set[tuple[str, int]]
) -> dict[tuple[str, int], Outcome]:
    outcomes: dict[tuple[str, int], Outcome] = {}
    rows: list[SavedRunRow | ErrorRow] = [*results, *errors]
    for row in rows:
        key = (row.case_id, row.rep)
        if key not in expected or key in outcomes:
            raise ValueError(f"Unexpected or duplicate conversation: {key}")
        error = isinstance(row, ErrorRow)
        outcomes[key] = Outcome(
            case_id=row.case_id,
            rep=row.rep,
            status="error" if error else "complete",
            issues=[f"infrastructure: {row.failure_class}"]
            if isinstance(row, ErrorRow)
            else [],
        )
    for case_id, rep in expected - outcomes.keys():
        outcomes[(case_id, rep)] = Outcome(
            case_id=case_id, rep=rep, status="missing", issues=["missing conversation"]
        )
    return outcomes


def _metrics(
    rows: list[SavedRunRow], outcomes: list[Outcome], prices: Prices | None
) -> dict[str, float | None]:
    turns = [turn for row in rows for turn in row.turns]
    calls = [call for turn in turns for call in turn.model_calls]
    passed = sum(not row.issues for row in outcomes)
    metrics: dict[str, float | None] = {
        "conversations.scheduled": len(outcomes),
        "conversations.complete": len(rows),
        "conversations.errors": sum(row.status == "error" for row in outcomes),
        "conversations.missing": sum(row.status == "missing" for row in outcomes),
        "behaviour.passed": passed,
        "behaviour.pass_rate": passed / len(rows) if rows else None,
        "replies": len(turns),
        "replies.truncated": sum(t.truncated for t in turns),
        "replies.empty": sum(not t.reply.strip() for t in turns),
        "quality.invalid_replies": sum(
            t.truncated or not t.reply.strip() for t in turns
        ),
        "model.calls": len(calls),
        "model.reported_retries": sum(c.retries for c in calls),
        "model.retry_records_missing": sum(
            "retries" not in c.model_fields_set for c in calls
        ),
        "model.input_tokens": sum(c.input_tokens for c in calls),
        "model.cached_input_tokens": sum(c.cache_read_tokens for c in calls),
        "model.output_tokens": sum(c.output_tokens for c in calls),
        "cost.jev_usd": None,
        "model.final_attempt_p50_ms": None,
        "model.final_attempt_p90_ms": None,
    }
    for name, values, expected in (
        (
            "first_token",
            [float(t.first_token_ms) for t in turns if t.first_token_ms is not None],
            len(turns),
        ),
        (
            "completion",
            [float(t.turn_ms) for t in turns if "turn_ms" in t.model_fields_set],
            len(turns),
        ),
        (
            "model_request",
            [float(c.model_ms) for c in calls if "model_ms" in c.model_fields_set],
            len(calls),
        ),
    ):
        metrics.update(timing_metrics(name, values, expected))
    metrics.update(coach_cost(calls, len(turns), prices))
    metrics["screening.turns_without_trace"] = sum(t.risk_trace is None for t in turns)
    for source in ("live", "scripted", "fake"):
        metrics[f"screening.{source}_calls"] = sum(
            call.source == source
            for turn in turns
            if turn.risk_trace
            for call in turn.risk_trace.calls
        )
    return metrics


def _add_judging(snapshot: Snapshot, directory: Path, jobs: list[JudgeJob]) -> None:
    meta, judgments, errors, missing = load_judgments(directory, jobs)
    snapshot.judge = meta
    snapshot.judge_models = sorted(
        {
            f"{row.attempts[-1].route}:{row.attempts[-1].served_model}"
            for row in judgments
        }
    )
    judge_rows: list[Judgment | JudgeErrorRow] = [*judgments, *errors]
    attempts = [attempt for row in judge_rows for attempt in row.attempts]
    failed = [row for row in judgments if row.verdict.hard_failures.any_failed]
    snapshot.judge_failures = sorted(
        {
            f"{row.case_id} rep {row.rep} turn {row.turn_index + 1}: {name}"
            for row in failed
            for name in type(row.verdict.hard_failures).model_fields
            if getattr(row.verdict.hard_failures, name).failed
        }
    )
    snapshot.metrics.update(
        {
            "judge.judgments": len(judgments),
            "judge.errors": len(errors),
            "judge.missing": missing,
            "judge.hard_fail_judgments": len(failed),
            "judge.hard_fail_replies": len(
                {(r.case_id, r.rep, r.turn_index) for r in failed}
            ),
            "judge.hard_fail_conversations": len({(r.case_id, r.rep) for r in failed}),
            "judge.attempts_without_cost": sum(a.cost_usd is None for a in attempts),
            "cost.judge_known_usd": sum(a.cost_usd or 0 for a in attempts),
        }
    )
    for name in ["hard_fail_rate", *QualityScores.model_fields]:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in judgments:
            value = (
                float(row.verdict.hard_failures.any_failed)
                if name == "hard_fail_rate"
                else float(getattr(row.verdict.scores, name).score)
            )
            groups[row.case_id].append(value)
        values = [value for group in groups.values() for value in group]
        key = f"judge.{name}"
        snapshot.metrics[key] = sum(values) / len(values) if values else None
        snapshot.intervals[key] = cluster_interval(groups)
