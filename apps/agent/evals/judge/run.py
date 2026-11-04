"""Save bounded judge requests separately from the coach's immutable recording."""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel

from evals.judge.assess import assess_reply, compare_replies, load_rubric, rubric_digest
from evals.judge.client import JudgeClient, JudgeError
from evals.judge.jobs import JudgeJob
from evals.judge.models import (
    JudgeErrorRow,
    Judgment,
    PairwiseJudgment,
    PairwiseVerdict,
    QualityVerdict,
)
from evals.judge.summary import summarize


@dataclass(frozen=True)
class JudgeOutcome:
    """Saved judgments and unscorable requests from a judge run."""

    judgments: list[Judgment]
    comparisons: list[PairwiseJudgment]
    errors: list[JudgeErrorRow]
    output_dir: Path


class _Identity(TypedDict):
    case_id: str
    rep: int
    turn_index: int
    judge_rep: int


def _append(path: Path, row: BaseModel) -> None:
    # Each append is synchronous within one event-loop task; partial runs stay readable.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(row.model_dump_json() + "\n")


def _job_digest(jobs: Sequence[JudgeJob]) -> str:
    data = [
        {
            "case_id": job.case_id,
            "rep": job.rep,
            "turn_index": job.turn_index,
            "input": job.input.model_dump(mode="json"),
        }
        for job in jobs
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _prepare_output(
    output_dir: Path,
    jobs: Sequence[JudgeJob],
    reps: int,
    client: JudgeClient,
    alternatives: Sequence[JudgeJob] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "rubric.md").write_text(load_rubric(), encoding="utf-8")
    for name in ("results.jsonl", "errors.jsonl", "pairs.jsonl"):
        (output_dir / name).touch()
    metadata = {
        "rubric_digest": rubric_digest(),
        "input_digest": _job_digest(jobs),
        "comparison_input_digest": _job_digest(alternatives)
        if alternatives is not None
        else None,
        "schema_digest": hashlib.sha256(
            json.dumps(
                (
                    PairwiseVerdict if alternatives is not None else QualityVerdict
                ).model_json_schema(),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "routes": [
            {
                "route": route.route,
                "requested_model": route.requested_model,
                "effort": route.settings.get(
                    "anthropic_effort", route.settings.get("openai_reasoning_effort")
                ),
                "max_tokens": route.settings.get("max_tokens"),
            }
            for route in client.routes
        ],
        "judge_reps": reps,
        "mode": "pairwise" if alternatives is not None else "quality",
        "replies": len(jobs),
        "calibrated": False,
    }
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


async def judge_jobs(
    jobs: Sequence[JudgeJob],
    client: JudgeClient,
    output_dir: Path,
    *,
    reps: int = 1,
    concurrency: int = 4,
    alternatives: Sequence[JudgeJob] | None = None,
) -> JudgeOutcome:
    """Judge saved replies or compare both blind orderings; persist every result."""
    if reps < 1 or concurrency < 1:
        raise ValueError("reps and concurrency must be positive")
    comparisons = _align_comparisons(jobs, alternatives)
    _prepare_output(output_dir, jobs, reps, client, alternatives)
    outcome = JudgeOutcome([], [], [], output_dir)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(job: JudgeJob, rep: int, reverse: bool | None) -> None:
        async with semaphore:
            await _judge_one(job, rep, reverse, comparisons, client, outcome)

    async with asyncio.TaskGroup() as group:
        for job in jobs:
            for rep in range(1, reps + 1):
                orders = (False, True) if alternatives is not None else (None,)
                for reverse in orders:
                    group.create_task(run_one(job, rep, reverse))
    summary = summarize(outcome.judgments, outcome.comparisons, outcome.errors)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return outcome


def _align_comparisons(
    jobs: Sequence[JudgeJob], alternatives: Sequence[JudgeJob] | None
) -> dict[tuple[str, int, int], JudgeJob]:
    if alternatives is None:
        return {}
    lookup = {(job.case_id, job.rep, job.turn_index): job for job in alternatives}
    for job in jobs:
        other = lookup.get((job.case_id, job.rep, job.turn_index))
        if other is None or other.input.user != job.input.user:
            raise ValueError("comparison runs must contain matching scripted turns")
    return lookup


async def _judge_one(
    job: JudgeJob,
    rep: int,
    reverse: bool | None,
    comparisons: dict[tuple[str, int, int], JudgeJob],
    client: JudgeClient,
    outcome: JudgeOutcome,
) -> None:
    identity: _Identity = {
        "case_id": job.case_id,
        "rep": job.rep,
        "turn_index": job.turn_index,
        "judge_rep": rep,
    }
    try:
        if reverse is None:
            verdict, attempts = await assess_reply(client, job.input)
            row = Judgment(
                **identity, input=job.input, verdict=verdict, attempts=attempts
            )
            outcome.judgments.append(row)
            _append(outcome.output_dir / "results.jsonl", row)
        else:
            other = comparisons[(job.case_id, job.rep, job.turn_index)]
            pair, attempts = await compare_replies(
                client, job.input, other.input, reverse=reverse
            )
            preference: dict[str, Literal["first", "second", "tie"]] = {
                "A": "second" if reverse else "first",
                "B": "first" if reverse else "second",
                "tie": "tie",
            }
            pair_row = PairwiseJudgment(
                **identity,
                reversed=reverse,
                input=job.input,
                alternative=other.input,
                verdict=pair,
                preferred_run=preference[pair.winner],
                attempts=attempts,
            )
            outcome.comparisons.append(pair_row)
            _append(outcome.output_dir / "pairs.jsonl", pair_row)
    except JudgeError as error:
        failed = JudgeErrorRow(
            **identity,
            failure_class=error.failure_class,
            attempts=error.attempts,
            mode="quality" if reverse is None else "pairwise",
            reversed=reverse,
        )
        outcome.errors.append(failed)
        _append(outcome.output_dir / "errors.jsonl", failed)
