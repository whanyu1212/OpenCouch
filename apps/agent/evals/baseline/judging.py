"""Validate judge-to-reply linkage before aggregating any verdicts."""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from evals.baseline.models import JudgeMeta
from evals.judge.jobs import JudgeJob
from evals.judge.models import JudgeErrorRow, Judgment
from evals.judge.run import job_digest


def read_rows[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Read a required JSONL file, validating each nonempty row."""
    return [
        model.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_judgments(
    directory: Path, jobs: list[JudgeJob]
) -> tuple[JudgeMeta, list[Judgment], list[JudgeErrorRow], int]:
    """Reject mismatched, duplicate or foreign judgments; count absent jobs."""
    meta = JudgeMeta.model_validate_json((directory / "meta.json").read_text())
    digest = hashlib.sha256((directory / "rubric.md").read_bytes()).hexdigest()
    if digest != meta.rubric_digest:
        raise ValueError("Saved judge rubric does not match its digest")
    if meta.input_digest != job_digest(jobs) or meta.replies != len(jobs):
        raise ValueError("Judge recording does not belong to these coach replies")
    judgments = read_rows(directory / "results.jsonl", Judgment)
    errors = read_rows(directory / "errors.jsonl", JudgeErrorRow)
    lookup = {(job.case_id, job.rep, job.turn_index): job.input for job in jobs}
    expected = {(*key, rep) for key in lookup for rep in range(1, meta.judge_reps + 1)}
    seen: set[tuple[str, int, int, int]] = set()
    rows: list[Judgment | JudgeErrorRow] = [*judgments, *errors]
    for row in rows:
        key = (row.case_id, row.rep, row.turn_index)
        identity = (*key, row.judge_rep)
        if identity not in expected or identity in seen:
            raise ValueError(f"Unexpected or duplicate judge result: {identity}")
        seen.add(identity)
        _validate_result(row, lookup[key])
    return meta, judgments, errors, len(expected - seen)


def _validate_result(row: Judgment | JudgeErrorRow, expected: BaseModel) -> None:
    if isinstance(row, Judgment):
        if row.input != expected:
            raise ValueError(f"Altered judge input: {row.case_id}:{row.turn_index}")
        if not row.attempts or row.attempts[-1].status != "ok":
            raise ValueError("A verdict must have a successful final judge attempt")
    elif row.mode != "quality":
        raise ValueError("Pairwise judge errors cannot be scored as scalar judgments")
