"""Saved-run files (`results.jsonl`, `errors.jsonl`, `meta.json`) for HY-239.

Every live or test-model suite run writes a directory under `evals/results/`:

- `meta.json`: `SuiteRunMeta` describing the suite, split, digest, requested
  model, and repeat count.
- `results.jsonl`: one `SavedRunRow` per completed `case x rep` with the full
  per-turn transcript, path, risk level, state, tool calls, timings, token
  usage per model call, served model, and `status` (`"ok"` or `"truncated"`).
- `errors.jsonl`: one `ErrorRow` per `case x rep` that failed due to an
  infrastructure error (`timeout`, `rate_limited`, `unavailable`,
  `served_model_mismatch`, or `api_error`). Infra failures never count as model
  failures.
- `report.txt`: rendered `pydantic-evals` report.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from evals.crisis.runs import portable_path, resolve_path
from evals.runner.cases import (
    ConversationCase,
    ExpectedPath,
    LevelName,
    Split,
    suite_digest,
)
from evals.runner.cases import load_suite as load_suite_cases
from evals.runner.tracing import RiskTrace

META_FILE: Final = "meta.json"
RESULTS_FILE: Final = "results.jsonl"
ERRORS_FILE: Final = "errors.jsonl"
REPORT_FILE: Final = "report.txt"

RunStatus = Literal["ok", "truncated"]
FailureClass = Literal[
    "timeout",
    "rate_limited",
    "unavailable",
    "served_model_mismatch",
    "api_error",
]


class ModelCallRecord(BaseModel):
    """Token usage, finish reason, latency and served model for one model call."""

    served_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    model_ms: int = 0
    retries: int = 0


class TurnRecord(BaseModel):
    """Everything recorded for one turn of a played conversation."""

    turn_index: int
    user: str
    reply: str
    path: ExpectedPath
    risk_level: LevelName
    state: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[str] = Field(default_factory=list)
    first_token_ms: int | None = None
    turn_ms: int = 0
    gate_ms: int = 0
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    truncated: bool = False
    risk_trace: RiskTrace | None = None


class SavedRunRow(BaseModel):
    """One row in `results.jsonl` (one `case x rep`)."""

    case_id: str
    rep: int = Field(ge=1)
    split: Split
    tags: list[str] = Field(default_factory=list)
    requested_model: str
    served_models: list[str] = Field(default_factory=list)
    status: RunStatus = "ok"
    turns: list[TurnRecord] = Field(default_factory=list)


class ErrorRow(BaseModel):
    """One row in `errors.jsonl` for an infrastructure failure."""

    case_id: str
    rep: int = Field(ge=1)
    turn_index: int | None = None
    failure_class: FailureClass
    error: str


class SuiteRunMeta(BaseModel):
    """Metadata stored in `meta.json` for a multi-turn eval run."""

    suite: str
    suite_path: str
    suite_digest: str
    split: Split | None = None
    requested_model: str
    served_models: list[str] = Field(default_factory=list)
    reps: int = Field(ge=1)
    started_at: str
    git_sha: str | None = None


def write_run_files(
    run_dir: Path,
    meta: SuiteRunMeta,
    results: Sequence[SavedRunRow],
    errors: Sequence[ErrorRow],
    report_text: str,
) -> None:
    """Write `meta.json`, `results.jsonl`, `errors.jsonl` and `report.txt`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / META_FILE).write_text(
        meta.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / RESULTS_FILE).write_text(
        "".join(row.model_dump_json() + "\n" for row in results),
        encoding="utf-8",
    )
    (run_dir / ERRORS_FILE).write_text(
        "".join(row.model_dump_json() + "\n" for row in errors),
        encoding="utf-8",
    )
    (run_dir / REPORT_FILE).write_text(report_text, encoding="utf-8")


def load_run_meta(run_dir: Path) -> SuiteRunMeta:
    """Read `meta.json` from a saved run directory."""
    return SuiteRunMeta.model_validate_json(
        (run_dir / META_FILE).read_text(encoding="utf-8")
    )


def read_saved_runs(run_dir: Path) -> list[SavedRunRow]:
    """Read all `SavedRunRow` items from `results.jsonl`."""
    return [
        SavedRunRow.model_validate_json(line)
        for line in _jsonl_lines(run_dir / RESULTS_FILE)
    ]


def read_error_rows(run_dir: Path) -> list[ErrorRow]:
    """Read all `ErrorRow` items from `errors.jsonl`."""
    return [
        ErrorRow.model_validate_json(line)
        for line in _jsonl_lines(run_dir / ERRORS_FILE)
    ]


def _jsonl_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_meta_cases(
    meta: SuiteRunMeta, run_dir: Path, *, suite_path: Path | None = None
) -> list[ConversationCase]:
    """Load the suite cases referenced by `meta`, verifying `suite_digest`.

    Raises:
        ValueError: If the suite file changed since `run_dir` was recorded.
    """
    path = suite_path if suite_path is not None else resolve_path(meta.suite_path)
    cases = load_suite_cases(path, split=meta.split)
    if suite_digest(cases) != meta.suite_digest:
        raise ValueError(
            f"{path} changed since {run_dir} was recorded "
            f"(expected digest {meta.suite_digest}). Record a new run."
        )
    return cases


def validate_run_rows(
    meta: SuiteRunMeta,
    cases: Sequence[ConversationCase],
    results: Sequence[SavedRunRow],
    errors: Sequence[ErrorRow],
) -> None:
    """Require exactly one result or infrastructure error per scheduled repetition.

    Raises:
        ValueError: If a saved run has missing, duplicate or unexpected repetitions.
    """
    expected = {(case.id, rep) for case in cases for rep in range(1, meta.reps + 1)}
    seen: set[tuple[str, int]] = set()
    rows: list[SavedRunRow | ErrorRow] = [*results, *errors]
    for row in rows:
        identity = (row.case_id, row.rep)
        if identity not in expected:
            raise ValueError(f"Unexpected saved run repetition: {identity}")
        if identity in seen:
            raise ValueError(f"Duplicate saved run repetition: {identity}")
        seen.add(identity)
    if missing := expected - seen:
        raise ValueError(
            f"Incomplete saved run; missing repetitions: {sorted(missing)}"
        )


__all__ = [
    "ERRORS_FILE",
    "META_FILE",
    "REPORT_FILE",
    "RESULTS_FILE",
    "ErrorRow",
    "FailureClass",
    "ModelCallRecord",
    "RunStatus",
    "SavedRunRow",
    "SuiteRunMeta",
    "TurnRecord",
    "load_meta_cases",
    "load_run_meta",
    "portable_path",
    "read_error_rows",
    "read_saved_runs",
    "resolve_path",
    "validate_run_rows",
    "write_run_files",
]
