"""The files a recorded run is made of, and the check that guards them.

A run directory holds:

- `meta.json`: what the run was made from (`RunMeta`).
- `recorded.jsonl`: one `RecordedRow` per case Jev answered.
- `errors.jsonl`: one `ErrorRow` per case that failed in the latest run.

A recording is only valid for the exact input Jev was sent and the exact cases
it was made from, so every run stores `jev_input_version()` (a hash of the
question text and the transcript framing) and a hash of its cases file.
Anything that reads a run checks both, and scoring also refuses a run with
cases that are neither recorded nor errored (an interrupted run).
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel

from evals.crisis.cases import Case, load_cases
from opencouch_agent.safety import USER_TURN_QUESTIONS, Turn
from opencouch_agent.safety.screening import user_check_state

AGENT_DIR: Final = Path(__file__).parents[2]
META_FILE: Final = "meta.json"
RECORDED_FILE: Final = "recorded.jsonl"
ERRORS_FILE: Final = "errors.jsonl"


class DatasetInfo(BaseModel):
    """Which revision of a downloaded dataset the cases came from."""

    dataset: str
    revision: str
    license: str


class RunMeta(BaseModel):
    """What a run was made from."""

    cases: str
    cases_path: str
    source: Literal["cases", "conversations"] = "cases"
    cases_digest: str
    limit: int | None = None
    jev_input_version: str
    started_at: str
    dataset: DatasetInfo | None = None


class RecordedRow(BaseModel):
    """A case and Jev's raw answers, keyed by question name."""

    id: str
    text: str
    label: str
    probabilities: dict[str, float]
    ms: int


class ErrorRow(BaseModel):
    """A case Jev didn't answer, and why."""

    id: str
    error: str


class RunCheckError(ValueError):
    """A run can't be trusted as it stands; the message says what to do."""


class InputVersionMismatchError(RunCheckError):
    """A run was recorded with different questions or transcript framing."""


class CasesChangedError(RunCheckError):
    """A run's cases file changed after it was recorded."""


class IncompleteRunError(RunCheckError):
    """Some of a run's cases have neither an answer nor an error."""


def portable_path(path: Path) -> str:
    """A path relative to apps/agent when inside it, else absolute.

    Committed runs then carry no local paths, and either form resolves from
    any working directory.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(AGENT_DIR):
        return str(resolved.relative_to(AGENT_DIR))
    return str(resolved)


def resolve_path(stored: str) -> Path:
    """Turn a stored path back into one that works from any directory."""
    path = Path(stored)
    return path if path.is_absolute() else AGENT_DIR / path


def _hashed_fields(case: Case) -> dict[str, Any]:
    fields = asdict(case)
    # Leave out an empty history, so single-message cases hash as they did
    # before cases could carry one and their recordings stay valid.
    if not case.history:
        del fields["history"]
    return fields


def cases_digest(cases: Sequence[Case]) -> str:
    """A short hash of every case's ID, text, label and history."""
    payload = json.dumps([_hashed_fields(case) for case in cases], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_source_cases(path: Path, source: str) -> list[Case]:
    """The cases in a cases file, or one per exchange of a conversations file."""
    if source == "conversations":
        # Imported here: conversations.py needs cases.py, not this module.
        from evals.crisis.conversations import exchange_cases, load_conversations

        return [
            case
            for conversation in load_conversations(path)
            for case in exchange_cases(conversation)
        ]
    return load_cases(path)


def jev_input_version() -> str:
    """A short hash of everything Jev is sent besides the case text."""
    framing = {
        "first_message": user_check_state([], "<case>"),
        "with_history": user_check_state(
            [Turn("user", "<user>"), Turn("assistant", "<assistant>")], "<case>"
        ),
    }
    payload = {"questions": dict(USER_TURN_QUESTIONS), "framing": framing}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def check_input_version(meta: RunMeta, run_dir: Path) -> None:
    """Refuse a run recorded with a different Jev input.

    Raises:
        InputVersionMismatchError: If the questions or framing changed since.
    """
    current = jev_input_version()
    if meta.jev_input_version != current:
        raise InputVersionMismatchError(
            f"{run_dir} was recorded with Jev input {meta.jev_input_version}, "
            f"but the questions or transcript framing are now {current}. "
            "Record a new run."
        )


def expected_cases(meta: RunMeta, run_dir: Path) -> list[Case]:
    """The cases a run is meant to cover, refusing a changed cases file.

    Raises:
        CasesChangedError: If the cases file differs from when it was recorded.
    """
    cases = load_source_cases(resolve_path(meta.cases_path), meta.source)
    if cases_digest(cases) != meta.cases_digest:
        raise CasesChangedError(
            f"{meta.cases_path} changed since {run_dir} was recorded "
            "(a case's text or label differs, or cases were added or removed). "
            "Record a new run."
        )
    return cases[: meta.limit]


def check_complete(
    expected: Sequence[Case],
    recorded: Sequence[RecordedRow],
    errors: Sequence[ErrorRow],
    run_dir: Path,
) -> None:
    """Refuse a run with cases that have neither an answer nor an error.

    Raises:
        IncompleteRunError: If the run was interrupted or is still running.
    """
    accounted = {row.id for row in recorded} | {row.id for row in errors}
    missing = [case.id for case in expected if case.id not in accounted]
    if missing:
        raise IncompleteRunError(
            f"{len(missing)} of {len(expected)} cases in {run_dir} have no "
            "answer or error yet. Finish the run with --resume."
        )


def load_meta(run_dir: Path) -> RunMeta:
    """Read a run's metadata."""
    return RunMeta.model_validate_json((run_dir / META_FILE).read_text())


def write_meta(run_dir: Path, meta: RunMeta) -> None:
    """Write a run's metadata."""
    (run_dir / META_FILE).write_text(meta.model_dump_json(indent=2) + "\n")


def read_recorded(run_dir: Path) -> list[RecordedRow]:
    """The run's answered cases, or none if nothing is recorded yet."""
    return [
        RecordedRow.model_validate_json(line)
        for line in _lines(run_dir / RECORDED_FILE)
    ]


def read_errors(run_dir: Path) -> list[ErrorRow]:
    """The cases that failed in the run's latest attempt."""
    return [
        ErrorRow.model_validate_json(line) for line in _lines(run_dir / ERRORS_FILE)
    ]


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip()]
