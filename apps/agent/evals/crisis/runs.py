"""The files a recorded run is made of, and the check that guards them.

A run directory holds:

- `meta.json`: what the run was made from (`RunMeta`).
- `recorded.jsonl`: one `RecordedRow` per case Jev answered.
- `errors.jsonl`: one `ErrorRow` per case that failed in the latest run.

A recording is only valid for the exact input Jev was sent, so every run
stores `jev_input_version()`, a hash of the question text and the transcript
framing around the case text. Anything that reads a run checks it first.
"""

import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from opencouch_agent.safety import USER_TURN_QUESTIONS, Turn
from opencouch_agent.safety.screening import user_check_state

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


class InputVersionMismatchError(ValueError):
    """A run was recorded with different questions or transcript framing."""


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
