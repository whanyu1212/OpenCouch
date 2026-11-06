"""Calibration labels with explicit, per-reply human confirmation."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.judge.jobs import JudgeJob
from evals.judge.models import JudgeInput, QualityVerdict


class CalibrationItem(BaseModel):
    """A synthetic reply and its independently assigned rubric labels."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    source_case_id: str
    source_rep: int = Field(ge=1)
    source_turn_index: int = Field(ge=0)
    kind: Literal["recorded", "control"]
    tags: list[str]
    input: JudgeInput
    verdict: QualityVerdict | None = None
    confirmed_by: Literal["Wu Hanyu"] | None = None

    @model_validator(mode="after")
    def require_confirmed_verdict(self) -> Self:
        """Prevent an unfinished annotation from counting as reviewed."""
        if self.confirmed_by is not None and self.verdict is None:
            raise ValueError(f"Confirmed item has no verdict: {self.id}")
        return self


class CalibrationSet(BaseModel):
    """Frozen inputs and rubric, independent of the judge's predictions."""

    model_config = ConfigDict(extra="forbid")
    rubric_digest: str
    labeller: str | None = None
    source_run: str
    items: list[CalibrationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        """Reject duplicate identities before creating judge jobs."""
        identities = [item.id for item in self.items]
        if len(identities) != len(set(identities)):
            raise ValueError("Calibration item IDs must be unique")
        return self

    def jobs(self) -> list[JudgeJob]:
        """Pass only candidate context, never labels or confirmation, to the judge."""
        return [JudgeJob(item.id, 1, 0, item.input) for item in self.items]


def load_labels(path: Path) -> CalibrationSet:
    """Validate a human-editable YAML annotation file."""
    return CalibrationSet.model_validate(yaml.safe_load(path.read_text()))
