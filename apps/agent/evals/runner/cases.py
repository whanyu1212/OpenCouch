"""YAML case format for multi-turn conversation eval suites (HY-239)."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opencouch_agent.program.models import FormulationKind, WorrySource
from opencouch_agent.safety.follow_up import CHECK_IN_TEXT
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import CheckInAnswer

Split = Literal["dev", "holdout"]
ExpectedPath = Literal["coach", "crisis"]
LevelName = Literal["NONE", "WATCH", "ELEVATED", "CRISIS"]


class PositionSeed(BaseModel):
    """Initial program position seeded into `RecordStore` before a case runs."""

    model_config = ConfigDict(extra="forbid")

    week: int = Field(ge=1)
    step: str = Field(min_length=1)


class WorrySeed(BaseModel):
    """A parked worry seeded into `RecordStore` before a case runs."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source: WorrySource = "quick_capture"


class FormulationSeed(BaseModel):
    """A confirmed worry-map entry seeded before a case runs."""

    model_config = ConfigDict(extra="forbid")

    kind: FormulationKind
    text: str = Field(min_length=1)


class RecordsSeed(BaseModel):
    """Per-user program records seeded before turn 1."""

    model_config = ConfigDict(extra="forbid")

    position: PositionSeed | None = None
    worries: list[WorrySeed] = Field(default_factory=list)
    formulation: list[FormulationSeed] = Field(default_factory=list)


class TurnExpectation(BaseModel):
    """Deterministic behaviour expectations checked after a turn."""

    model_config = ConfigDict(extra="forbid")

    path: ExpectedPath | None = None
    risk_level: LevelName | None = None
    tools_allowed: list[str] | None = None
    tools_forbidden: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    crisis_resources: list[str] | None = None
    check_in_offered: bool | None = None

    @property
    def expected_risk(self) -> RiskLevel | None:
        """The expected `RiskLevel` enum value, if specified."""
        return RiskLevel[self.risk_level] if self.risk_level is not None else None


class TurnScript(BaseModel):
    """One scripted user turn and its expectations."""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1)
    minutes: float | None = None
    check_in: CheckInAnswer | None = None
    jev_reading: dict[str, float] = Field(default_factory=dict)
    expect: TurnExpectation = Field(default_factory=TurnExpectation)

    @model_validator(mode="after")
    def _check_tap_text(self) -> Self:
        if self.check_in is not None and self.user != CHECK_IN_TEXT[self.check_in]:
            expected = CHECK_IN_TEXT[self.check_in]
            raise ValueError(f"check_in={self.check_in!r} requires user={expected!r}")
        return self


class ConversationCase(BaseModel):
    """A scripted multi-turn conversation case in a YAML suite."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: Split
    tags: list[str] = Field(min_length=1)
    turns: list[TurnScript] = Field(min_length=1)
    records: RecordsSeed = Field(default_factory=RecordsSeed)
    quality_focus: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _check_minutes_order(self) -> Self:
        previous = 0.0
        for index, turn in enumerate(self.turns):
            if turn.minutes is not None:
                if turn.minutes < previous:
                    raise ValueError(f"Case {self.id!r}: turn minutes go backwards")
                previous = turn.minutes
            elif index > 0:
                previous += 1.0
        return self


def load_suite(path: Path, *, split: Split | None = None) -> list[ConversationCase]:
    """Load and validate a YAML suite of multi-turn conversation cases.

    Args:
        path: Path to the YAML suite file.
        split: Optional split filter (`"dev"` or `"holdout"`).

    Returns:
        The validated cases, filtered by `split` when given.

    Raises:
        ValueError: If the YAML is not a non-empty list of cases with unique IDs.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty YAML list of cases")
    cases = [ConversationCase.model_validate(item) for item in raw]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate case IDs")
    if split is not None:
        return [case for case in cases if case.split == split]
    return cases


def suite_digest(cases: Sequence[ConversationCase]) -> str:
    """Short SHA-256 digest of a suite's cases and expectations."""
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
