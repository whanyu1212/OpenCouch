"""Versioned baseline snapshots, provenance and explicit price assumptions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.runner.runs import SuiteRunMeta


class TokenRates(BaseModel):
    """USD per million tokens; cached input is a subset of total input."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)


class Prices(BaseModel):
    """Explicit rate snapshot, keyed by the exact served model name."""

    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    as_of: str = Field(min_length=1)
    models: dict[str, TokenRates]


class JudgeRoute(BaseModel):
    """Recorded judge settings that must match for scalar comparisons."""

    route: str
    requested_model: str
    effort: str | None
    max_tokens: int | None


class JudgeMeta(BaseModel):
    """Metadata from a scalar judge recording, not a pairwise run."""

    mode: Literal["quality"]
    rubric_digest: str
    schema_digest: str
    input_digest: str
    judge_reps: int = Field(ge=1)
    replies: int = Field(ge=0)
    routes: list[JudgeRoute]
    calibrated: bool


class Outcome(BaseModel):
    """One scheduled conversation; issues contain no generated reply text."""

    case_id: str
    rep: int
    status: Literal["complete", "error", "missing"]
    issues: list[str] = Field(default_factory=list)


class Interval(BaseModel):
    """Exploratory 95% scenario-cluster bootstrap interval."""

    low: float
    high: float
    clusters: int


class Snapshot(BaseModel):
    """Portable aggregates, never raw messages, verdict reasons or credentials."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    run: str
    meta: SuiteRunMeta
    evaluator_digest: str
    prompt_digest: str | None
    judge: JudgeMeta | None
    judge_models: list[str]
    judge_failures: list[str] = Field(default_factory=list)
    prices: Prices | None
    outcomes: list[Outcome]
    metrics: dict[str, float | None]
    intervals: dict[str, Interval | None]
    notes: list[str]
