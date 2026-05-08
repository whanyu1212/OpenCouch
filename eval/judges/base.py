"""Base classes for LLM-as-judge evals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Common structured verdict returned by an LLM judge."""

    passed: bool = Field(
        description="Whether the artifact passes the judge rubric overall."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score from 0.0 to 1.0.",
    )
    reasoning: str = Field(description="Concise rationale for the verdict.")
    strengths: list[str] = Field(
        default_factory=list,
        description="Specific things the artifact did well.",
    )
    failures: list[str] = Field(
        default_factory=list,
        description="Specific rubric misses or quality problems.",
    )
    safety_concerns: list[str] = Field(
        default_factory=list,
        description="Potential safety or user-harm concerns.",
    )


VerdictT = TypeVar("VerdictT", bound=JudgeVerdict)
ArtifactT = TypeVar("ArtifactT")


class StructuredJudgeLLM(Protocol):
    """Minimal LLM protocol required by judge classes."""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[VerdictT],
        system_instruction: str | None = None,
    ) -> VerdictT:
        """Generate a structured judge verdict.

        Args:
            prompt (str): Judge task prompt.
            response_schema (type[VerdictT]): Pydantic verdict schema.
            system_instruction (str | None): Optional judge system instruction.

        Returns:
            VerdictT: Structured judge verdict.
        """


@dataclass(frozen=True)
class JudgeOutcome(Generic[VerdictT]):
    """Combined result after applying hard checks and an LLM judge verdict."""

    passed: bool
    score: float
    failures: list[str]
    verdict: VerdictT

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome for `EvalResult.details`.

        Returns:
            dict[str, Any]: JSON-compatible outcome payload.
        """

        return {
            "passed": self.passed,
            "score": self.score,
            "failures": self.failures,
            "verdict": self.verdict.model_dump(mode="json"),
        }


class BaseLLMJudge(Generic[ArtifactT, VerdictT], ABC):
    """Base class for reusable LLM-as-judge graders."""

    verdict_schema: type[VerdictT]

    def __init__(
        self,
        *,
        llm_client: StructuredJudgeLLM,
        name: str | None = None,
    ) -> None:
        """Initialize the judge.

        Args:
            llm_client (StructuredJudgeLLM): LLM client used for judging.
            name (str | None): Optional judge name.

        Returns:
            None.
        """

        self.llm_client = llm_client
        self.name = name or self.__class__.__name__

    @property
    @abstractmethod
    def system_instruction(self) -> str:
        """Return the system instruction for this judge.

        Returns:
            str: Judge system instruction.
        """

    @abstractmethod
    def build_prompt(self, artifact: ArtifactT) -> str:
        """Build the judge prompt for an artifact.

        Args:
            artifact (ArtifactT): Artifact being judged.

        Returns:
            str: Judge prompt.
        """

    async def judge(self, artifact: ArtifactT) -> VerdictT:
        """Judge an artifact with the configured LLM.

        Args:
            artifact (ArtifactT): Artifact being judged.

        Returns:
            VerdictT: Structured verdict.
        """

        return await self.llm_client.generate_structured(
            prompt=self.build_prompt(artifact),
            response_schema=self.verdict_schema,
            system_instruction=self.system_instruction,
        )

    def combine(
        self,
        *,
        verdict: VerdictT,
        hard_failures: Sequence[str] = (),
        min_score: float = 0.7,
    ) -> JudgeOutcome[VerdictT]:
        """Combine deterministic failures with a judge verdict.

        Args:
            verdict (VerdictT): LLM judge verdict.
            hard_failures (Sequence[str]): Deterministic failures from the runner.
            min_score (float): Minimum acceptable judge score.

        Returns:
            JudgeOutcome[VerdictT]: Combined pass/fail outcome.
        """

        failures = [*hard_failures, *verdict.failures, *verdict.safety_concerns]
        if not verdict.passed and not failures:
            failures.append("judge verdict did not pass")
        if verdict.score < min_score:
            failures.append(
                f"judge score {verdict.score:.2f} below minimum {min_score:.2f}"
            )
        passed = not failures and verdict.passed
        return JudgeOutcome(
            passed=passed,
            score=0.0 if hard_failures else verdict.score,
            failures=failures,
            verdict=verdict,
        )
