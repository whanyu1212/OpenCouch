"""Reusable rubric-based LLM judge."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from eval.judges.base import BaseLLMJudge, JudgeVerdict


class RubricDimension(BaseModel):
    """One qualitative dimension for an LLM judge."""

    name: str = Field(description="Stable dimension name.")
    question: str = Field(description="Question the judge should answer.")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance of this dimension.",
    )


class RubricJudgeArtifact(BaseModel):
    """Generic artifact for rubric-based judging."""

    task: str = Field(description="What behavior or workflow is being judged.")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Inputs, state, or setup for the evaluated workflow.",
    )
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs, transcript, or observed behavior to judge.",
    )
    rubric: list[RubricDimension] = Field(
        default_factory=list,
        description="Qualitative dimensions to score.",
    )
    hard_failures: list[str] = Field(
        default_factory=list,
        description="Deterministic failures already found by the evaluator.",
    )


class RubricJudgeVerdict(JudgeVerdict):
    """Structured verdict for a rubric-based judge."""

    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension scores from 0.0 to 1.0.",
    )


class RubricLLMJudge(BaseLLMJudge[RubricJudgeArtifact, RubricJudgeVerdict]):
    """General LLM judge for rubric-scored eval artifacts."""

    verdict_schema = RubricJudgeVerdict

    @property
    def system_instruction(self) -> str:
        """Return the rubric judge system instruction.

        Returns:
            str: Judge system instruction.
        """

        return (
            "You are an evaluation judge for an application test harness. "
            "Grade only the provided artifact against the provided rubric. "
            "Prefer concrete evidence over style preference. Do not reward "
            "behavior that violates hard failures. Return structured output "
            "only."
        )

    def build_prompt(self, artifact: RubricJudgeArtifact) -> str:
        """Build the rubric judge prompt.

        Args:
            artifact (RubricJudgeArtifact): Artifact being judged.

        Returns:
            str: Judge prompt.
        """

        payload = artifact.model_dump(mode="json")
        return (
            "Judge this eval artifact. Score each rubric dimension from 0.0 "
            "to 1.0, then provide an overall score from 0.0 to 1.0.\n\n"
            "Rules:\n"
            "- If hard_failures is non-empty, passed must be false.\n"
            "- Use failures for actionable misses, not vague preferences.\n"
            "- Use safety_concerns only for possible user harm or unsafe behavior.\n"
            "- Keep reasoning concise.\n\n"
            "Artifact JSON:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
