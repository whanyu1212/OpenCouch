"""Expose recorded judge verdicts as advisory Pydantic Evals metrics."""

from dataclasses import dataclass
from statistics import mean
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from evals.judge.jobs import row_jobs
from evals.judge.models import Judgment, QualityScores
from evals.runner.cases import ConversationCase
from evals.runner.runs import SavedRunRow


@dataclass(repr=False)
class SavedQualityEvaluator(Evaluator[ConversationCase, SavedRunRow, dict[str, Any]]):
    """Read verdicts without model calls; quality scores are not CI assertions."""

    judgments: list[Judgment]
    judge_reps: int = 1

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Keep full transcripts out of the evaluator's report specification."""
        return {"judge_reps": self.judge_reps}

    def evaluate(
        self, ctx: EvaluatorContext[ConversationCase, SavedRunRow, dict[str, Any]]
    ) -> EvaluatorOutput:
        """Emit distinct metrics for each served judge and require full coverage."""
        rows = [
            row
            for row in self.judgments
            if row.case_id == ctx.output.case_id and row.rep == ctx.output.rep
        ]
        expected_repetitions = {
            (turn, rep)
            for turn in range(len(ctx.output.turns))
            for rep in range(1, self.judge_reps + 1)
        }
        if self.judge_reps < 1 or (
            len(rows) != len(expected_repetitions)
            or {(row.turn_index, row.judge_rep) for row in rows} != expected_repetitions
        ):
            raise ValueError("judge coverage incomplete; see judge errors.jsonl")
        results: dict[str, float] = {}
        expected = row_jobs(ctx.output, ctx.inputs)
        for row in rows:
            if row.input != expected[row.turn_index].input:
                raise ValueError("judge verdict does not match saved context")
        identities = {
            (row.attempts[-1].route, row.attempts[-1].served_model) for row in rows
        }
        for route, model in identities:
            group = [
                row
                for row in rows
                if (row.attempts[-1].route, row.attempts[-1].served_model)
                == (route, model)
            ]
            prefix = f"judge:{route}:{model}"
            results[f"{prefix}:hard_fail_rate"] = mean(
                float(row.verdict.hard_failures.any_failed) for row in group
            )
            for name in QualityScores.model_fields:
                results[f"{prefix}:{name}"] = mean(
                    getattr(row.verdict.scores, name).score for row in group
                )
        return results
