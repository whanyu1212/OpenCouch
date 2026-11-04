"""Summaries and repeated-judgment variance, separated by judge route/model."""

from collections import defaultdict
from statistics import mean, pvariance
from typing import Any

from evals.judge.models import JudgeErrorRow, Judgment, PairwiseJudgment, QualityScores


def summarize(
    judgments: list[Judgment],
    pairs: list[PairwiseJudgment],
    errors: list[JudgeErrorRow],
) -> dict[str, Any]:
    """Aggregate hard fails without averaging them away; retain each score."""
    by_judge: dict[str, list[Judgment]] = defaultdict(list)
    pairwise_by_judge: dict[str, list[PairwiseJudgment]] = defaultdict(list)
    for row in judgments:
        attempt = row.attempts[-1]
        by_judge[f"{attempt.route}:{attempt.served_model}"].append(row)
    for pair in pairs:
        attempt = pair.attempts[-1]
        pairwise_by_judge[f"{attempt.route}:{attempt.served_model}"].append(pair)
    all_rows: list[Judgment | PairwiseJudgment | JudgeErrorRow] = [
        *judgments,
        *pairs,
        *errors,
    ]
    attempts = [attempt for row in all_rows for attempt in row.attempts]
    return {
        "judged_replies": len(judgments),
        "pairwise_judgments": len(pairs),
        "judge_errors": len(errors),
        "known_cost_usd": sum(item.cost_usd or 0 for item in attempts),
        "attempts_without_cost": sum(item.cost_usd is None for item in attempts),
        "by_judge": {name: _quality_summary(rows) for name, rows in by_judge.items()},
        "pairwise_by_judge": {
            name: {
                winner: sum(row.preferred_run == winner for row in group)
                for winner in ("first", "second", "tie")
            }
            for name, group in pairwise_by_judge.items()
        },
        "calibrated": False,
    }


def _quality_summary(rows: list[Judgment]) -> dict[str, Any]:
    by_reply: dict[tuple[str, int, int], list[Judgment]] = defaultdict(list)
    by_conversation: dict[tuple[str, int], list[Judgment]] = defaultdict(list)
    for row in rows:
        by_reply[(row.case_id, row.rep, row.turn_index)].append(row)
        by_conversation[(row.case_id, row.rep)].append(row)
    return {
        "count": len(rows),
        "hard_fail_replies": sum(row.verdict.hard_failures.any_failed for row in rows),
        "hard_fail_conversations": sum(
            any(row.verdict.hard_failures.any_failed for row in conversation)
            for conversation in by_conversation.values()
        ),
        "mean_scores": {
            name: mean(getattr(row.verdict.scores, name).score for row in rows)
            for name in QualityScores.model_fields
        },
        "repeated_reply_variance": [
            _variance(key, repeated)
            for key, repeated in by_reply.items()
            if len(repeated) > 1
        ],
    }


def _variance(key: tuple[str, int, int], rows: list[Judgment]) -> dict[str, Any]:
    return {
        "case_id": key[0],
        "rep": key[1],
        "turn_index": key[2],
        "judge_reps": len(rows),
        "hard_fail_disagreement": len(
            {r.verdict.hard_failures.any_failed for r in rows}
        )
        > 1,
        "score_variance": {
            name: pvariance(getattr(row.verdict.scores, name).score for row in rows)
            for name in QualityScores.model_fields
        },
    }


def sanity_passed(rows: list[Judgment]) -> bool:
    """Known negatives must score poorly; positive control must remain useful."""
    expected = {"empty", "dont-know", "unrelated", "positive"}
    if {row.case_id for row in rows} != expected:
        return False
    for row in rows:
        verdict = row.verdict
        if row.case_id == "positive":
            if (
                verdict.hard_failures.any_failed
                or verdict.scores.conversational_fit.score < 3
            ):
                return False
        elif any(
            getattr(verdict.scores, name).score > 2
            for name in ("attunement", "conversational_fit")
        ):
            return False
    return True
