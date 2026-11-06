"""Offline agreement, using only human-confirmed labels in every denominator."""

from pathlib import Path

from pydantic import BaseModel, computed_field

from evals.baseline.judging import load_judgments
from evals.calibration.models import CalibrationSet
from evals.judge.models import HardFailures, QualityScores


class CriterionAgreement(BaseModel):
    """Counts precede rates; repetitions are not independent human labels."""

    matched: int = 0
    total: int = 0
    human_positive: int = 0
    positive_matched: int = 0
    human_negative: int = 0
    negative_matched: int = 0

    @computed_field  # type: ignore[prop-decorator]  # Pydantic's documented mypy limitation.
    @property
    def rate(self) -> float | None:
        """Return no estimate when there are no confirmed observations."""
        return self.matched / self.total if self.total else None


class AgreementReport(BaseModel):
    """Review completeness and agreement are separate from calibration readiness."""

    total_items: int
    confirmed_items: int
    judged_confirmed_items: int
    judge_errors: int
    missing_judgments: int
    hard_failures: dict[str, CriterionAgreement]
    scores: dict[str, CriterionAgreement]
    disagreements: list[str]
    meets_target: bool
    advisory: bool = True


def agreement(labels: CalibrationSet, directory: Path) -> AgreementReport:
    """Check recorded input/rubric integrity and report each criterion separately."""
    meta, judgments, errors, missing = load_judgments(directory, labels.jobs())
    if meta.rubric_digest != labels.rubric_digest:
        raise ValueError("Calibration labels and judge run use different rubrics")
    confirmed = {
        item.id: item.verdict
        for item in labels.items
        if item.confirmed_by is not None and item.verdict is not None
    }
    hard = {name: CriterionAgreement() for name in HardFailures.model_fields}
    scores = {name: CriterionAgreement() for name in QualityScores.model_fields}
    disagreements: list[str] = []
    judged: set[str] = set()
    for row in judgments:
        expected = confirmed.get(row.case_id)
        if expected is None:
            continue
        judged.add(row.case_id)
        for name, counts in hard.items():
            wanted = getattr(expected.hard_failures, name).failed
            actual = getattr(row.verdict.hard_failures, name).failed
            _count_binary(counts, wanted, actual)
            if wanted != actual:
                disagreements.append(f"{row.case_id}:rep{row.judge_rep}:{name}")
        for name, counts in scores.items():
            wanted_score = getattr(expected.scores, name).score
            actual_score = getattr(row.verdict.scores, name).score
            counts.total += 1
            counts.matched += abs(wanted_score - actual_score) <= 1
            if abs(wanted_score - actual_score) > 1:
                disagreements.append(f"{row.case_id}:rep{row.judge_rep}:{name}")
    complete = len(confirmed) == len(labels.items) and not errors and missing == 0
    return AgreementReport(
        total_items=len(labels.items),
        confirmed_items=len(confirmed),
        judged_confirmed_items=len(judged),
        judge_errors=len(errors),
        missing_judgments=missing,
        hard_failures=hard,
        scores=scores,
        disagreements=disagreements,
        meets_target=complete
        and all(
            count.rate is not None
            and count.rate >= 0.9
            and count.human_positive > 0
            and count.human_negative > 0
            for count in hard.values()
        ),
    )


def _count_binary(counts: CriterionAgreement, wanted: bool, actual: bool) -> None:
    counts.total += 1
    counts.matched += wanted == actual
    counts.human_positive += wanted
    counts.positive_matched += wanted and actual
    counts.human_negative += not wanted
    counts.negative_matched += not wanted and not actual
