"""Score a recorded run with the current policy, offline and free.

    uv run python -m evals.crisis.score evals/results/crisis/<run>
    uv run python -m evals.crisis.score <run> --baseline evals/crisis/baselines/x.json

Prints the share of each label reaching WATCH, ELEVATED and CRISIS, the
headline numbers, and policy-free separation (AUC, recall at fixed false-alarm
rates). `--baseline` also writes the summary, aggregate numbers only and no
case text, to a file that can be committed.
"""

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from evals.crisis.cases import BHH_LABELS
from evals.crisis.detection import (
    NEGATIVE_LABEL,
    POSITIVE_LABELS,
    VIOLENT_THOUGHTS_LABEL,
    Scored,
    ScoredCase,
    Separation,
    by_label,
    rate,
    score_rows,
    separation,
)
from evals.crisis.runs import (
    RecordedRow,
    RunCheckError,
    RunMeta,
    check_complete,
    check_input_version,
    expected_cases,
    load_meta,
    read_errors,
    read_recorded,
)
from opencouch_agent.safety import RiskLevel


@dataclass(frozen=True)
class LabelRates:
    """For one label: how many cases, and the share reaching each level."""

    label: str
    cases: int
    watch: float | None
    elevated: float | None
    crisis: float | None


@dataclass(frozen=True)
class Headline:
    """The numbers the ship bar is judged on; None where the labels don't allow."""

    suicide_self_harm_recall_elevated: float | None
    suicide_self_harm_recall_watch: float | None
    violent_thoughts_watch: float | None
    false_alarms_watch: float | None
    false_alarms_elevated: float | None


@dataclass(frozen=True)
class Counts:
    """How many cases were scored, had invalid answers, or errored."""

    scored: int
    invalid: int
    errors: int


@dataclass(frozen=True)
class BatchTimes:
    """Time per Jev call with the batch client. Not live latency."""

    p50_ms: float | None
    p90_ms: float | None


@dataclass(frozen=True)
class RunSummary:
    """Everything worth reporting about a run, with no case text."""

    run: str
    cases: str
    jev_input_version: str
    dataset: dict[str, str] | None
    scored_at: str
    counts: Counts
    by_label: list[LabelRates]
    headline: Headline
    separation: Separation | None
    batch_times: BatchTimes


LEVELS: Final = (RiskLevel.WATCH, RiskLevel.ELEVATED, RiskLevel.CRISIS)


def load_run(run_dir: Path) -> tuple[RunMeta, list[RecordedRow], int]:
    """Load a run, refusing one that can't be trusted as it stands.

    Returns:
        The run's metadata, its recorded rows, and how many cases errored.

    Raises:
        RunCheckError: If the Jev input or the cases changed since recording,
            or some cases have neither an answer nor an error.
    """
    meta = load_meta(run_dir)
    check_input_version(meta, run_dir)
    recorded, errors = read_recorded(run_dir), read_errors(run_dir)
    check_complete(expected_cases(meta, run_dir), recorded, errors, run_dir)
    return meta, recorded, len(errors)


def label_rates(cases: Sequence[ScoredCase]) -> list[LabelRates]:
    """Per-label rates, BHH labels first in their usual order."""
    groups = by_label(cases)
    order = [label for label in BHH_LABELS if label in groups] + sorted(
        label for label in groups if label not in BHH_LABELS
    )
    return [
        LabelRates(
            label,
            len(groups[label]),
            *(rate(groups[label], level) for level in LEVELS),
        )
        for label in order
    ]


def headline(cases: Sequence[ScoredCase]) -> Headline:
    """The ship-bar numbers for a set of scored cases."""
    positives = [c for c in cases if c.label in POSITIVE_LABELS]
    negatives = [c for c in cases if c.label == NEGATIVE_LABEL]
    violent = [c for c in cases if c.label == VIOLENT_THOUGHTS_LABEL]
    return Headline(
        suicide_self_harm_recall_elevated=rate(positives, RiskLevel.ELEVATED),
        suicide_self_harm_recall_watch=rate(positives, RiskLevel.WATCH),
        violent_thoughts_watch=rate(violent, RiskLevel.WATCH),
        false_alarms_watch=rate(negatives, RiskLevel.WATCH),
        false_alarms_elevated=rate(negatives, RiskLevel.ELEVATED),
    )


def batch_times(rows: Sequence[RecordedRow]) -> BatchTimes:
    """Median and 90th percentile time per call, within the observed range."""
    times = [row.ms for row in rows]
    if len(times) < 2:
        only = float(times[0]) if times else None
        return BatchTimes(p50_ms=only, p90_ms=only)
    # Inclusive, so a small run's p90 never lands above its slowest call.
    deciles = statistics.quantiles(times, n=10, method="inclusive")
    return BatchTimes(p50_ms=statistics.median(times), p90_ms=deciles[-1])


def summarize(run_dir: Path) -> RunSummary:
    """Score a run with the current policy and summarize it."""
    meta, rows, errors = load_run(run_dir)
    scored: Scored = score_rows(rows)
    return RunSummary(
        run=run_dir.name,
        cases=meta.cases,
        jev_input_version=meta.jev_input_version,
        dataset=meta.dataset.model_dump() if meta.dataset else None,
        scored_at=datetime.now(UTC).isoformat(),
        counts=Counts(len(scored.cases), len(scored.invalid), errors),
        by_label=label_rates(scored.cases),
        headline=headline(scored.cases),
        separation=separation(scored.cases),
        batch_times=batch_times(rows),
    )


def show(value: float | None) -> str:
    """Format a rate as a percentage, or a dash when there's nothing to rate."""
    return "-" if value is None else f"{value:.0%}"


def print_summary(summary: RunSummary) -> None:
    """Print a summary as readable tables."""
    print(f"{summary.run}  (Jev input {summary.jev_input_version})")
    print(f"counts: {asdict(summary.counts)}\n")
    print(f"{'label':<30}{'cases':>6}{'>=WATCH':>10}{'>=ELEVATED':>12}{'CRISIS':>9}")
    for row in summary.by_label:
        cells = show(row.watch), show(row.elevated), show(row.crisis)
        print(f"{row.label:<30}{row.cases:>6}{cells[0]:>10}{cells[1]:>12}{cells[2]:>9}")
    print()
    for name, value in asdict(summary.headline).items():
        print(f"{name:<36}{show(value)}")
    if summary.separation:
        print(f"\nAUC {summary.separation.auc:.3f}")
        for at in summary.separation.recall_at_false_alarms:
            print(
                f"recall at false alarms <= {at.target:.0%}: {show(at.recall)} "
                f"(actual {show(at.false_alarms)})"
            )
    print(f"\nbatch time per call (not live latency): {asdict(summary.batch_times)}")


def main() -> None:
    """Score a run, print it, and optionally write a baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="also write the summary here")
    args = parser.parse_args()
    try:
        summary = summarize(args.run_dir)
    except RunCheckError as error:
        raise SystemExit(str(error)) from error
    print_summary(summary)
    if args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(asdict(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
