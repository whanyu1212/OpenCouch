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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.crisis.cases import BHH_LABELS
from evals.crisis.detection import (
    NEGATIVE_LABEL,
    POSITIVE_LABELS,
    Scored,
    ScoredCase,
    by_label,
    rate,
    score_rows,
    separation,
)
from evals.crisis.record import question_set_version, read_jsonl
from opencouch_agent.safety import RiskLevel

LEVELS = (RiskLevel.WATCH, RiskLevel.ELEVATED, RiskLevel.CRISIS)


def load_run(run_dir: Path) -> tuple[dict[str, Any], Scored, int]:
    """Load a run and score it, refusing a run made with other questions.

    Returns:
        The run's metadata, its scored cases, and how many cases errored.
    """
    meta: dict[str, Any] = json.loads((run_dir / "meta.json").read_text())
    if meta["question_set_version"] != question_set_version():
        raise SystemExit(
            f"{run_dir} was recorded with question set "
            f"{meta['question_set_version']}, but the questions are now "
            f"{question_set_version()}. Record a new run."
        )
    scored = score_rows(read_jsonl(run_dir / "recorded.jsonl"))
    errors = len(read_jsonl(run_dir / "errors.jsonl"))
    return meta, scored, errors


def label_table(cases: Sequence[ScoredCase]) -> dict[str, dict[str, Any]]:
    """Per-label counts and the share reaching each level."""
    groups = by_label(cases)
    order = [label for label in BHH_LABELS if label in groups] + sorted(
        label for label in groups if label not in BHH_LABELS
    )
    return {
        label: {"n": len(groups[label])}
        | {f">={level}": rate(groups[label], level) for level in LEVELS}
        for label in order
    }


def headline(cases: Sequence[ScoredCase]) -> dict[str, Any]:
    """The numbers the ship bar is judged on, where the labels allow."""
    positives = [c for c in cases if c.label in POSITIVE_LABELS]
    negatives = [c for c in cases if c.label == NEGATIVE_LABEL]
    violent = [c for c in cases if c.label == "violent_thoughts"]
    return {
        "si_self_harm_recall_elevated": rate(positives, RiskLevel.ELEVATED),
        "si_self_harm_recall_watch": rate(positives, RiskLevel.WATCH),
        "violent_thoughts_watch": rate(violent, RiskLevel.WATCH),
        "false_alarms_watch": rate(negatives, RiskLevel.WATCH),
        "false_alarms_elevated": rate(negatives, RiskLevel.ELEVATED),
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    """Everything worth reporting about a run, as plain data."""
    meta, scored, errors = load_run(run_dir)
    times = [row["ms"] for row in read_jsonl(run_dir / "recorded.jsonl")]
    return {
        "run": run_dir.name,
        "cases": meta["cases"],
        "question_set_version": meta["question_set_version"],
        "dataset": meta.get("dataset"),
        "scored_at": datetime.now(UTC).isoformat(),
        "counts": {
            "scored": len(scored.cases),
            "invalid": len(scored.invalid),
            "errors": errors,
        },
        "by_label": label_table(scored.cases),
        "headline": headline(scored.cases),
        "separation": separation(scored.cases),
        "batch_ms": {
            "p50": statistics.median(times) if times else None,
            "p90": statistics.quantiles(times, n=10)[-1] if len(times) > 1 else None,
        },
    }


def show(value: object) -> str:
    """Format a rate as a percentage, or a dash when there's nothing to rate."""
    if isinstance(value, float):
        return f"{value:.0%}"
    return "-" if value is None else str(value)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a summary as readable tables."""
    print(f"{summary['run']}  (questions {summary['question_set_version']})")
    print(f"counts: {summary['counts']}\n")
    print(f"{'label':<30}{'n':>5}{'>=WATCH':>10}{'>=ELEVATED':>12}{'CRISIS':>9}")
    for label, row in summary["by_label"].items():
        cells = [show(row[f">={level}"]) for level in LEVELS]
        print(f"{label:<30}{row['n']:>5}{cells[0]:>10}{cells[1]:>12}{cells[2]:>9}")
    print()
    for name, value in summary["headline"].items():
        print(f"{name:<32}{show(value)}")
    if summary["separation"]:
        print(f"\nAUC {summary['separation']['auc']:.3f}")
        for target, at in summary["separation"]["recall_at_false_alarms"].items():
            print(
                f"recall at false alarms <= {target}: {show(at['recall'])} "
                f"(actual {show(at['false_alarms'])})"
            )
    print(f"\nbatch time per call (not live latency): {summary['batch_ms']}")


def main() -> None:
    """Score a run, print it, and optionally write a baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="also write the summary here")
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    print_summary(summary)
    if args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
