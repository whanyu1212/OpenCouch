"""Compatibility checks and named regressions against a committed baseline."""

from evals.baseline.models import Snapshot


def incompatibilities(baseline: Snapshot, candidate: Snapshot) -> list[str]:
    """Require the same experiment and evaluator, allowing model/prompt changes."""
    mismatches = []
    for name in ("suite_digest", "split", "reps"):
        if getattr(baseline.meta, name) != getattr(candidate.meta, name):
            mismatches.append(name)
    if baseline.evaluator_digest != candidate.evaluator_digest:
        mismatches.append("evaluator_digest")
    if (baseline.judge is None) != (candidate.judge is None):
        mismatches.append("judge recording presence")
    if baseline.judge is not None and candidate.judge is not None:
        for name in (
            "rubric_digest",
            "schema_digest",
            "judge_reps",
            "routes",
            "calibrated",
        ):
            if getattr(baseline.judge, name) != getattr(candidate.judge, name):
                mismatches.append(f"judge.{name}")
        if (
            baseline.judge_models
            and candidate.judge_models
            and baseline.judge_models != candidate.judge_models
        ):
            mismatches.append("served judge models")
    return mismatches


def comparison(baseline: Snapshot, candidate: Snapshot) -> tuple[str, bool]:
    """Print metric deltas and return whether deterministic checks regressed.

    Raises:
        ValueError: If the two experiments cannot be compared directly.
    """
    if mismatches := incompatibilities(baseline, candidate):
        raise ValueError(f"Incompatible comparison: {', '.join(mismatches)}")
    old = {(row.case_id, row.rep): row for row in baseline.outcomes}
    new = {(row.case_id, row.rep): row for row in candidate.outcomes}
    if old.keys() != new.keys():
        raise ValueError("Incompatible comparison: scheduled conversation identities")
    lines = [f"# {candidate.run} vs {baseline.run}", "", "## Conversation changes"]
    regressed = False
    for key in sorted(old):
        added = set(new[key].issues) - set(old[key].issues)
        removed = set(old[key].issues) - set(new[key].issues)
        regressed |= bool(added) or new[key].status != "complete"
        for label, issues in (("NEW", added), ("RESOLVED", removed)):
            lines.extend(
                f"- {label} {key[0]} rep {key[1]}: {issue}" for issue in sorted(issues)
            )
    if len(lines) == 3:
        lines.append("No changed conversation checks (existing failures may remain).")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "Metric | Baseline | Candidate | Delta",
            "--- | ---: | ---: | ---:",
        ]
    )
    lines.extend(_metric_lines(baseline, candidate))
    lines.extend(["", "## Judge hard-failure changes (advisory)"])
    for label, failures in (
        ("NEW", set(candidate.judge_failures) - set(baseline.judge_failures)),
        ("RESOLVED", set(baseline.judge_failures) - set(candidate.judge_failures)),
    ):
        lines.extend(f"- {label}: {failure}" for failure in sorted(failures))
    lines.extend(
        [
            "",
            f"Model: {baseline.meta.requested_model}"
            f" → {candidate.meta.requested_model}",
            f"Code: {baseline.meta.git_sha} → {candidate.meta.git_sha}",
            f"Prompt source: {baseline.prompt_digest} → {candidate.prompt_digest}",
            "",
            "Quality, latency and cost deltas are descriptive, not significance tests. "
            "Judge scores remain advisory. See intervals and missing-data counts.",
        ]
    )
    if baseline.prices != candidate.prices:
        lines.append("Price snapshots differ; coach cost deltas are not like-for-like.")
    if not baseline.prompt_digest or not candidate.prompt_digest:
        lines.append(
            "Prompt provenance is incomplete; inspect source revisions "
            "before interpreting changes."
        )
    incomplete_judge = any(
        candidate.metrics.get(f"judge.{key}", 0) for key in ("errors", "missing")
    )
    if incomplete_judge:
        lines.append(
            "Judge coverage is incomplete; comparison is not a clean confirmation."
        )
    return "\n".join(lines) + "\n", regressed or incomplete_judge


def _metric_lines(baseline: Snapshot, candidate: Snapshot) -> list[str]:
    lines = []
    for name in sorted(baseline.metrics.keys() | candidate.metrics.keys()):
        old, new = baseline.metrics.get(name), candidate.metrics.get(name)
        delta = new - old if old is not None and new is not None else None
        if name.startswith("cost.coach") and baseline.prices != candidate.prices:
            delta = None
        values = [
            "unavailable" if value is None else f"{value:.6g}"
            for value in (old, new, delta)
        ]
        lines.append(f"{name} | {' | '.join(values)}")
    return lines


def render(snapshot: Snapshot) -> str:
    """Render counts, separate metrics, clustered intervals and limitations."""
    lines = [
        f"# {snapshot.run}",
        "",
        f"Model: {snapshot.meta.requested_model}; "
        f"served: {snapshot.meta.served_models}",
        f"Code: {snapshot.meta.git_sha}; prompt digest: {snapshot.prompt_digest}",
        f"Suite: {snapshot.meta.suite_digest}; split: {snapshot.meta.split}; "
        f"repetitions: {snapshot.meta.reps}",
        f"Judge: {snapshot.judge_models}; rubric: "
        f"{snapshot.judge.rubric_digest if snapshot.judge else 'not supplied'}",
        "",
        "Metric | Value | Exploratory 95% interval",
        "--- | ---: | ---",
    ]
    for name, value in sorted(snapshot.metrics.items()):
        interval = snapshot.intervals.get(name)
        bounds = (
            f"[{interval.low:.3f}, {interval.high:.3f}] ({interval.clusters} scenarios)"
            if interval
            else "—"
        )
        lines.append(
            f"{name} | {value if value is not None else 'unavailable'} | {bounds}"
        )
    lines.extend(["", "## Failed or incomplete conversations"])
    for row in snapshot.outcomes:
        lines.extend(f"- {row.case_id} rep {row.rep}: {issue}" for issue in row.issues)
    lines.extend(["", "## Interpretation", *[f"- {note}" for note in snapshot.notes]])
    return "\n".join(lines) + "\n"
