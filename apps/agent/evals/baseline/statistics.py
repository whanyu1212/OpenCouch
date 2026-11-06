"""Descriptive percentiles and exploratory uncertainty at scenario level."""

import random
from collections.abc import Sequence
from statistics import mean

from evals.baseline.models import Interval


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolate inclusively, including the single-value case."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_interval(groups: dict[str, list[float]]) -> Interval | None:
    """Resample whole scenarios; constant samples cannot identify uncertainty.

    Repetitions, turns and duplicate judgments stay together. This estimates
    the pooled mean, not an equally weighted mean of scenario means.
    """
    clusters = [values for _, values in sorted(groups.items()) if values]
    if len(clusters) < 2 or len({mean(values) for values in clusters}) == 1:
        return None
    rng = random.Random(241)
    means = [
        mean(
            value for group in rng.choices(clusters, k=len(clusters)) for value in group
        )
        for _ in range(2000)
    ]
    return Interval(
        low=sorted(means)[49], high=sorted(means)[1949], clusters=len(clusters)
    )


def timing_metrics(
    name: str, values: list[float], expected: int
) -> dict[str, float | None]:
    """Retain timing coverage, percentiles, and counts above three seconds."""
    return {
        f"{name}.count": len(values),
        f"{name}.missing": expected - len(values),
        f"{name}.p50_ms": percentile(values, 0.5),
        f"{name}.p90_ms": percentile(values, 0.9),
        f"{name}.over_3s": sum(value > 3000 for value in values),
        f"{name}.share_over_3s": (
            sum(value > 3000 for value in values) / len(values) if values else None
        ),
    }
