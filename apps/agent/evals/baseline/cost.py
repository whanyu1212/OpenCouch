"""Cost estimates from explicit rate snapshots, never guessed provider prices."""

from evals.baseline.models import Prices
from evals.runner.runs import ModelCallRecord


def coach_cost(
    calls: list[ModelCallRecord], turns: int, prices: Prices | None
) -> dict[str, float | None]:
    """Report covered-call estimates separately from an incomplete run total."""
    known = 0.0
    missing = 0
    for call in calls:
        rates = prices.models.get(call.served_model) if prices else None
        if rates is None:
            missing += 1
            continue
        if (
            not 0 <= call.cache_read_tokens <= call.input_tokens
            or call.output_tokens < 0
        ):
            raise ValueError("Invalid token counts for cost calculation")
        known += (
            (call.input_tokens - call.cache_read_tokens) * rates.input
            + call.cache_read_tokens * rates.cached_input
            + call.output_tokens * rates.output
        ) / 1_000_000
    total = known if calls and not missing else None
    return {
        "cost.coach_unpriced_calls": missing,
        "cost.coach_known_estimate_usd": known
        if calls and missing < len(calls)
        else None,
        "cost.coach_estimate_usd": total,
        "cost.coach_estimate_per_turn_usd": total / turns
        if total is not None and turns
        else None,
    }
