"""Timing helpers for agent diagnostics."""

from __future__ import annotations

import time


def elapsed_ms(started_at: float) -> float:
    """Return elapsed monotonic time in milliseconds.

    Args:
        started_at (float): Start timestamp from ``time.monotonic()``.

    Returns:
        float: Elapsed milliseconds rounded for diagnostics.
    """

    return round((time.monotonic() - started_at) * 1000, 2)
