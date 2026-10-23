"""The per-user risk floor: a new thread can't reset risk (HY-208).

When a thread ends at ELEVATED or CRISIS, the user's next thread starts at
the same level for a while. Only numbers and times carry over, never
conversation text, so incognito stays consistent. Storing the floor per user
needs accounts (HY-207); these functions are pure.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import RiskChange, RiskState

# Starting point from the 2026-10-16 decision; tune with evals.
FLOOR_HOURS: Final = 72.0


@dataclass(frozen=True)
class RiskFloor:
    """What carries into a new thread: numbers and times, no conversation text."""

    level: RiskLevel
    peak: RiskLevel
    score: float
    last_risk_at: datetime
    expires_at: datetime


def risk_floor(state: RiskState, hours: float = FLOOR_HOURS) -> RiskFloor | None:
    """What should carry into the user's next thread, if anything.

    Only ELEVATED and CRISIS carry. It expires `hours` after the last risk
    signal.
    """
    if not state.level.at_least(RiskLevel.ELEVATED) or state.last_risk_at is None:
        return None
    return RiskFloor(
        level=state.level,
        peak=state.peak,
        score=state.score,
        last_risk_at=state.last_risk_at,
        expires_at=state.last_risk_at + timedelta(hours=hours),
    )


def initial_risk(floor: RiskFloor | None, at: datetime) -> RiskState:
    """The starting risk for a new thread, from the user's floor if it holds."""
    if floor is None or at >= floor.expires_at:
        return RiskState()
    carried = RiskChange(0, RiskLevel.NONE, floor.level, "carried from another thread")
    return RiskState(
        level=floor.level,
        peak=floor.peak,
        score=floor.score,
        last_risk_at=floor.last_risk_at,
        changes=(carried,),
    )
