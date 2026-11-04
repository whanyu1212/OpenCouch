"""The coach agent, its run dependencies, shared state and per-turn instructions."""

from opencouch_agent.coach.agent import build_coach_agent
from opencouch_agent.coach.deps import CoachDeps, SessionMode
from opencouch_agent.coach.instructions import (
    BASE_INSTRUCTIONS,
    ELEVATED_INSTRUCTIONS,
    REDIRECT_INSTRUCTIONS,
    UNSCREENED_INSTRUCTIONS,
    WATCH_INSTRUCTIONS,
    coach_instructions,
)
from opencouch_agent.coach.state import CompanionState

__all__ = [
    "BASE_INSTRUCTIONS",
    "ELEVATED_INSTRUCTIONS",
    "REDIRECT_INSTRUCTIONS",
    "UNSCREENED_INSTRUCTIONS",
    "WATCH_INSTRUCTIONS",
    "CoachDeps",
    "CompanionState",
    "SessionMode",
    "build_coach_agent",
    "coach_instructions",
]
