"""Per-run dependencies passed to the coach agent."""

from dataclasses import dataclass, field
from typing import Literal

from opencouch_agent.coach.state import CompanionState
from opencouch_agent.program.models import FormulationEntry, ProgramPosition, Worry
from opencouch_agent.safety.policy import RiskLevel

SessionMode = Literal["normal", "worry_time", "weekly_session"]


@dataclass
class CoachDeps:
    """Run dependencies for the coach.

    Must stay a dataclass with a `state` field so the AG-UI adapter can treat
    it as a `StateHandler` and sync shared state with the frontend.

    Every other field is loaded on the server from the risk state or the
    authenticated user's records in `RecordStore`. Client-sent state is never
    copied into these fields.
    """

    state: CompanionState = field(default_factory=CompanionState)
    risk_level: RiskLevel = RiskLevel.NONE
    unscreened: bool = False
    redirect_reassurance: bool = False
    session_mode: SessionMode = "normal"
    position: ProgramPosition | None = None
    lesson_content: str | None = None
    worries: tuple[Worry, ...] = ()
    formulation: tuple[FormulationEntry, ...] = ()
