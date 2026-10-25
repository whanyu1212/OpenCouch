"""The safety gate: screen every user turn and choose who replies.

Runs in code before any agent, on every turn. The reply path comes from the
conversation's level after `update_risk`, never from one message's reading,
so a calm message inside a crisis still goes to the crisis responder.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from pydantic_ai.messages import ModelMessage

from opencouch_agent.safety.jev import FailureReason, JevClient
from opencouch_agent.safety.policy import CrisisRoute, RiskLevel, crisis_routes
from opencouch_agent.safety.risk import RiskState, update_risk
from opencouch_agent.safety.screening import (
    ScreeningFailed,
    conversation_turns,
    screen_user_message,
)

ReplyPath = Literal["companion", "crisis"]

# The CRISIS rules that mean the danger may be immediate: a timeline, a plan
# with the means, or steps already taken. Intent alone isn't.
_IMMINENT_ROUTES: Final[frozenset[CrisisRoute]] = frozenset(
    {"imminent", "plan_means", "preparation"}
)


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided for one user turn.

    Attributes:
        risk: The conversation's risk after this turn, to persist before the
            agent runs.
        path: Who replies.
        imminent: Whether this message suggests the danger is immediate, so
            the crisis reply puts the emergency number first.
        screening_failure: Why screening didn't run this turn, if it didn't.
    """

    risk: RiskState
    path: ReplyPath
    imminent: bool = False
    screening_failure: FailureReason | None = None


async def run_gate(
    jev: JevClient,
    risk: RiskState,
    history: Sequence[ModelMessage],
    latest_message: str,
    *,
    at: datetime,
) -> GateDecision:
    """Screen the latest user message and decide who replies.

    Args:
        jev: The Jev client.
        risk: The conversation's risk before this turn.
        history: The stored conversation before `latest_message`.
        latest_message: The text of the user's new message.
        at: When the turn arrived; must be timezone-aware.

    Returns:
        The updated risk and the reply path. Never raises on a failed check.
    """
    screening = await screen_user_message(
        jev, conversation_turns(history), latest_message
    )
    if isinstance(screening, ScreeningFailed):
        # Fail closed: without a reading, reply as if this could be a crisis.
        # The level stays as it was, since there's no evidence to change it.
        # Not imminent: there's no reading saying so, and the emergency number
        # is still shown, just not first.
        return GateDecision(
            risk=risk, path="crisis", screening_failure=screening.reason
        )
    updated = update_risk(risk, screening.signals, at=at)
    path: ReplyPath = "crisis" if updated.level is RiskLevel.CRISIS else "companion"
    imminent = bool(_IMMINENT_ROUTES.intersection(crisis_routes(screening.signals)))
    return GateDecision(risk=updated, path=path, imminent=imminent)
