"""Server-owned crisis check-in offers and access to help after step-down."""

import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Final

from pydantic import BaseModel, Field

from opencouch_agent.agent.state import CompanionState
from opencouch_agent.safety.gate import GateDecision
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import CheckInAnswer, RiskState, check_in_due

CHECK_IN_TEXT: Final[dict[CheckInAnswer, str]] = {
    "safe": "I'm safe right now",
    "not_sure": "I'm not sure",
    "need_help": "I need help now",
}
RESOURCE_PIN_TIME: Final = timedelta(hours=2)
RETURN_AFTER: Final = timedelta(minutes=30)
RETURN_CHECK_IN: Final = "Are you safe right now?"
CHECK_IN_GUIDANCE: Final = (
    "The safety check-in buttons are visible below this reply. Ask the user to "
    "tap the answer that fits: I'm safe right now / I'm not sure / I need help now. "
    "Typed reassurance does not complete the check-in. Keep the conversation open."
)


class CheckInSubmission(BaseModel):
    """Untrusted structured tap, validated once at the request boundary."""

    offer_id: str = Field(min_length=1, max_length=64)
    answer: CheckInAnswer


class TurnActions(BaseModel):
    """Supported AG-UI forwarded properties."""

    crisis_check_in: CheckInSubmission | None = None


def returning_to_crisis(risk: RiskState, *, at: datetime) -> bool:
    """Whether this turn returns to an ongoing crisis after a gap."""
    return (
        risk.level is RiskLevel.CRISIS
        and risk.last_turn_at is not None
        and at - risk.last_turn_at >= RETURN_AFTER
    )


def accepted_check_in(
    submission: CheckInSubmission | None, risk: RiskState, text: str, *, at: datetime
) -> CheckInAnswer | None:
    """Accept only the current server offer with its exact screened button text."""
    if (
        submission is None
        or submission.offer_id != risk.check_in_offer
        or text != CHECK_IN_TEXT[submission.answer]
    ):
        return None
    # Safety: an old safe tap after silence must first get a fresh check-in.
    # Asking for help remains meaningful even if the offer has aged.
    if submission.answer == "safe" and returning_to_crisis(risk, at=at):
        return None
    return submission.answer


def prepare_follow_up(
    decision: GateDecision,
    previous: RiskState,
    resources: tuple[str, ...],
    *,
    at: datetime,
) -> GateDecision:
    """Consume the old offer and prepare the next, before recording the turn."""
    risk = decision.risk
    pinned_until = previous.resources_pinned_until
    if decision.path == "crisis":
        pinned_until = (
            None if risk.level is RiskLevel.CRISIS else at + RESOURCE_PIN_TIME
        )
    elif previous.level is RiskLevel.CRISIS:
        pinned_until = at + RESOURCE_PIN_TIME
    elif risk.peak is not RiskLevel.CRISIS:
        resources = ()
    offer = None
    if (
        decision.screening_failure is None
        and not risk.unscreened
        and check_in_due(risk)
    ):
        offer = uuid.uuid4().hex
    return replace(
        decision,
        risk=replace(
            risk,
            check_in_offer=offer,
            resources_pinned_until=pinned_until,
            crisis_resources=resources,
        ),
    )


def apply_follow_up(state: CompanionState, risk: RiskState, *, at: datetime) -> None:
    """Project persisted server decisions into UI state, including on reload."""
    resources = (
        list(risk.crisis_resources)
        if risk.crisis_resources is not None
        else state.crisis_resources
    )
    expiry = risk.resources_pinned_until
    state.crisis_resources = resources if expiry is None or at < expiry else []
    state.crisis_resources_until = expiry
    state.get_help_resources = resources if risk.peak is RiskLevel.CRISIS else []
    state.crisis_check_in = (
        None if returning_to_crisis(risk, at=at) else risk.check_in_offer
    )
