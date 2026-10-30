"""The safety gate: screen every user turn and choose who replies.

Runs in code before any agent, on every turn. The reply path comes from the
conversation's level after `update_risk`, never from one message's reading,
so a calm message inside a crisis still goes to the crisis responder.

When Jev doesn't answer (HY-229), the turn is marked unscreened and screened
again at the start of the next turn, alongside the new message. A recheck
that raises the level applies from that turn on.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal

from pydantic_ai.messages import ModelMessage, ModelRequest

from opencouch_agent.safety.jev import FailureReason, JevClient
from opencouch_agent.safety.policy import CrisisRoute, RiskLevel, crisis_routes
from opencouch_agent.safety.questions import USER_TURN_QUESTIONS, UserSignals
from opencouch_agent.safety.risk import (
    CheckInAnswer,
    RiskState,
    UnscreenedTurn,
    update_risk,
)
from opencouch_agent.safety.screening import (
    ScreeningFailed,
    ScreeningResult,
    conversation_turns,
    screen_user_message,
    turn_text,
)

logger = logging.getLogger(__name__)

ReplyPath = Literal["companion", "crisis"]

MAX_UNSCREENED: Final = 4
"""Unscreened turns kept for a recheck: as many user turns as Jev's window
holds, so older ones are still read as context by the newer ones' rechecks."""
MAX_RECHECKS: Final = 3
"""Failed rechecks before a turn is given up on, so a message Jev always
rejects can't hold the turns after it back for good."""

# The CRISIS rules that mean the danger may be immediate: a timeline, a plan
# with the means, steps already taken, a medical emergency, or violence soon.
# Intent alone isn't.
_IMMINENT_ROUTES: Final[frozenset[CrisisRoute]] = frozenset(
    {"imminent", "plan_means", "preparation", "medical", "violence"}
)


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided for one user turn.

    Attributes:
        risk: The conversation's risk after this turn, to persist before the
            agent runs.
        path: Who replies.
        imminent: Whether this message, or a missed one rechecked this turn,
            suggests the danger is immediate, so the crisis reply puts the
            emergency number first.
        screening_failure: Why screening didn't run this turn, if it didn't.
        abandoned: How many missed turns were given up on unscreened this
            turn, for the audit log.
    """

    risk: RiskState
    path: ReplyPath
    imminent: bool = False
    screening_failure: FailureReason | None = None
    abandoned: int = 0


async def run_gate(
    jev: JevClient,
    risk: RiskState,
    history: Sequence[ModelMessage],
    latest: Sequence[ModelMessage],
    *,
    at: datetime,
    check_in: CheckInAnswer | None = None,
) -> GateDecision:
    """Recheck any turns screening missed, screen the new one, and decide who replies.

    Args:
        jev: The Jev client.
        risk: The conversation's risk before this turn.
        history: The stored conversation before `latest`.
        latest: The user's new messages.
        at: When the turn arrived; must be timezone-aware.
        check_in: A structured tap already matched to the server's current offer.

    Returns:
        The updated risk and the reply path. Never raises on a failed check.
    """
    pending = risk.unscreened
    rechecks, screening = await asyncio.gather(
        asyncio.gather(*(_recheck(jev, history, turn) for turn in pending)),
        screen_user_message(jev, conversation_turns(history), turn_text(latest)),
    )
    failed = isinstance(screening, ScreeningFailed)
    rechecked = _apply_rechecks(
        risk, pending, rechecks, at=at, floor=risk.level if failed else None
    )
    # Safety: unresolved or abandoned missed evidence cannot support step-down.
    if risk.unscreened and check_in == "safe":
        check_in = None
    if isinstance(screening, ScreeningFailed):
        if check_in == "need_help":
            # No classifier evidence: the explicit help request itself supplies
            # risk and forces CRISIS, so these zero inputs cannot lower the level.
            rechecked = replace(
                rechecked,
                risk=update_risk(
                    rechecked.risk,
                    UserSignals(**dict.fromkeys(USER_TURN_QUESTIONS, 0.0)),
                    at=at,
                    check_in=check_in,
                ),
                imminent=True,
            )
        missed = UnscreenedTurn(start=len(history), count=len(latest))
        return _unscreened_decision(rechecked, missed, screening.reason)
    updated = update_risk(rechecked.risk, screening.signals, at=at, check_in=check_in)
    path: ReplyPath = "crisis" if updated.level is RiskLevel.CRISIS else "companion"
    return GateDecision(
        risk=updated,
        path=path,
        imminent=check_in == "need_help"
        or rechecked.imminent
        or _imminent(screening.signals),
        abandoned=rechecked.abandoned,
    )


@dataclass(frozen=True)
class _Rechecked:
    risk: RiskState
    imminent: bool
    abandoned: int


def _imminent(signals: UserSignals) -> bool:
    return bool(_IMMINENT_ROUTES.intersection(crisis_routes(signals)))


async def _recheck(
    jev: JevClient, history: Sequence[ModelMessage], turn: UnscreenedTurn
) -> ScreeningResult | None:
    """Screen a missed turn as Jev would have then; None if it can't be found."""
    messages = history[turn.start : turn.start + turn.count]
    text = turn_text(messages)
    found = len(messages) == turn.count and all(
        isinstance(message, ModelRequest) for message in messages
    )
    if not found or not text:
        # Safety: screening the wrong text, or none, could read as calm, so a
        # marker that no longer points at a user turn is given up on, loudly.
        logger.error("Giving up on an unscreened-turn marker that points elsewhere")
        return None
    context = conversation_turns(history[: turn.start])
    return await screen_user_message(jev, context, text)


def _apply_rechecks(
    risk: RiskState,
    pending: Sequence[UnscreenedTurn],
    rechecks: Sequence[ScreeningResult | None],
    *,
    at: datetime,
    floor: RiskLevel | None,
) -> _Rechecked:
    """Apply the rechecked turns oldest first; the rest wait for the next turn.

    A turn after one that waits waits too, so turns always reach `update_risk`
    in the order they were sent. With a `floor`, a recheck that would take the
    level below it waits for a turn whose own screening succeeds.
    """
    waiting: list[UnscreenedTurn] = []
    imminent = False
    abandoned = 0
    for turn, recheck in zip(pending, rechecks, strict=True):
        if recheck is None:
            abandoned += 1
        elif isinstance(recheck, ScreeningFailed):
            failed = replace(turn, failed_rechecks=turn.failed_rechecks + 1)
            if failed.failed_rechecks < MAX_RECHECKS:
                waiting.append(failed)
            else:
                logger.warning(
                    "Giving up on a missed turn after %d rechecks", MAX_RECHECKS
                )
                abandoned += 1
        elif waiting:
            waiting.append(turn)
        else:
            updated = _rechecked(risk, recheck.signals, at=at)
            # Safety: on a turn whose own screening failed the level can't go
            # down, so a calm recheck can't settle it before routing.
            if floor is not None and updated.level.rank < floor.rank:
                waiting.append(turn)
                continue
            risk = updated
            imminent = imminent or _imminent(recheck.signals)
    return _Rechecked(replace(risk, unscreened=tuple(waiting)), imminent, abandoned)


def _rechecked(risk: RiskState, signals: UserSignals, *, at: datetime) -> RiskState:
    # Safety: dated to this turn rather than the missed one. A later time can
    # only restart the calm clock later, never bring a step-down sooner.
    updated = update_risk(risk, signals, at=at)
    marked = tuple(
        replace(change, reason=f"recheck of a missed turn: {change.reason}")
        for change in updated.changes[len(risk.changes) :]
    )
    return replace(updated, changes=(*risk.changes, *marked))


def _unscreened_decision(
    rechecked: _Rechecked, missed: UnscreenedTurn, reason: FailureReason
) -> GateDecision:
    risk = rechecked.risk
    unscreened = (*risk.unscreened, missed)
    overflow = max(0, len(unscreened) - MAX_UNSCREENED)
    if overflow:
        logger.warning(
            "Over %d unscreened turns; giving up on the oldest", MAX_UNSCREENED
        )
    # Fail closed: without a reading the level can't go down. From ELEVATED
    # up, where someone has already shown risk, the crisis responder replies;
    # below it the companion does, told that screening didn't run. Imminent
    # only if a rechecked turn said so; otherwise the crisis reply still shows
    # the emergency number, just not first.
    path: ReplyPath = (
        "crisis" if risk.level.at_least(RiskLevel.ELEVATED) else "companion"
    )
    return GateDecision(
        risk=replace(risk, unscreened=unscreened[overflow:]),
        path=path,
        imminent=rechecked.imminent,
        screening_failure=reason,
        abandoned=rechecked.abandoned + overflow,
    )
