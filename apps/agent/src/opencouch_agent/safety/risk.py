"""Conversation-level risk (HY-208): each turn's check is evidence, the level a state.

`update_risk` is pure (no API calls, storage or clock), so the live gate and
the evals run the same code. Adapted from FirstRead's `risk.ts` as reference:

1. Evidence-backed CRISIS: intent, preparation, a medical emergency and
   violence soon stand alone; imminence and plan-plus-means also need
   self-harm context and a message that adds something, or the turn stops at
   ELEVATED.
2. Building score: what each message adds (`latest_adds_risk`) builds up and
   fades, and can raise the level to WATCH or ELEVATED, never CRISIS.
3. ELEVATED settles to WATCH after several calm turns, never to NONE.
4. Ours: CRISIS steps down to ELEVATED only after the score falls below
   `crisis_exit_score` and the clock has run `min_calm_minutes`, so the reply
   offers the check-in, and the user taps "I'm safe right now". Typed text
   never counts: a Jev question for it scored statements about the situation
   ("gave the pills to my sister") as high as real answers. After CRISIS the
   level never goes below ELEVATED, and new risk takes it straight back.

The per-user floor is in `floor.py`. Not here yet: the ledger and when to
show resources at ELEVATED.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Final, Literal

from opencouch_agent.safety.crisis_resources import CrisisType
from opencouch_agent.safety.policy import (
    CrisisRoute,
    RiskLevel,
    assess_level,
    crisis_routes,
    crisis_types,
)
from opencouch_agent.safety.questions import UserSignals

CheckInAnswer = Literal["safe", "not_sure", "need_help"]
"""A tap on the crisis check-in: I'm safe right now / not sure / need help now."""

# These stand alone; the other CRISIS routes need self-harm context (part 1).
# A medical emergency or violence soon isn't about self-harm, so asking for
# self-harm context would keep it from ever reaching CRISIS.
_SELF_STANDING: Final[frozenset[CrisisRoute]] = frozenset(
    {"intent", "preparation", "medical", "violence"}
)


@dataclass(frozen=True)
class RiskConfig:
    """Settings for `update_risk`.

    The first nine are FirstRead's tuned values. Change one only with evidence
    from the crisis evals. The step-down settings below them are tuned on
    HY-208's labelled step-down cases, except `min_calm_minutes`: no case
    needs it, so it stays as a cautious prior rather than being tuned away.
    """

    context_ideation: float = 0.3
    context_self_harm: float = 0.3
    evidence_floor: float = 0.2
    """Below this, what a message adds is background noise."""
    decay: float = 0.6
    """How much of the score carries to the next turn."""
    score_watch: float = 0.9
    score_elevated: float = 1.0
    """Both above what one message can add, so the score needs several turns."""
    calm_evidence: float = 0.15
    """`latest_adds_risk` below this is a calm turn."""
    acute_adds: float = 0.2
    settle_after: int = 4
    crisis_entry_score: float = 0.5
    """Entering CRISIS lifts the score to at least this; 0 turns it off."""
    crisis_exit_score: float = 0.3
    """CRISIS can't step down until the score is below this."""
    min_calm_minutes: float = 10.0
    """CRISIS can't step down until this long after the last risk signal."""
    clock_reset_adds: float = 0.35
    """`latest_adds_risk` at or above this restarts the `min_calm_minutes` clock."""


DEFAULT_RISK_CONFIG: Final = RiskConfig()


@dataclass(frozen=True)
class RiskChange:
    """A change of level and why, for the audit log."""

    turn: int
    from_level: RiskLevel
    to_level: RiskLevel
    reason: str


@dataclass(frozen=True)
class UnscreenedTurn:
    """A user turn that screening missed, to screen again on a later turn.

    Where the turn's messages sit in the stored history, never their text.
    Trimming history (HY-206) must shift these positions.

    Attributes:
        start: The index of the turn's first message in the stored history.
        count: How many messages the turn has.
        failed_rechecks: How many times a recheck has failed so far.
    """

    start: int
    count: int
    failed_rechecks: int = 0


@dataclass(frozen=True)
class RiskState:
    """A conversation's risk after its latest user turn.

    Attributes:
        level: The conversation's level.
        peak: The highest level reached, including any carried in.
        score: Building evidence; a policy measure, not a probability.
        turns: User turns so far.
        calm_turns: Calm turns in a row.
        last_risk_at: When a message last restarted the calm clock or raised
            the level.
        last_turn_at: When the latest user turn arrived.
        changes: Every change of level, oldest first.
        unscreened: Turns screening missed and hasn't rechecked, oldest first.
        crisis_types: Types observed in this thread, retained through calm turns
            and screening failures. Empty for older records or unknown risk.
    """

    level: RiskLevel = RiskLevel.NONE
    peak: RiskLevel = RiskLevel.NONE
    score: float = 0.0
    turns: int = 0
    calm_turns: int = 0
    last_risk_at: datetime | None = None
    last_turn_at: datetime | None = None
    changes: tuple[RiskChange, ...] = ()
    unscreened: tuple[UnscreenedTurn, ...] = ()
    crisis_types: frozenset[CrisisType] = frozenset()
    check_in_offer: str | None = None
    """Single-turn offer ID, persisted before the UI receives it."""
    resources_pinned_until: datetime | None = None
    """Wall-clock expiry after step-down, never extended by calm turns."""
    crisis_resources: tuple[str, ...] | None = None
    """Last server-selected services; None means a record predating follow-up."""


@dataclass(frozen=True)
class _Turn:
    """One turn's facts, worked out before deciding the level."""

    check_in: CheckInAnswer | None
    reading: RiskLevel
    reason: str
    adds: float
    calm: bool
    score: float
    calm_turns: int


def read_turn(
    signals: UserSignals, state: RiskState, config: RiskConfig = DEFAULT_RISK_CONFIG
) -> tuple[RiskLevel, str]:
    """This turn's own level after the CRISIS context rule, with the reason."""
    level = assess_level(signals)
    if level is not RiskLevel.CRISIS:
        return level, f"this message read as {level.name}"
    routes = crisis_routes(signals)
    if _SELF_STANDING.intersection(routes):
        return level, f"CRISIS: {', '.join(routes)}"
    has_context = (
        signals.ideation >= config.context_ideation
        or signals.self_harm >= config.context_self_harm
        or state.level.at_least(RiskLevel.ELEVATED)
    )
    if not has_context:
        return RiskLevel.ELEVATED, "held at ELEVATED: no self-harm context"
    if signals.latest_adds_risk < config.acute_adds:
        return RiskLevel.ELEVATED, "held at ELEVATED: echoes earlier messages"
    return level, f"CRISIS: {', '.join(routes)} with self-harm context"


def update_risk(
    state: RiskState,
    signals: UserSignals,
    *,
    at: datetime,
    check_in: CheckInAnswer | None = None,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> RiskState:
    """The conversation's risk after a user turn.

    Args:
        state: Risk before this turn.
        signals: Jev's answers for the turn's message.
        at: When the turn arrived; must be timezone-aware.
        check_in: The user's tap on the crisis check-in, if this turn is one.
        config: Settings; the defaults are the measured baseline.

    Returns:
        The new state. Only this function lowers a level, and never below
        ELEVATED once the conversation has been in CRISIS.

    Raises:
        ValueError: If `at` has no timezone.
    """
    if at.tzinfo is None:
        raise ValueError("at must be timezone-aware")
    turn = _assess(state, signals, check_in, config)
    level, reason = _next_level(state, turn, config)
    score = turn.score
    if level is RiskLevel.CRISIS and state.level is not RiskLevel.CRISIS:
        score = max(score, config.crisis_entry_score)
    restarts_clock = level.rank > state.level.rank or _restarts_clock(
        state, turn, config
    )
    changes = state.changes
    if level is not state.level:
        change = RiskChange(state.turns + 1, state.level, level, reason)
        changes = (*changes, change)
    return replace(
        state,
        level=level,
        peak=max(state.peak, level, key=lambda each: each.rank),
        score=score,
        turns=state.turns + 1,
        calm_turns=turn.calm_turns,
        last_risk_at=at if restarts_clock else state.last_risk_at,
        last_turn_at=at,
        changes=changes,
        unscreened=state.unscreened,
        crisis_types=state.crisis_types | crisis_types(signals),
    )


def check_in_due(state: RiskState, config: RiskConfig = DEFAULT_RISK_CONFIG) -> bool:
    """Whether the reply to the latest turn should offer the crisis check-in.

    True once everything but the user's answer allows CRISIS to step down,
    and only after a calm turn. A "safe" tap only counts on the turn after
    this held, so the server decides when to ask and the client can't step
    down on its own.
    """
    return (
        state.level is RiskLevel.CRISIS
        and state.last_turn_at is not None
        and state.calm_turns > 0
        and state.score < config.crisis_exit_score
        and _calm_for_long_enough(state, state.last_turn_at, config)
    )


def _assess(
    state: RiskState,
    signals: UserSignals,
    check_in: CheckInAnswer | None,
    config: RiskConfig,
) -> _Turn:
    reading, reason = read_turn(signals, state, config)
    adds = signals.latest_adds_risk
    # Asking for help is risk, whatever Jev made of the message.
    if check_in == "need_help":
        adds = 1.0
        reason = "the user asked for help now"
    calm = adds < config.calm_evidence
    return _Turn(
        check_in=check_in,
        reading=reading,
        reason=reason,
        adds=adds,
        calm=calm,
        score=config.decay * state.score + max(0.0, adds - config.evidence_floor),
        calm_turns=state.calm_turns + 1 if calm else 0,
    )


def _next_level(
    state: RiskState, turn: _Turn, config: RiskConfig
) -> tuple[RiskLevel, str]:
    raised = _raised_level(state, turn, config)
    if raised is not None:
        return raised
    if state.level is RiskLevel.CRISIS and _may_step_down(state, turn, config):
        return RiskLevel.ELEVATED, "the user tapped safe on the check-in"
    if _may_settle(state, turn, config):
        return RiskLevel.WATCH, f"{turn.calm_turns} calm turns with no new signal"
    return state.level, ""


def _raised_level(
    state: RiskState, turn: _Turn, config: RiskConfig
) -> tuple[RiskLevel, str] | None:
    """The higher level this turn calls for, or None if it doesn't raise it."""
    if turn.check_in == "need_help":
        return RiskLevel.CRISIS, turn.reason
    if _relapsed(state, turn):
        return RiskLevel.CRISIS, f"new risk after CRISIS: {turn.reason}"
    # On a calm turn, a reading no higher than the conversation has already
    # reached echoes an earlier message: it mustn't lift a settled conversation
    # back up. A first-time rise always counts.
    echo = turn.calm and turn.reading.rank <= state.peak.rank
    candidates = [(_score_level(turn.score, config), "evidence building across turns")]
    if not echo:
        candidates.append((turn.reading, turn.reason))
    level, reason = max(candidates, key=lambda candidate: candidate[0].rank)
    return (level, reason) if level.rank > state.level.rank else None


def _relapsed(state: RiskState, turn: _Turn) -> bool:
    # Safety: after a step-down, any message that adds risk and reads as
    # ELEVATED or above goes straight back to CRISIS, with no calm-turn wait.
    return (
        state.peak is RiskLevel.CRISIS
        and state.level is not RiskLevel.CRISIS
        and not turn.calm
        and turn.reading.at_least(RiskLevel.ELEVATED)
    )


def _restarts_clock(state: RiskState, turn: _Turn, config: RiskConfig) -> bool:
    # Safety: once a conversation has been in CRISIS, a message that adds
    # acute risk and reads as ELEVATED or above ("i still want to do it")
    # restarts the clock, however little it adds.
    renewed = (
        state.peak is RiskLevel.CRISIS
        and turn.adds >= config.acute_adds
        and turn.reading.at_least(RiskLevel.ELEVATED)
    )
    return renewed or turn.adds >= config.clock_reset_adds


def _may_step_down(state: RiskState, turn: _Turn, config: RiskConfig) -> bool:
    # Safety: the check-in must have been offered (the score and the time
    # floor stop a crisis ending early), and the answer must be a calm "safe"
    # tap (deflection and typed "I'm fine" never end it). A user who taps
    # "safe" to make the messages stop still gets through; ELEVATED keeps
    # resources showing.
    return check_in_due(state, config) and turn.calm and turn.check_in == "safe"


def _calm_for_long_enough(state: RiskState, at: datetime, config: RiskConfig) -> bool:
    # Fail closed: without a time for the last risk, calm can't be shown to
    # have lasted. Every rise sets it, so this only guards a malformed state.
    if state.last_risk_at is None:
        return False
    return at - state.last_risk_at >= timedelta(minutes=config.min_calm_minutes)


def _may_settle(state: RiskState, turn: _Turn, config: RiskConfig) -> bool:
    # Safety: a conversation that has been in CRISIS stays at ELEVATED or above.
    return (
        state.level is RiskLevel.ELEVATED
        and state.peak is not RiskLevel.CRISIS
        and turn.calm_turns >= config.settle_after
        and turn.score < config.score_elevated
    )


def _score_level(score: float, config: RiskConfig) -> RiskLevel:
    if score >= config.score_elevated:
        return RiskLevel.ELEVATED
    if score >= config.score_watch:
        return RiskLevel.WATCH
    return RiskLevel.NONE
