"""Turns Jev's probabilities into a risk level.

Pure functions with no API calls, so they are easy to test and tune. Ported
from the FirstRead repo (whanyu1212/firstread, `packages/firstread/src/policy.ts`
at `25b8aef`), where the thresholds were tuned on BHH, `cases.jsonl` and
`conversations.jsonl`. They lean low on purpose: missing a real crisis costs
far more than a false alarm. Don't change a threshold here without re-running
FirstRead's evals.
"""

from enum import StrEnum
from typing import Final

from opencouch_agent.safety.questions import UserSignals


class RiskLevel(StrEnum):
    """How much risk a conversation shows, lowest first."""

    NONE = "none"
    WATCH = "watch"
    """Possible passive ideation or distress: keep an eye on it."""
    ELEVATED = "elevated"
    """Active ideation, self-harm, or a plan: the reply must point to support."""
    CRISIS = "crisis"
    """Intent, preparation, or imminence: take the crisis path."""

    @property
    def rank(self) -> int:
        """Position in severity order, from 0 for `NONE`."""
        return _SEVERITY_ORDER.index(self)

    def at_least(self, minimum: "RiskLevel") -> bool:
        """Whether this level is `minimum` or more severe."""
        return self.rank >= minimum.rank


_SEVERITY_ORDER: Final = tuple(RiskLevel)

# Borderline: no single signal reaches ELEVATED, but the self-harm signals add
# up to this much.
BORDERLINE_SCORE: Final = 0.2
# On a borderline turn, any follow-up at or above this confirms the risk.
FOLLOW_UP_CONFIRMS: Final = 0.5

# Preparation (gathering means, goodbye letters, giving things away) at or above
# this is CRISIS.
PREPARATION_CRISIS: Final = 0.4
# Jev gives intent and imminence 0.2-0.36 on strong ideation with no plan or
# timing ("i've been thinking about killing myself"). These are the lowest
# cutoffs that stop such disclosures reading as CRISIS while every real crisis
# in FirstRead's multi-turn dev set still escalated (intent 0.6-0.8, imminent
# 0.5-0.6).
INTENT_CRISIS: Final = 0.35
IMMINENT_CRISIS: Final = 0.4
# Below the CRISIS cutoffs, intent or imminence still makes a turn at least
# ELEVATED, so raising those cutoffs changed how severe a turn is, not whether
# it's flagged. A veiled goodbye with nothing else firing needs this.
INTENT_ELEVATED: Final = 0.2
IMMINENT_ELEVATED: Final = 0.2
# Asking how to self-injure, ease or hide injuries, or avoid care.
METHOD_SEEKING_ELEVATED: Final = 0.4


def noisy_or(signals: UserSignals) -> float:
    """Chance that at least one self-harm signal is real, if they were independent.

    The signals overlap, so this overstates risk. Use it to rank turns and tune
    `BORDERLINE_SCORE` on evals; don't read it as a probability.
    """
    none_real = 1.0
    for probability in (
        signals.ideation,
        signals.self_harm,
        signals.plan,
        signals.means,
        signals.intent,
        signals.imminent,
    ):
        none_real *= 1 - probability
    return 1 - none_real


def assess_level(signals: UserSignals) -> RiskLevel:
    """Risk level for a single user turn."""
    if _is_crisis(signals):
        return RiskLevel.CRISIS
    if _is_elevated(signals) or _borderline_confirmed(signals):
        return RiskLevel.ELEVATED
    if _is_watch(signals):
        return RiskLevel.WATCH
    return RiskLevel.NONE


def escalate(previous: RiskLevel, current: RiskLevel) -> RiskLevel:
    """Combine a conversation's level with a new turn's. Risk only goes up.

    One calm message after a disclosure must not reset the conversation to
    safe. Lowering the level is a decision for a person, never for this code or
    for anything the client sends.
    """
    return current if current.rank > previous.rank else previous


def _is_crisis(signals: UserSignals) -> bool:
    return (
        signals.intent >= INTENT_CRISIS
        or signals.imminent >= IMMINENT_CRISIS
        or signals.preparation >= PREPARATION_CRISIS
        or (signals.plan >= 0.3 and signals.means >= 0.3)
    )


def _is_elevated(signals: UserSignals) -> bool:
    return (
        signals.intent >= INTENT_ELEVATED
        or signals.imminent >= IMMINENT_ELEVATED
        or signals.plan >= 0.2
        or signals.self_harm >= 0.3
        or signals.method_seeking >= METHOD_SEEKING_ELEVATED
        or signals.ideation >= 0.4
        or signals.harm_to_others >= 0.4
    )


def _borderline_confirmed(signals: UserSignals) -> bool:
    # Weak signals that add up, confirmed by a follow-up. Follow-ups can only
    # raise the level.
    return noisy_or(signals) >= BORDERLINE_SCORE and (
        signals.own_death_wish >= FOLLOW_UP_CONFIRMS
        or signals.disguised_request >= FOLLOW_UP_CONFIRMS
    )


def _is_watch(signals: UserSignals) -> bool:
    if (
        signals.ideation >= 0.15
        or signals.self_harm >= 0.15
        or signals.harm_to_others >= 0.2
    ):
        return True
    # counselor_concern is too broad to escalate on (depression, grief,
    # addiction), so it can only set WATCH.
    return (
        noisy_or(signals) >= BORDERLINE_SCORE
        and signals.counselor_concern >= FOLLOW_UP_CONFIRMS
    )
