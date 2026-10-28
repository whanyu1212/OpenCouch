"""Turns Jev's probabilities into a risk level.

Pure functions with no API calls, so they are easy to test and tune. Ported
from the FirstRead repo (whanyu1212/firstread, `packages/firstread/src/policy.ts`
at `25b8aef`), where the thresholds were tuned on BHH, `cases.jsonl` and
`conversations.jsonl`. They lean low on purpose: missing a real crisis costs
far more than a false alarm. They are now our measured baseline: change one
only with evidence from the crisis evals in `evals/crisis/`.

The crisis types beyond self-harm (HY-298) have cutoffs of our own, tuned on
our cases in `evals/crisis/samples/`. Each can raise the level by itself, so
someone in danger who isn't suicidal still reaches the crisis path.

`latest_adds_risk` doesn't set a turn's level; it is for conversation-level
risk. `asks_for_person` doesn't either: see `asks_for_person`.
"""

from enum import StrEnum
from typing import Final, Literal

from opencouch_agent.safety.crisis_resources import CrisisType
from opencouch_agent.safety.questions import UserSignals

CrisisRoute = Literal[
    "intent", "imminent", "preparation", "plan_means", "medical", "violence"
]
"""Which CRISIS rule a turn met. `risk.py` trusts some routes more than others."""


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
HARM_TO_OTHERS_ELEVATED: Final = 0.4
# Tuned on the 2026-10-28 recording (HY-298). Jev gives `abuse` 0.4-0.9 to
# many BHH `no_crisis` disclosures (controlling parents, emotionally abusive
# partners): lower cutoffs broke the ship bar on false alarms. Our abuse cases
# score 0.93 and up.
ABUSE_ELEVATED: Final = 0.9
# Danger now is CRISIS on much weaker abuse evidence: what matters is the
# danger, not how clearly the abuse is described.
ABUSE_WITH_VIOLENCE_SOON: Final = 0.4
# Jev reads ongoing abuse as "soon" (up to 0.79 on our cases with no
# immediate danger); those in danger now score 0.87 and up.
VIOLENCE_SOON_CRISIS: Final = 0.85
# No BHH `no_crisis` case reaches 0.3. Panic with chest symptoms does reach
# 0.54 ("chest is tight and my hands are shaking"), so ELEVATED starts above
# it; our medical emergencies score 0.96 and up.
MEDICAL_ELEVATED: Final = 0.6
MEDICAL_CRISIS: Final = 0.8
ASKS_FOR_PERSON: Final = 0.5


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


def crisis_routes(signals: UserSignals) -> list[CrisisRoute]:
    """Every CRISIS rule the turn meets; empty when it isn't CRISIS."""
    met: dict[CrisisRoute, bool] = {
        "intent": signals.intent >= INTENT_CRISIS,
        "imminent": signals.imminent >= IMMINENT_CRISIS,
        "preparation": signals.preparation >= PREPARATION_CRISIS,
        "plan_means": signals.plan >= 0.3 and signals.means >= 0.3,
        "medical": signals.medical_emergency >= MEDICAL_CRISIS,
        "violence": _abuse_in_danger_now(signals) or _harm_to_others_soon(signals),
    }
    return [route for route, is_met in met.items() if is_met]


def crisis_types(signals: UserSignals) -> frozenset[CrisisType]:
    """The kinds of crisis a turn shows; empty below ELEVATED.

    A turn can show several (abuse and suicidal thoughts). Every rule that
    makes a turn ELEVATED or CRISIS belongs to one of these types.
    """
    shown: dict[CrisisType, bool] = {
        "suicide_self_harm": _self_harm_shown(signals),
        "harm_to_others": signals.harm_to_others >= HARM_TO_OTHERS_ELEVATED,
        "abuse_violence": signals.abuse >= ABUSE_ELEVATED
        or _abuse_in_danger_now(signals),
        "medical_emergency": signals.medical_emergency >= MEDICAL_ELEVATED,
    }
    return frozenset(crisis_type for crisis_type, is_shown in shown.items() if is_shown)


def asks_for_person(signals: UserSignals) -> bool:
    """Whether the user is asking for a human, at any level.

    Not risk: it never changes the level, only whether the ways to reach a
    person are shown.
    """
    return signals.asks_for_person >= ASKS_FOR_PERSON


def _is_crisis(signals: UserSignals) -> bool:
    return bool(crisis_routes(signals))


# Violence soon alone can't say who is in danger, so it needs abuse or harm to
# others alongside.
def _abuse_in_danger_now(signals: UserSignals) -> bool:
    return (
        signals.abuse >= ABUSE_WITH_VIOLENCE_SOON
        and signals.violence_soon >= VIOLENCE_SOON_CRISIS
    )


def _harm_to_others_soon(signals: UserSignals) -> bool:
    return (
        signals.harm_to_others >= HARM_TO_OTHERS_ELEVATED
        and signals.violence_soon >= VIOLENCE_SOON_CRISIS
    )


def _is_elevated(signals: UserSignals) -> bool:
    return _self_harm_elevated(signals) or (
        signals.harm_to_others >= HARM_TO_OTHERS_ELEVATED
        or signals.abuse >= ABUSE_ELEVATED
        or signals.medical_emergency >= MEDICAL_ELEVATED
    )


def _self_harm_elevated(signals: UserSignals) -> bool:
    return (
        signals.intent >= INTENT_ELEVATED
        or signals.imminent >= IMMINENT_ELEVATED
        or signals.plan >= 0.2
        or signals.self_harm >= 0.3
        or signals.method_seeking >= METHOD_SEEKING_ELEVATED
        or signals.ideation >= 0.4
    )


def _self_harm_shown(signals: UserSignals) -> bool:
    # Preparation is the one self-harm CRISIS rule the ELEVATED rules don't
    # already cover.
    return (
        _self_harm_elevated(signals)
        or signals.preparation >= PREPARATION_CRISIS
        or _borderline_confirmed(signals)
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
