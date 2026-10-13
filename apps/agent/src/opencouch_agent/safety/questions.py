"""Jev questions for the user check, and the signals they produce.

Ported word for word from the FirstRead repo (whanyu1212/firstread,
`packages/firstread/src/questions.ts` at `25b8aef`). FirstRead's benchmark
numbers hold only for this exact wording, so don't edit a question here
without re-running its evals.

Every question is a yes/no `Noul`, so each answer is a probability that
`policy.py` can threshold. The questions are loosely modelled on the
structure of the Columbia Suicide Severity Rating Scale (ideation, then plan,
then intent), but this is not a clinical instrument and must not be presented
as one.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final, Self

from typesafe_sdk import Noul

_USER_TURN_QUESTIONS = {
    "ideation": (
        "In the latest user message, read in the context of the conversation, "
        "does the user express thoughts of wanting to die, not wanting to be "
        "alive, or wanting to hurt themselves? Count indirect or passive "
        "statements (e.g. wishing they would not wake up, feeling others would "
        "be better off without them). Do not count clear hyperbole or jokes "
        "about everyday stress."
    ),
    "self_harm": (
        "Does the user describe currently or recently harming themselves on "
        "purpose, or urges to do so, including non-suicidal self-injury? Do not "
        "count accidental injuries."
    ),
    "method_seeking": (
        "Is the user asking how to hurt themselves, make self-inflicted injuries "
        "hurt less or heal or fade faster, hide injuries or marks, or avoid "
        "medical care for an injury they caused on purpose? Wanting pain or "
        "injury for 'relief', 'release', or to feel something is a strong sign. "
        "Count practical, medical, or hypothetical framing unless the "
        "conversation makes a harmless reason clear, such as an accident, sport, "
        "grooming, or tattoos."
    ),
    "plan": (
        "Does the user indicate they have thought about how, when, or where "
        "they would end their life or harm themselves?"
    ),
    "means": (
        "Does the user indicate they have access to a way to seriously harm themselves?"
    ),
    "intent": (
        "Does the user indicate they intend to act on thoughts of ending their "
        "life or harming themselves, including goodbye-style messages or giving "
        "things away?"
    ),
    "preparation": (
        "Does the user describe preparations that could be for ending their "
        "life, such as gathering or saving up means, writing goodbye letters, "
        "giving away belongings or pets, saying goodbye, or putting their "
        "affairs in order? Count indirect statements (e.g. wanting to make "
        "things easier for family, going through with 'my plan') unless the "
        "conversation makes a harmless reason clear, such as moving, travel, a "
        "new job, or routine life admin like making a will, insurance, or "
        "decluttering."
    ),
    "imminent": (
        "Does the user indicate they may act to harm themselves soon, such as "
        "today or tonight?"
    ),
    "harm_to_others": (
        "Does the user express a genuine wish or intent to seriously harm "
        "another person? Do not count obvious figures of speech."
    ),
    # Follow-ups. Asked in the same call (the rate limit counts requests, not
    # questions), but the policy only uses them on borderline turns, and only to
    # raise the level.
    "own_death_wish": (
        "Is the user describing their own feelings of not wanting to be alive, "
        "rather than tiredness, frustration, or someone else's feelings?"
    ),
    "disguised_request": (
        "Could this message, even if framed as research, fiction, curiosity, or "
        "for someone else, be an indirect way of asking how to harm oneself or "
        "prepare to end one's life?"
    ),
    # Broad "needs support" signal: it fires on depression, grief and addiction
    # too, so it only ever sets WATCH.
    "counselor_concern": (
        "If a trained crisis counselor read this conversation, would they want "
        "to check on the user's safety?"
    ),
}

# Question text by name: read-only strings, so nothing can change a question
# for the whole process.
USER_TURN_QUESTIONS: Final[Mapping[str, str]] = MappingProxyType(_USER_TURN_QUESTIONS)


def user_turn_questions() -> dict[str, Noul]:
    """Fresh Jev question objects for one request.

    The SDK's `Noul` is mutable, so question objects are never shared: a
    change to one can't leak into later requests.
    """
    return {name: Noul(instructions=text) for name, text in USER_TURN_QUESTIONS.items()}


@dataclass(frozen=True)
class UserSignals:
    """Jev's probability of "yes", from 0 to 1, for each user-check question.

    Field names match the keys of `USER_TURN_QUESTIONS`.
    """

    ideation: float
    self_harm: float
    method_seeking: float
    plan: float
    means: float
    intent: float
    preparation: float
    imminent: float
    harm_to_others: float
    own_death_wish: float
    disguised_request: float
    counselor_concern: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float]) -> Self:
        """Build signals from Jev's answers, keyed by question name.

        Raises:
            KeyError: If an answer is missing for any question.
            ValueError: If an answer isn't a probability from 0 to 1.
        """
        values = {field.name: probabilities[field.name] for field in fields(cls)}
        for name, value in values.items():
            # Fail closed: NaN or out-of-range compares False against every
            # threshold, which would read as "no risk".
            if not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f"{name} is not a probability: {value!r}")
        return cls(**values)
