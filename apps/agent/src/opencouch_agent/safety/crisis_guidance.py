"""Fixed per-type responder guidance and failure replies (HY-414).

Numbers come from the shipped Singapore catalog; tests check every fallback
against it. Emergency roles were checked against SCDF's Emergency Medical
Services page, SPF's SMS 70999 page and MSF Family Assist on 2026-10-30.
"""

from typing import Final

from opencouch_agent.safety.crisis_resources import (
    CrisisType,
    ordered_crisis_types,
)

TYPE_INSTRUCTIONS: Final[dict[CrisisType, str]] = {
    "medical_emergency": """## Medical emergency
Prioritise urgent medical help over counselling. For an overdose, serious injury,
or new or severe symptoms such as chest pain or trouble breathing, direct them
to 995. Do not diagnose or reassure them that symptoms are harmless. If they
describe their familiar panic without new or severe symptoms, acknowledge that
and slow down with them; do not insist every panic attack is an emergency.
Do not delay emergency help with exercises or questions about intent.""",
    "harm_to_others": """## Risk of harm to another person
If someone may be hurt now, direct them to police on 999. Focus on preventing
harm without blame or confrontation. Do not treat the user as an abuse victim
or recommend NAVH unless the conversation also says they are being harmed.
Unwanted intrusive thoughts alone are not intent to hurt anyone.""",
    "abuse_violence": """## Abuse or violence toward the user
Ask whether it is safe to use this device; it may be shared or monitored.
If someone may hurt them now, use 999, or SMS 70999 if it is unsafe to call or
they cannot speak. For abuse support, offer NAVH when safe to contact it.
Do not tell them to confront the person, announce plans or take an action that
could expose them. If harm to others is also flagged, distinguish who is at risk
from the conversation; do not assume the user is the perpetrator.""",
    "suicide_self_harm": """## Suicide or self-harm
Ask about immediate safety and encourage reaching SOS or a trusted person.
For an attempt underway or urgent medical danger, direct them to 995.
If medical danger is also flagged, medical help comes before counselling.""",
}

GENERAL_INSTRUCTIONS: Final = """## General support
The kind of crisis is unknown or has no specific guidance. Do not assume
self-harm. Ask one direct safety question and offer the listed services.
Use 995 for urgent medical help and 999 for immediate danger from violence."""

IMMINENT_INSTRUCTIONS: Final = """## Right now
This turn suggests immediate danger. Keep the reply short and lead with the
appropriate emergency service: 995 for medical danger or an attempt underway;
999 for imminent violence, or SMS 70999 when unsafe to call or unable to speak.
When both apply, prioritise urgent medical help without ignoring violence.
Then ask one direct safety question. Do not delay emergency help."""

MEDICAL_FALLBACK: Final = (
    "If you have taken an overdose or have new or severe symptoms such as chest "
    "pain or trouble breathing, call 995 now."
)
SELF_HARM_FALLBACK: Final = (
    "If you might act on thoughts of hurting yourself, call 995 now."
)
VIOLENCE_FALLBACK: Final = (
    "If someone may be hurt by another person now, call 999, or SMS 70999 if "
    "it is unsafe to call or you cannot speak."
)
GENERAL_FALLBACK: Final = (
    "If you need urgent medical help, call 995; if someone may be hurt by another "
    "person now, call 999, or SMS 70999 if it is unsafe to call."
)
FALLBACK_QUESTIONS: Final[dict[CrisisType, str]] = {
    "medical_emergency": "Have you been able to call for medical help?",
    "harm_to_others": "Is anyone in immediate danger right now?",
    "abuse_violence": "Is it safe to use this device?",
}


def crisis_fallback(types: frozenset[CrisisType]) -> str:
    """A fixed reply of at most four sentences, with emergency help first."""
    medical = "medical_emergency" in types
    self_harm = "suicide_self_harm" in types
    violence = bool(types & {"harm_to_others", "abuse_violence"})
    parts = []
    if medical or self_harm:
        parts.append(MEDICAL_FALLBACK if medical else SELF_HARM_FALLBACK)
    if violence:
        parts.append(VIOLENCE_FALLBACK)
    if not parts:
        parts.append(GENERAL_FALLBACK)
    if "abuse_violence" in types:
        parts.append(
            "If it is safe to use this device, you can contact NAVH "
            "on 1800-777-0000 for abuse support."
        )
    elif self_harm:
        parts.append("You can also call SOS on 1767 or WhatsApp 9151 1767.")
    elif not medical and not violence:
        parts.append("You can talk to someone at mindline on 1771.")
    primary = ordered_crisis_types(types)[0]
    parts.append(FALLBACK_QUESTIONS.get(primary, "Are you safe right now?"))
    return " ".join(parts)
