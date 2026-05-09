"""Keyword patterns for the LiveKit voice crisis safety net.

The top-level graph crisis gate is LLM-only. These patterns are kept for the
voice turn-completion safety net, which can hand off to the crisis agent before
the voice model chooses a function tool.
"""

from __future__ import annotations

# High-confidence immediate danger signals (plan/means/timing/finality).
IMMINENT_PATTERNS = (
    r"\b(?:tonight|today|right now|this (?:morning|afternoon|evening)|soon)\b.{0,35}\b(?:kill myself|end my life|end it|overdose|hang myself|jump)\b",
    r"\b(?:plan to|going to|gonna|will)\s+(?:kill myself|end my life|hurt myself|harm myself|overdose|jump)\b",
    r"\b(?:have|got)\s+(?:a|the|my)?\s*(?:gun|rope|knife|pills?|razor|blade)\b.{0,35}\b(?:for|to)\b.{0,20}\b(?:myself|end it|kill myself)\b",
    r"\b(?:this is my last|last message|goodbye everyone|you won't hear from me)\b",
    r"\b(?:swallow|take)\b.{0,20}\b(?:all|whole bottle|entire bottle)\b.{0,20}\b(?:pills?|tablets?)\b",
    r"\b(?:going to|headed to|driving to|walking to)\s+(?:the\s+)?(?:bridge|roof|railroad|tracks|river)\b",
    r"\b(?:saved up|stockpil(?:ed|ing)|collected|got enough)\s+(?:the\s+|my\s+|enough\s+)*(?:pills?|meds|medication|tablets?|sleeping pills?)\b",
    r"\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b.{0,40}\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now|when\b.{0,20}\bleaves?)\b",
    r"\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now)\b.{0,40}\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b",
)

# Explicit self-harm/suicidal language, including colloquialisms/evasion slang.
CLEAR_SELF_HARM_PATTERNS = (
    r"\b(?:kill myself|k[i1]ll m(?:yself|y self)|suicid(?:e|al)|end my life|want to die|die by suicide)\b",
    r"\b(?:hurt(?:ing)? myself|harm(?:ing)? myself|self[- ]?harm|cut(?:ting)? again|burn(?:ing)? myself)\b",
    r"\b(?:off myself|unalive myself|kms)\b",
    r"\b(?:don't want to be here anymore|nobody would miss me if i was gone|better off dead)\b",
    r"\b(?:go to sleep and (?:never|not) wake up|wish i didn't exist)\b",
    r"\b(?:end(?:ing)? it(?: all)?|want it to be over|ready for it to end)\b",
    r"\b(?:no|any)\s+(?:reason|point)\s+(?:to|in)\s+liv(?:e|ing)\b",
    r"\b(?:nothing (?:left )?to live for|can't see a (?:future|reason))\b",
)

# Possible risk language requiring clarification.
AMBIGUOUS_PATTERNS = (
    r"\b(?:wish i could disappear|wish i wasn't here|wish i could go away)\b",
    r"\b(?:i can't do this anymore|i am done|i'm done|give up)\b",
    r"\b(?:dark thoughts|not sure i can keep going|can't keep going)\b",
    r"\b(?:i want to disappear|i don't want to exist)\b",
    r"\b(?:better off without me|i'm a burden|everyone would be better without me)\b",
    r"\bno way out\b",
    r"\b(?:don't see the point|no point)\b.{0,30}\b(?:getting (?:out of )?bed|waking up|going on|anything|even trying)\b",
)

__all__ = [
    "AMBIGUOUS_PATTERNS",
    "CLEAR_SELF_HARM_PATTERNS",
    "IMMINENT_PATTERNS",
]
