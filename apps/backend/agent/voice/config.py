"""Voice configuration, constants, and prompt construction.

Shared utilities for the LiveKit voice agent — voice normalization,
system prompt building, and realtime transcription defaults.
"""

from __future__ import annotations

DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE = "en"
DEFAULT_REALTIME_TRANSCRIPTION_PROMPT = (
    "This is a real-time spoken support conversation. Prefer exact wording. "
    "Preserve names, numbers, acronyms, contractions, and filler words when "
    "clearly heard."
)
DEFAULT_ASSISTANT_VOICE = "cedar"
SUPPORTED_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES = (
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "ko",
    "zh",
)


def normalize_assistant_voice(
    value: str | None,
    *,
    default: str = DEFAULT_ASSISTANT_VOICE,
) -> str:
    """Return a supported assistant voice name.

    Args:
        value (str | None): Candidate voice name from client or config.
        default (str): Fallback voice when ``value`` is blank or unsupported.

    Returns:
        str: Supported realtime voice name.
    """

    normalized = (value or "").strip().lower()
    if normalized in SUPPORTED_REALTIME_VOICES:
        return normalized
    if default in SUPPORTED_REALTIME_VOICES:
        return default
    return DEFAULT_ASSISTANT_VOICE


_MAX_PROMPT_CHARS = 12_000
_MAX_MEMORY_ITEMS = 6
_MAX_MEMORY_ITEM_CHARS = 220


def _trim_text(text: str, limit: int) -> str:
    """Trim text to a stable maximum length."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "\u2026"


def _trim_items(items: list[str] | None, *, max_items: int) -> list[str]:
    """Trim and bound a list of prompt items."""

    if not items:
        return []
    trimmed: list[str] = []
    for item in items[:max_items]:
        value = _trim_text(item, _MAX_MEMORY_ITEM_CHARS)
        if value:
            trimmed.append(value)
    return trimmed


def build_voice_system_prompt(
    *,
    semantic_facts: list[str] | None = None,
    episodic_arcs: list[str] | None = None,
    procedural_rules: list[str] | None = None,
    proactive_recall_enabled: bool = False,
) -> str:
    """Build a bounded voice-system prompt for Realtime sessions.

    The old voice prompt was large enough to exceed the Realtime
    instruction budget. This prompt stays deliberately compact and
    keeps only the guidance that materially affects live audio turns.

    Args:
        semantic_facts (list[str] | None): Short user facts to keep in working context.
        episodic_arcs (list[str] | None): Short summaries from prior sessions.
        procedural_rules (list[str] | None): Short style or preference rules for the user.
        proactive_recall_enabled (bool): Whether the user has allowed explicit
            references to relevant past memories.

    Returns:
        str: A compact prompt string safe for the Realtime session.
    """

    sections = [
        "\n".join(
            [
                "You are OpenCouch, a calm and emotionally intelligent voice support assistant.",
                "Speak like a thoughtful human conversation partner, not like a document.",
                "Respond in English unless the user explicitly asks to switch to another language.",
                "Prioritize verbal brevity. One or two meaningful sentences are often better than a perfect paragraph.",
                "Use plain spoken language. Do not use markdown, lists, headings, or long monologues.",
                "Do not make every reply follow reflection -> insight -> question. Let some moments be simple presence, warmth, or a short practical response.",
                "Use natural spoken backchannels sparingly, like 'mhm', 'yeah', 'I see', or 'I'm with you', when they would help the user feel accompanied.",
                "When the user is venting, processing, or still finding words, stay close to their experience. Do not rush to organize the conversation.",
                "Ask at most one focused follow-up when needed, and make it feel like interest rather than assessment.",
                "Prefer practical, grounded support over generic reassurance or abstract reflection.",
                "Active support does not mean jumping to a structured exercise; it can be quiet presence, naming a pattern, narrowing the target, reframing gently, or suggesting one small experiment.",
                "If the user interrupts or talks over you, yield immediately. Do not try to finish your sentence. Respond to what they said most recently.",
                "Do not introduce grounding, breathing, or other structured exercises just because the user sounds anxious, overwhelmed, or dysregulated.",
                "Use structured exercises only when the user explicitly asks for one, clearly agrees after you offer one, or is too activated to continue a normal conversation usefully.",
                "Once the user asks for a structured exercise or says yes to one you offered, begin the first step instead of asking for confirmation again.",
                "If you are unsure whether to keep talking or start a structured exercise, keep talking and offer one conversational next step.",
                "Do not present yourself as a licensed clinician or give medical or legal advice.",
                "If the user sounds in immediate danger or asks for crisis help, stop the normal conversation and tell them to contact local emergency services or a crisis line immediately. If they are in the US or Canada, tell them they can call or text 988.",
            ]
        )
    ]

    trimmed_rules = _trim_items(procedural_rules, max_items=_MAX_MEMORY_ITEMS)
    if trimmed_rules:
        sections.append(
            "User preferences:\n"
            + "\n".join(f"- {rule}" for rule in trimmed_rules)
            + "\nFollow these preferences silently. Do not quote, cite, or narrate them."
        )

    trimmed_facts = _trim_items(semantic_facts, max_items=_MAX_MEMORY_ITEMS)
    if trimmed_facts:
        sections.append(
            "Known context about the user:\n"
            + "\n".join(f"- {fact}" for fact in trimmed_facts)
        )

    trimmed_arcs = _trim_items(episodic_arcs, max_items=3)
    if trimmed_arcs:
        sections.append(
            "Relevant prior sessions:\n" + "\n".join(f"- {arc}" for arc in trimmed_arcs)
        )

    if proactive_recall_enabled:
        sections.append(
            "Memory reference guidance: proactive recall is on. If memory context "
            "is present and strongly relevant, you may briefly reference it, but "
            "do so sparingly and never force past material into the conversation."
        )
    else:
        sections.append(
            "Memory reference guidance: proactive recall is off. If memory context "
            "is present, use it only to shape warmth, pacing, and continuity. Do "
            "not explicitly mention past sessions or past statements unless the "
            "user asks about them."
        )

    prompt = "\n\n".join(sections)
    return _trim_text(prompt, _MAX_PROMPT_CHARS)
