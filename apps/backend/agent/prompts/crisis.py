"""Crisis-mode prompt builders.

Minimal interim module retained during the legacy cleanup. The previous
catalog/loader/modes/system machinery has been deleted; this file loads the
same markdown knowledge files directly so the runtime prompt content is
preserved until the prompt subsystem is redesigned.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from agent.state import AgentState

# Composition layers used to build the crisis system prompts. Each tuple is a
# sequence of relative paths under ``knowledge/`` that will be concatenated in
# order with blank-line separators.
_CORE_KNOWLEDGE = (
    "soul.md",
    "identity.md",
    "policy/boundaries.md",
    "policy/privacy.md",
)
_CRISIS_RESPONSE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "policy/crisis.md",
    "response_modes/crisis_response.md",
    "modalities/pfa.md",
    "modalities/dbt_skills.md",
)
_CRISIS_CLASSIFIER_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "policy/crisis.md",
)


def _knowledge_root() -> Path:
    """Return the absolute path to the repo-level ``knowledge/`` directory."""

    return Path(__file__).resolve().parents[4] / "knowledge"


@lru_cache(maxsize=32)
def _load_knowledge_file(relative_path: str) -> str:
    """Load one markdown knowledge file by its relative path.

    Raises:
        ValueError: If the resolved path escapes the knowledge root.
    """

    root = _knowledge_root().resolve()
    path = (root / relative_path).resolve()
    # Prevent directory traversal — relative_to raises if path escapes root.
    path.relative_to(root)
    return path.read_text(encoding="utf-8").strip()


def _compose(*relative_paths: str) -> str:
    """Concatenate knowledge files into a single prompt block."""

    parts = [_load_knowledge_file(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)


def _format_recent_history(state: AgentState, *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection."""

    history = state["history"][-limit:]
    if not history:
        return "(no prior history)"

    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in history
        if turn.get("content")
    )


def _format_found_resources(resources: list[dict[str, str]]) -> str:
    """Format a list of crisis resource dicts as a readable bullet list."""

    if not resources:
        return ""
    lines: list[str] = []
    for resource in resources:
        name = resource.get("name", "Crisis Line")
        phone = resource.get("phone", "")
        url = resource.get("url", "")
        entry = f"- {name}"
        if phone:
            entry += f": {phone}"
        if url:
            entry += f" ({url})"
        lines.append(entry)
    return "\n".join(lines)


def build_crisis_response_system_prompt() -> str:
    """Build the system prompt for crisis replies."""

    return _compose(*_CRISIS_RESPONSE_KNOWLEDGE)


def build_crisis_response_prompt(state: AgentState) -> str:
    """Build the user prompt for crisis replies.

    When search-derived resources are present in state
    (``response.found_resources``), they are injected into the prompt so the
    model can reference verified local hotlines. The model is explicitly
    instructed not to invent phone numbers.
    """

    crisis = state["crisis"]
    urgency = (
        "The user may be in immediate danger."
        if crisis.level >= 3
        else (
            "The user appears to have self-harm or suicidal ideation without "
            "a clear imminent plan."
        )
    )

    response_state = state.get("response", {})
    found_resources: list[dict[str, str]] = response_state.get("found_resources", [])
    inferred_location: str = response_state.get("inferred_location", "")

    if found_resources:
        resource_list = _format_found_resources(found_resources)
        resource_block = (
            f"\nVerified local crisis resources for "
            f"{inferred_location or 'the user\u2019s region'}:\n"
            f"{resource_list}\n"
            "Include at least one of these specific resources in your response. "
            "Do not modify phone numbers.\n"
        )
    else:
        resource_block = (
            "\nNo verified local resources were found. "
            "If the user is willing, ask for their country or region so the right "
            "local crisis line can be looked up. Do not invent phone numbers.\n"
        )

    return (
        "Write the next assistant message for a mental health support "
        "conversation in crisis mode.\n\n"
        "Acknowledge directly and calmly. Prioritize immediate safety — "
        "encourage contacting local emergency services and a trusted person. "
        "Ask at most one safety question. Be concise and clear.\n\n"
        f"Crisis context: {urgency}\n"
        f"Classifier reason: {crisis.reason}\n"
        f"{resource_block}\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_crisis_classifier_system_prompt() -> str:
    """Build the system prompt for the crisis classifier."""

    return _compose(*_CRISIS_CLASSIFIER_KNOWLEDGE)


def build_crisis_classifier_prompt(state: AgentState) -> str:
    """Build the structured classification prompt for the crisis gate.

    Level definitions are deliberately sharp and include side-by-side
    examples of adjacent levels. Over-escalation (level 1 → level 2)
    annoys users and erodes trust by triggering crisis interventions on
    ordinary distress; under-escalation (level 2 → level 1) is far worse
    because it treats real ideation as venting. This prompt is tuned to
    hold the line at level 2 — requiring EXPLICIT self-harm vocabulary —
    while still catching ambiguous warning signs at level 1.
    """

    return (
        "Assess the user's current risk level for self-harm or suicide.\n\n"
        "Return only the structured schema.\n\n"
        "═══ LEVEL DEFINITIONS (be precise — levels have sharp boundaries) ═══\n"
        "\n"
        "Level 0 — no acute risk:\n"
        "  The message contains no self-harm signal. Ordinary distress,\n"
        "  frustration, exhaustion, grief, overwhelm, low mood, or lack of\n"
        "  motivation ALL belong at level 0 unless they are paired with\n"
        "  explicit self-harm or suicidal vocabulary.\n"
        "  Examples (ALL level 0):\n"
        "    - 'I'm so tired of everything right now.' — venting, not ideation\n"
        "    - 'I'm at the end of my rope with this project.' — idiom for\n"
        "      frustration, NOT suicidal imagery\n"
        "    - 'My grandmother died last week and I'm having a hard time.' —\n"
        "      grief, not crisis\n"
        "    - 'I could kill my sister, she's so annoying.' — hyperbolic idiom\n"
        "    - 'lack of motivation to start the day' — low mood, not crisis\n"
        "    - 'Work has been stressful this week.' — transient stress\n"
        "    - 'I feel down today.' — transient low mood\n"
        "\n"
        "Level 1 — concerning but ambiguous; CLARIFY, don't crisis:\n"
        "  The message contains warning-sign language without explicit\n"
        "  self-harm vocabulary. Common level-1 markers: 'hopeless',\n"
        "  'trapped', 'can't do this anymore', 'wish I could disappear',\n"
        "  'wish I wasn't here', 'breaking point', 'no way out', 'burden',\n"
        "  'don't see the point'. These warrant a clarifying safety question\n"
        "  but NOT a full crisis response. The user is in distress but has\n"
        "  not said anything explicit about hurting themselves.\n"
        "  Examples (ALL level 1 — set needs_clarification=true):\n"
        "    - 'I feel completely hopeless and trapped.'\n"
        "    - 'I just wish I could disappear.'\n"
        "    - 'I can't do this anymore.'\n"
        "    - 'I'm at my breaking point and I don't know how much longer\n"
        "       I can hold it together.'\n"
        "    - 'Sometimes I wish I wasn't here anymore.'\n"
        "    - 'I feel like a burden on everyone.'\n"
        "    - 'It feels like there's no way out of this.'\n"
        "\n"
        "Level 2 — explicit self-harm or suicidal ideation WITHOUT\n"
        "imminent plan/means/timing:\n"
        "  The message contains direct, unambiguous language about wanting\n"
        "  to die, kill oneself, or hurt oneself. 'Want to', 'thinking\n"
        "  about', 'don't want to be here' + self-harm vocabulary all\n"
        "  qualify. No timing or means required.\n"
        "  Examples (ALL level 2 — set needs_crisis_response=true):\n"
        "    - 'I want to die.'\n"
        "    - 'I've been thinking about ending it all.'\n"
        "    - 'I keep thinking about hurting myself.'\n"
        "    - 'I don't want to be here anymore.' (in mental health context)\n"
        "    - 'I don't see any reason to live anymore.'\n"
        "    - 'I've been cutting again.'\n"
        "\n"
        "Level 3 — imminent risk: plan, means, or near-term timing:\n"
        "  Level 2 + specific plan, named means (pills, rope, gun), or\n"
        "  near-term timing ('tonight', 'tomorrow morning', 'this week').\n"
        "  Final preparations (farewell letters, stockpiling) also qualify.\n"
        "  Examples (ALL level 3 — set needs_crisis_response=true):\n"
        "    - 'I'm going to the bridge tonight.'\n"
        "    - 'I have a gun and I'm using it tomorrow morning.'\n"
        "    - 'I've saved up enough pills and written letters to my family.'\n"
        "\n"
        "═══ CORE RULES ═══\n"
        "\n"
        "1. Mark needs_crisis_response=true IFF level is 2 or 3.\n"
        "2. Mark needs_clarification=true IFF level is 1.\n"
        "3. Do NOT escalate from level 1 to level 2 without explicit\n"
        "   self-harm vocabulary. 'Hopeless', 'trapped', 'breaking point',\n"
        "   'wish I wasn't here', 'can't do this' are LEVEL 1. They require\n"
        "   clarification, not a crisis intervention.\n"
        "4. Do NOT escalate from level 0 to level 1 on ordinary distress\n"
        "   keywords. 'Tired', 'stressed', 'frustrated', 'grieving', 'low\n"
        "   mood' are level 0. Idioms like 'end of my rope', 'kill me',\n"
        "   'dying of laughter' with clear non-suicidal context are level 0.\n"
        "5. Do NOT treat grief language as crisis. 'Died', 'death', 'loss',\n"
        "   'passed away' in a bereavement context are level 0.\n"
        "6. Do NOT treat ordinary planning or prioritization language as\n"
        "   self-harm planning unless explicit self-harm content is present.\n"
        "7. False positives (over-escalating level 1 → level 2) are NOT\n"
        "   safe — they trigger crisis interventions on ordinary distress,\n"
        "   which erodes trust and trains users to hide distress signals.\n"
        "   The correct response to ambiguity is level 1 with clarification.\n"
        "8. When signals legitimately cluster (hopelessness + trapped +\n"
        "   explicit self-harm reference), you MAY escalate to level 2.\n"
        "   But distress alone never crosses the level-2 threshold.\n"
        "\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )
