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
    """Build the structured classification prompt for the crisis gate."""

    return (
        "Assess the user's current risk level for self-harm or suicide.\n\n"
        "Return only the structured schema.\n"
        "Use these levels:\n"
        "- 0: no acute risk\n"
        "- 1: high distress or ambiguous concerning language, but no clear "
        "self-harm signal\n"
        "- 2: self-harm or suicidal ideation without clear imminent plan\n"
        "- 3: imminent risk with plan, means, or near-term intent\n\n"
        "Mark needs_crisis_response true for levels 2 or 3.\n"
        "Mark needs_clarification true when the message is concerning but "
        "ambiguous.\n\n"
        "Do not treat ordinary planning or prioritization language as "
        "self-harm planning unless self-harm content is also present.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )
