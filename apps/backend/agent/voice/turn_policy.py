"""Observe-only voice turn policy helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceTurnPolicy(BaseModel):
    """App-owned guidance for a finalized voice user transcript."""

    route: str
    response_style: str
    required_tool_name: str | None = None
    required_tool_arguments: dict[str, object] = Field(default_factory=dict)
    instructions: str


LOOKUP_MARKERS = (
    "look up",
    "lookup",
    "current",
    "latest",
    "official",
    "source",
    "verify",
)

CRISIS_MARKERS = (
    "hurt myself",
    "kill myself",
    "suicide",
    "crisis hotline",
    "emergency",
)


def build_voice_turn_policy(
    *,
    user_text: str,
    memory_mode: str,
    has_active_guided_exercise: bool,
    pending_memory_action: bool,
) -> VoiceTurnPolicy:
    """Return deterministic observe-only policy for one voice user turn."""

    del has_active_guided_exercise, pending_memory_action
    normalized = " ".join(user_text.lower().split())
    if any(marker in normalized for marker in CRISIS_MARKERS):
        return VoiceTurnPolicy(
            route="crisis",
            response_style="crisis_response",
            required_tool_name="lookup_crisis_resources",
            instructions=(
                "If specific crisis resources are needed, call "
                "lookup_crisis_resources. Be direct, supportive, and do not "
                "invent phone numbers or URLs."
            ),
        )

    if any(marker in normalized for marker in LOOKUP_MARKERS):
        query = user_text.strip()
        return VoiceTurnPolicy(
            route="grounded_lookup",
            response_style="grounded_lookup",
            required_tool_name="answer_grounded_lookup",
            required_tool_arguments={"query": query},
            instructions=(
                "Call answer_grounded_lookup exactly once before answering. "
                "Answer only from the tool result."
            ),
        )

    mode_hint = (
        "persistent" if memory_mode.strip().lower() == "persistent" else "incognito"
    )
    return VoiceTurnPolicy(
        route="therapeutic",
        response_style="voice",
        instructions=(
            "Respond naturally for spoken therapeutic support. Use tools only "
            f"when their descriptions match the user's explicit request. "
            f"Memory mode is {mode_hint}."
        ),
    )
