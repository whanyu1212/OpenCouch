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

    if pending_memory_action:
        return VoiceTurnPolicy(
            route="memory_control",
            response_style="memory_control",
            instructions=(
                "A pending memory deletion is waiting for the user's decision. "
                "Do not continue ordinary therapeutic dispatch until the user "
                "clearly confirms, cancels, or moves away from that deletion. "
                "If they confirm, call confirm_memory_deletion. If they cancel "
                "or decline, call cancel_memory_deletion. If they change topics, "
                "briefly acknowledge that memory was not changed before "
                "answering the current request."
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
    instructions = (
        "Respond naturally for spoken therapeutic support. Use tools only "
        f"when their descriptions match the user's explicit request. "
        f"Memory mode is {mode_hint}."
    )
    if has_active_guided_exercise:
        instructions = (
            f"{instructions} Continue the current guided exercise if the user "
            "is responding to it. Do not start a new guided exercise."
        )
    return VoiceTurnPolicy(
        route="therapeutic",
        response_style="voice",
        instructions=instructions,
    )
