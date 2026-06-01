"""State-transition helpers for finalized voice turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from agent.models import Channel, CrisisAssessment
from agent.state import AgentState
from agent.voice.transcript import voice_turn_to_transcript_entries
from agent.voice.turn_metadata import VoiceTurnMetadata, infer_voice_turn_metadata


@dataclass(frozen=True, slots=True)
class VoiceTurnStateInputs:
    """Inputs required to build the persisted state for one voice turn."""

    thread_id: str
    user_id: str | None
    user_text: str
    assistant_text: str
    route: str | None
    response_style: str | None
    tool_calls: list[dict[str, Any]]
    prior_state: AgentState | None
    initial_state: AgentState
    prior_turn_count: int


@dataclass(frozen=True, slots=True)
class VoiceTurnStateResult:
    """Final persisted state plus derived voice metadata."""

    state: AgentState
    metadata: VoiceTurnMetadata


def build_voice_turn_state(inputs: VoiceTurnStateInputs) -> VoiceTurnStateResult:
    """Build the final persisted state for one voice turn."""
    state = cast(
        AgentState,
        {**dict(inputs.initial_state), **dict(inputs.prior_state or {})},
    )
    # Crisis-resource fields persist across turns so a within-turn
    # get_crisis_support_template can reuse the lookup from the same
    # turn (see persist/rehydrate in build_voice_tool_context). The
    # prior-wins merge above also carries a *previous* turn's lookup
    # into this turn, though, so reset to the per-turn baseline here:
    # this turn's real lookup, if any, is written back below from its
    # own tool calls, and a turn with no fresh lookup must not surface
    # a stale hotline from an earlier turn.
    state["resource_lookup_status"] = "not_attempted"
    state["found_resources"] = []
    state["inferred_location"] = ""

    prior_transcript = (
        list(inputs.prior_state.get("transcript", []) or [])
        if inputs.prior_state is not None
        else []
    )
    grounded_lookup: dict[str, Any] = {}
    crisis_resource_output: dict[str, Any] = {}
    voice_tool_calls = list(inputs.tool_calls)

    for call in voice_tool_calls:
        if not isinstance(call, Mapping):
            continue
        output = call.get("output")
        if not isinstance(output, Mapping):
            continue
        grounded_output = output.get("grounded_lookup")
        if isinstance(grounded_output, Mapping):
            grounded_lookup = dict(grounded_output)
        if call.get("tool_name") == "lookup_crisis_resources":
            crisis_resource_output = dict(output)

    metadata = infer_voice_turn_metadata(
        route=inputs.route,
        response_style=inputs.response_style,
        tool_calls=voice_tool_calls,
        has_grounded_lookup=bool(grounded_lookup),
    )
    entries = voice_turn_to_transcript_entries(
        user_text=inputs.user_text,
        assistant_text=inputs.assistant_text,
        response_style=metadata.response_style,
    )
    if not entries:
        raise ValueError("record_voice_turn requires user_text or assistant_text.")

    state.update(
        {
            "message": inputs.user_text.strip(),
            "channel": Channel.VOICE,
            "user_id": inputs.user_id,
            "session_id": inputs.thread_id,
            "response_text": inputs.assistant_text.strip(),
            "response_style": metadata.response_style,
            "route": metadata.route,
            "session_action": "none",
            "should_persist_memory": False,
            "transcript": [*prior_transcript, *entries],
            "grounded_lookup": {
                **dict(state.get("grounded_lookup", {}) or {}),
                **grounded_lookup,
            },
            "diagnostics": {
                **dict(state.get("diagnostics", {}) or {}),
                "voice_runtime": "openai_realtime",
                "voice_tool_calls": [
                    str(call.get("tool_name"))
                    for call in voice_tool_calls
                    if isinstance(call, Mapping) and call.get("tool_name")
                ],
            },
            "session_progress": {
                **dict(state.get("session_progress", {}) or {}),
                "turn_count": inputs.prior_turn_count + 1,
                "is_guest": inputs.user_id is None,
            },
        }
    )

    if metadata.route == "crisis":
        populate_voice_crisis_audit_state(
            state,
            crisis_resource_output=crisis_resource_output,
            voice_tool_calls=voice_tool_calls,
        )

    return VoiceTurnStateResult(state=state, metadata=metadata)


def populate_voice_crisis_audit_state(
    state: AgentState,
    *,
    crisis_resource_output: Mapping[str, Any],
    voice_tool_calls: list[dict[str, Any]],
) -> None:
    """Synthesize crisis audit fields for a voice crisis turn.

    Voice crisis handling is prompt-driven: the route is inferred from the
    model calling ``lookup_crisis_resources`` (see ``infer_voice_turn_metadata``).
    There is no server classifier, so the audit record records the
    tool-call signal rather than a classifier verdict. ``level`` is held at
    2 because voice has no independent imminence judgment to justify 3, and
    ``crisis_classifier_path`` is intentionally omitted so ``write_crisis_log``
    keeps its ``llm_primary`` default rather than inventing a new enum value.
    """
    state["crisis"] = CrisisAssessment(
        level=2,
        confidence="medium",
        reason="voice_crisis_tool_call",
        needs_crisis_response=True,
    )
    state["crisis_audit"] = {
        "crisis_override_kind": "none",
        "crisis_llm_failure_occurred": False,
    }

    status = crisis_resource_output.get("resource_lookup_status")
    if isinstance(status, str) and status:
        state["resource_lookup_status"] = status
    found_resources = crisis_resource_output.get("found_resources")
    if isinstance(found_resources, list):
        state["found_resources"] = [dict(row) for row in found_resources]
    inferred_location = crisis_resource_output.get("inferred_location")
    if isinstance(inferred_location, str) and inferred_location:
        state["inferred_location"] = inferred_location

    crisis_tool_names = [
        str(call.get("tool_name"))
        for call in voice_tool_calls
        if isinstance(call, Mapping) and call.get("tool_name")
    ]
    diagnostics = dict(state.get("diagnostics", {}) or {})
    diagnostics["openai_crisis_tool_calls"] = crisis_tool_names
    state["diagnostics"] = diagnostics


__all__ = [
    "VoiceTurnStateInputs",
    "VoiceTurnStateResult",
    "build_voice_turn_state",
    "populate_voice_crisis_audit_state",
]
