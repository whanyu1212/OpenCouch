"""State-transition helpers for finalized voice turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from agent.models import Channel, CrisisAssessment
from agent.runtime.state_ops import build_effective_turn_state
from agent.observability.decorators import trace_event
from agent.observability.events import VOICE_TURN_STATE_BUILT
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
    correlation_hash: str | None = None
    outcome: Literal["completed", "connection_interrupted", "safety_interrupted"] = (
        "completed"
    )
    safety_assessment: CrisisAssessment | None = None


@dataclass(frozen=True, slots=True)
class VoiceTurnStateResult:
    """Final persisted state plus derived voice metadata."""

    state: AgentState
    metadata: VoiceTurnMetadata


def build_voice_turn_state(inputs: VoiceTurnStateInputs) -> VoiceTurnStateResult:
    """Build the final persisted state for one voice turn."""
    if (
        inputs.outcome in {"connection_interrupted", "safety_interrupted"}
        and not inputs.user_text.strip()
    ):
        raise ValueError(f"{inputs.outcome} requires user_text")

    state = build_effective_turn_state(
        inputs.prior_state,
        inputs.initial_state,
        prior_state_wins=True,
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
    crisis_resource_lookup_observed = False
    voice_tool_calls = list(inputs.tool_calls)
    assistant_text = inputs.assistant_text
    route = inputs.route
    response_style = inputs.response_style
    if inputs.outcome in {"connection_interrupted", "safety_interrupted"}:
        voice_tool_calls = [
            call
            for call in voice_tool_calls
            if call.get("status") in {"completed", "failed"}
        ]
        assistant_text = ""
        route = f"voice_{inputs.outcome}"
        response_style = f"voice_{inputs.outcome}"

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
            crisis_resource_lookup_observed = True

    metadata = infer_voice_turn_metadata(
        route=route,
        response_style=response_style,
        tool_calls=voice_tool_calls,
        has_grounded_lookup=bool(grounded_lookup),
    )
    entries = voice_turn_to_transcript_entries(
        user_text=inputs.user_text,
        assistant_text=assistant_text,
        response_style=metadata.response_style,
    )
    if not entries:
        raise ValueError("record_voice_turn requires user_text or assistant_text.")

    grounded_lookup_state = dict(
        cast(Mapping[str, Any], state.get("grounded_lookup", {}) or {})
    )
    diagnostics_state = dict(
        cast(Mapping[str, Any], state.get("diagnostics", {}) or {})
    )
    for key in (
        "voice_turn_outcome",
        "voice_tool_call_outcomes",
        "openai_crisis_tool_calls",
        "voice_crisis_resource_turn_hash",
    ):
        diagnostics_state.pop(key, None)
    session_progress_state = dict(
        cast(Mapping[str, Any], state.get("session_progress", {}) or {})
    )
    state_values = cast(dict[str, Any], state)
    state_values.update(
        {
            "message": inputs.user_text.strip(),
            "channel": Channel.VOICE,
            "user_id": inputs.user_id,
            "session_id": inputs.thread_id,
            "response_text": assistant_text.strip(),
            "response_style": metadata.response_style,
            "route": metadata.route,
            "session_action": "none",
            "should_persist_memory": False,
            "transcript": [*prior_transcript, *entries],
            "grounded_lookup": {**grounded_lookup_state, **grounded_lookup},
            "diagnostics": {
                **diagnostics_state,
                "voice_runtime": "openai_realtime",
                "voice_tool_calls": [
                    str(call.get("tool_name"))
                    for call in voice_tool_calls
                    if isinstance(call, Mapping) and call.get("tool_name")
                ],
            },
            "session_progress": {
                **session_progress_state,
                "turn_count": inputs.prior_turn_count + 1,
                "is_guest": inputs.user_id is None,
            },
        }
    )

    if inputs.outcome in {"connection_interrupted", "safety_interrupted"}:
        diagnostics = dict(state.get("diagnostics", {}) or {})
        diagnostics["voice_turn_outcome"] = inputs.outcome
        diagnostics["voice_tool_call_outcomes"] = [
            {
                "tool_name": str(call.get("tool_name") or ""),
                "status": str(call.get("status") or ""),
            }
            for call in voice_tool_calls
            if call.get("tool_name")
        ]
        if inputs.safety_assessment is not None:
            diagnostics["openai_crisis_tool_calls"] = [
                str(call.get("tool_name"))
                for call in voice_tool_calls
                if call.get("tool_name")
            ]
        state["diagnostics"] = diagnostics
        if inputs.safety_assessment is not None:
            state["crisis"] = inputs.safety_assessment.model_copy(deep=True)
            state["crisis_audit"] = {
                "crisis_override_kind": "none",
                "crisis_classifier_path": "voice_concurrent",
                "crisis_llm_failure_occurred": False,
            }
        _apply_server_crisis_resources(
            state,
            _matching_crisis_resource_state(inputs)
            if crisis_resource_lookup_observed
            else None,
        )
        if (
            inputs.safety_assessment is not None
            and state.get("resource_lookup_status") == "not_attempted"
        ):
            # Overlay resources resolve asynchronously after playback stops.
            state["resource_lookup_status"] = "pending"

    if inputs.outcome == "completed" and metadata.route == "crisis":
        populate_voice_crisis_audit_state(
            state,
            crisis_resource_state=(
                _matching_crisis_resource_state(inputs)
                if crisis_resource_lookup_observed
                else None
            ),
            voice_tool_calls=voice_tool_calls,
        )

    trace_event(
        VOICE_TURN_STATE_BUILT,
        {
            "voice_runtime": "openai_realtime",
            "route": metadata.route,
            "response_style": metadata.response_style,
            "tool_call_count": len(voice_tool_calls),
            "resource_lookup_status": state.get("resource_lookup_status"),
            "crisis_level": getattr(state.get("crisis"), "level", None),
        },
    )
    return VoiceTurnStateResult(state=state, metadata=metadata)


def populate_voice_crisis_audit_state(
    state: AgentState,
    *,
    crisis_resource_state: Mapping[str, Any] | None,
    voice_tool_calls: list[dict[str, Any]],
) -> None:
    """Synthesize crisis audit fields for a voice crisis turn.

    The live Realtime response remains prompt/tool driven: the crisis route is
    inferred from the model calling a crisis tool (see
    ``infer_voice_turn_metadata``). A separate app-owned classifier runs after
    non-crisis voice turns for audit/safety-net purposes, but it intentionally
    does not alter the already-spoken response. This in-turn audit record keeps
    the tool-call signal rather than a classifier verdict. ``level`` is held at
    2 because the live voice route has no independent imminence judgment to
    justify 3, and ``crisis_classifier_path`` is intentionally omitted so
    ``write_crisis_log`` keeps its ``llm_primary`` default rather than inventing
    a new enum value.
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

    _apply_server_crisis_resources(state, crisis_resource_state)

    crisis_tool_names = [
        str(call.get("tool_name"))
        for call in voice_tool_calls
        if isinstance(call, Mapping) and call.get("tool_name")
    ]
    diagnostics = dict(state.get("diagnostics", {}) or {})
    diagnostics["openai_crisis_tool_calls"] = crisis_tool_names
    state["diagnostics"] = diagnostics


def _apply_server_crisis_resources(
    state: AgentState,
    crisis_resource_state: Mapping[str, Any] | None,
) -> None:
    if crisis_resource_state is None:
        return
    status = crisis_resource_state.get("resource_lookup_status")
    if isinstance(status, str) and status:
        state["resource_lookup_status"] = status
    resources = crisis_resource_state.get("found_resources")
    if isinstance(resources, list):
        state["found_resources"] = [
            dict(row) for row in resources if isinstance(row, Mapping)
        ]
    location = crisis_resource_state.get("inferred_location")
    if isinstance(location, str):
        state["inferred_location"] = location


def _matching_crisis_resource_state(
    inputs: VoiceTurnStateInputs,
) -> Mapping[str, Any] | None:
    prior_state = inputs.prior_state
    if prior_state is None:
        return None
    diagnostics = prior_state.get("diagnostics", {})
    marker = (
        diagnostics.get("voice_crisis_resource_turn_hash")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if inputs.correlation_hash is None:
        return prior_state if marker is None else None
    return prior_state if marker == inputs.correlation_hash else None


__all__ = [
    "VoiceTurnStateInputs",
    "VoiceTurnStateResult",
    "build_voice_turn_state",
    "populate_voice_crisis_audit_state",
]
