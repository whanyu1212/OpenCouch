"""Route metadata inferred from observed Realtime voice tool use."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceTurnMetadata:
    route: str
    response_style: str


_MEMORY_TOOL_NAMES = {
    "show_memory_status",
    "show_saved_memory",
    "recall_saved_memory",
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
}

_GUIDED_EXERCISE_TOOL_NAMES = {
    "list_guided_exercise_skills",
    "load_guided_exercise_skill",
    "record_guided_exercise_progress",
}

_TOOL_ROUTE_PRIORITY: tuple[tuple[str, VoiceTurnMetadata], ...] = (
    ("lookup_crisis_resources", VoiceTurnMetadata("crisis", "crisis_response")),
    ("answer_grounded_lookup", VoiceTurnMetadata("grounded_lookup", "grounded_lookup")),
    *(
        (name, VoiceTurnMetadata("memory_control", "memory_control"))
        for name in sorted(_MEMORY_TOOL_NAMES)
    ),
    *(
        (name, VoiceTurnMetadata("guided_exercise", "guided_exercise"))
        for name in sorted(_GUIDED_EXERCISE_TOOL_NAMES)
    ),
)


def infer_voice_turn_metadata(
    *,
    route: str | None,
    response_style: str | None,
    tool_calls: list[dict[str, Any]],
    has_grounded_lookup: bool,
) -> VoiceTurnMetadata:
    """Resolve persisted voice route/style without a pre-response policy call."""

    normalized_route = _clean(route)
    normalized_style = _clean(response_style)
    if normalized_route and normalized_style:
        return VoiceTurnMetadata(normalized_route, normalized_style)

    tool_names = {
        str(call.get("tool_name") or "")
        for call in tool_calls
        if isinstance(call, Mapping)
    }
    for tool_name, metadata in _TOOL_ROUTE_PRIORITY:
        if tool_name in tool_names:
            return VoiceTurnMetadata(
                route=normalized_route or metadata.route,
                response_style=normalized_style or metadata.response_style,
            )

    if has_grounded_lookup:
        return VoiceTurnMetadata(
            route=normalized_route or "grounded_lookup",
            response_style=normalized_style or "grounded_lookup",
        )

    return VoiceTurnMetadata(
        route=normalized_route or "therapeutic",
        response_style=normalized_style or "voice",
    )


def _clean(value: str | None) -> str:
    return str(value or "").strip()


__all__ = ["VoiceTurnMetadata", "infer_voice_turn_metadata"]
