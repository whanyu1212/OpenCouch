"""Route metadata inferred from observed Realtime voice tool use."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.voice.tools.specs import VOICE_TOOL_SPECS


@dataclass(frozen=True)
class VoiceTurnMetadata:
    route: str
    response_style: str


# Either crisis tool forces the crisis route so the turn is audit-logged.
# The model is instructed to call get_crisis_support_template independently of
# lookup_crisis_resources, so a template-only crisis turn must still route to
# crisis (otherwise it falls through to "therapeutic" and is never recorded).
_TOOL_ROUTE_PRIORITY: tuple[tuple[str, VoiceTurnMetadata], ...] = tuple(
    (
        spec.name,
        VoiceTurnMetadata(
            route=spec.route,
            response_style=spec.response_style,
        ),
    )
    for spec in sorted(
        (
            spec
            for spec in VOICE_TOOL_SPECS
            if spec.route is not None and spec.response_style is not None
        ),
        key=lambda spec: spec.route_priority if spec.route_priority is not None else 0,
    )
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
