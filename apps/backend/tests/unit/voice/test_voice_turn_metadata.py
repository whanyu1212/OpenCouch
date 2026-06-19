"""Unit tests for post-turn voice route inference (``infer_voice_turn_metadata``).

This classifier runs at audit/metadata time (``state_transition.build_voice_turn_state``)
AFTER the realtime model has already responded, not on the speech-to-speech audio
path, so it adds no turn latency. It decides which persisted route a voice turn
gets — and crucially, whether the always-on crisis audit log is written.
"""

from __future__ import annotations

from agent.voice.turn_metadata import infer_voice_turn_metadata


def _tool(name: str) -> dict[str, str]:
    return {"tool_name": name}


def test_lookup_crisis_resources_routes_to_crisis() -> None:
    metadata = infer_voice_turn_metadata(
        route=None,
        response_style=None,
        tool_calls=[_tool("lookup_crisis_resources")],
        has_grounded_lookup=False,
    )

    assert metadata.route == "crisis"
    assert metadata.response_style == "crisis_response"


def test_crisis_support_template_only_turn_routes_to_crisis() -> None:
    # Regression for #157: the model is instructed to call get_crisis_support_template
    # independently of lookup_crisis_resources. A crisis turn that calls ONLY the
    # template tool must still route to "crisis" so populate_voice_crisis_audit_state
    # runs and record_crisis_outcome writes the audit record. Previously this tool was
    # absent from _TOOL_ROUTE_PRIORITY, so the turn fell through to "therapeutic" and
    # was never audit-logged.
    metadata = infer_voice_turn_metadata(
        route=None,
        response_style=None,
        tool_calls=[_tool("get_crisis_support_template")],
        has_grounded_lookup=False,
    )

    assert metadata.route == "crisis"
    assert metadata.response_style == "crisis_response"


def test_ordinary_therapeutic_turn_is_not_misrouted_to_crisis() -> None:
    # Guard the other direction: a turn with no crisis tool must stay therapeutic,
    # so the change does not over-trigger the crisis route / audit log.
    metadata = infer_voice_turn_metadata(
        route=None,
        response_style=None,
        tool_calls=[_tool("show_memory_status")],
        has_grounded_lookup=False,
    )

    assert metadata.route == "memory_control"


def test_explicit_route_short_circuits_tool_inference() -> None:
    # When the caller already supplies route + style, tool inference is bypassed.
    metadata = infer_voice_turn_metadata(
        route="therapeutic",
        response_style="voice",
        tool_calls=[_tool("get_crisis_support_template")],
        has_grounded_lookup=False,
    )

    assert metadata.route == "therapeutic"
    assert metadata.response_style == "voice"
