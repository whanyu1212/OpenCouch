"""Tests for routing trace diagnostics bridge."""

from __future__ import annotations

from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import ROUTING_DECISION
from agent.observability.recorder import InMemoryTraceRecorder
from agent.observability.routing_trace import (
    append_routing_trace,
    routing_trace_from_diagnostics,
)


def test_append_routing_trace_preserves_legacy_return_shape() -> None:
    delta = append_routing_trace(
        None,
        {
            "stage": " safety ",
            "decision": " normal ",
            "reason": " No   crisis\nsignal detected. ",
        },
    )

    assert delta == {
        "routing_trace": [
            {
                "stage": "safety",
                "decision": "normal",
                "reason": "No crisis signal detected.",
            }
        ]
    }


def test_append_routing_trace_ignores_invalid_entries() -> None:
    delta = append_routing_trace(
        {"routing_trace": [{"stage": "safety", "decision": "normal"}]},
        {"stage": "missing-decision"},
    )

    assert delta == {"routing_trace": [{"stage": "safety", "decision": "normal"}]}


def test_routing_trace_from_diagnostics_normalizes_entries() -> None:
    entries = routing_trace_from_diagnostics(
        {
            "routing_trace": [
                {"stage": " safety ", "decision": " normal "},
                {"stage": "invalid"},
                "bad",
            ]
        }
    )

    assert entries == ({"stage": "safety", "decision": "normal"},)


def test_append_routing_trace_emits_privacy_safe_trace_event_when_enabled() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        append_routing_trace(
            None,
            {
                "stage": " safety ",
                "decision": " normal ",
                "reason": " No   crisis\nsignal detected. ",
            },
        )

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.name == ROUTING_DECISION
    assert event.attributes == {
        "stage": "safety",
        "decision": "normal",
    }


def test_append_routing_trace_does_not_emit_event_when_tracing_disabled() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1")

    with use_trace_context(context, recorder):
        delta = append_routing_trace(
            None,
            {"stage": "safety", "decision": "normal"},
        )

    assert delta == {"routing_trace": [{"stage": "safety", "decision": "normal"}]}
    assert recorder.events == []


def test_append_routing_trace_does_not_emit_event_for_invalid_entry() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        append_routing_trace(None, {"stage": "missing-decision"})

    assert recorder.events == []
