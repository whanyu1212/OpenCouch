"""Tests for state diagnostics trace exporter."""

from __future__ import annotations

from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.decorators import trace_event, trace_span
from agent.observability.exporters.state import StateDiagnosticsRecorder


def test_state_diagnostics_recorder_exports_bounded_trace_summary() -> None:
    recorder = StateDiagnosticsRecorder(
        max_spans=1, max_events=1, max_attribute_length=8
    )
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("first", attrs={"long": "x" * 20})
    def first() -> None:
        trace_event("event.one", {"value": "y" * 20})

    @trace_span("second")
    def second() -> None:
        trace_event("event.two")

    with use_trace_context(context, recorder):
        first()
        second()

    diagnostics = recorder.to_diagnostics()
    trace = diagnostics["trace"]

    assert len(trace["spans"]) == 1
    assert len(trace["events"]) == 1
    assert trace["spans"][0]["name"] == "first"
    assert trace["spans"][0]["attributes"]["long"] == "xxxxx..."
    assert trace["events"][0]["name"] == "event.one"
    assert trace["events"][0]["attributes"]["value"] == "yyyyy..."
