"""Tests for the OpenTelemetry trace exporter."""

from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.decorators import trace_event, trace_span
from agent.observability.exporters.opentelemetry import OpenTelemetryTraceRecorder
from agent.observability.recorder import CompositeTraceRecorder, InMemoryTraceRecorder


def _recorder_with_exporter() -> tuple[
    OpenTelemetryTraceRecorder, InMemorySpanExporter
]:
    exporter = InMemorySpanExporter()
    return OpenTelemetryTraceRecorder(span_exporter=exporter), exporter


def test_opentelemetry_recorder_exports_completed_span() -> None:
    recorder, exporter = _recorder_with_exporter()

    span = recorder.start_span(
        "test.span",
        {
            "value": "ok",
            "user_message": "private",
            "nested": {"payload": "private"},
        },
    )
    span.end(attributes={"result_count": 1})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    exported = spans[0]
    assert exported.name == "test.span"
    assert exported.attributes["value"] == "ok"
    assert exported.attributes["user_message"] == "[redacted]"
    assert exported.attributes["nested"] == "[complex]"
    assert exported.attributes["result_count"] == 1
    assert exported.status.status_code == StatusCode.OK


def test_opentelemetry_recorder_preserves_parent_child_relationship() -> None:
    recorder, exporter = _recorder_with_exporter()

    parent = recorder.start_span("parent")
    child = recorder.start_span("child", parent_span_id=parent.span_id)
    child.end()
    parent.end()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["child"].parent is not None
    assert spans["child"].parent.span_id == spans["parent"].context.span_id


def test_opentelemetry_recorder_adds_events_to_active_span() -> None:
    recorder, exporter = _recorder_with_exporter()

    span = recorder.start_span("with.event")
    recorder.event(
        "test.event",
        {"value": "ok", "token": "redacted-test-value"},
        span_id=span.span_id,
    )
    span.end()

    exported = exporter.get_finished_spans()[0]
    assert len(exported.events) == 1
    event = exported.events[0]
    assert event.name == "test.event"
    assert event.attributes["value"] == "ok"
    assert event.attributes["token"] == "[redacted]"


def test_opentelemetry_recorder_exports_standalone_event_span() -> None:
    recorder, exporter = _recorder_with_exporter()

    recorder.event("orphan.event", {"count": 1})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    exported = spans[0]
    assert exported.name == "event.orphan.event"
    assert exported.attributes["agent.event.name"] == "orphan.event"
    assert exported.attributes["agent.event.standalone"] is True
    assert exported.attributes["count"] == 1


def test_opentelemetry_recorder_does_not_export_raw_error_message() -> None:
    recorder, exporter = _recorder_with_exporter()

    span = recorder.start_span("error.span")
    span.end(error=RuntimeError("private failure detail"))

    exported = exporter.get_finished_spans()[0]
    assert exported.status.status_code == StatusCode.ERROR
    assert exported.attributes["error.type"] == "RuntimeError"
    assert "private failure detail" not in str(exported.attributes)
    assert "private failure detail" not in str(exported.events)


def test_opentelemetry_span_handle_ignores_double_end() -> None:
    recorder, exporter = _recorder_with_exporter()

    span = recorder.start_span("single.end")
    span.end()
    span.end()

    assert len(exporter.get_finished_spans()) == 1


def test_opentelemetry_recorder_works_with_composite_span_ids() -> None:
    otel_recorder, exporter = _recorder_with_exporter()
    memory_recorder = InMemoryTraceRecorder()
    recorder = CompositeTraceRecorder([memory_recorder, otel_recorder])

    @trace_span("outer")
    def outer() -> None:
        trace_event("outer.event")
        inner()

    @trace_span("inner")
    def inner() -> None:
        trace_event("inner.event")

    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))
    with use_trace_context(context, recorder):
        outer()

    memory_spans = {span.name: span for span in memory_recorder.completed_spans}
    otel_spans = {span.name: span for span in exporter.get_finished_spans()}

    assert memory_recorder.events[0].span_id == memory_spans["outer"].span_id
    assert memory_recorder.events[1].span_id == memory_spans["inner"].span_id
    assert memory_spans["inner"].parent_span_id == memory_spans["outer"].span_id
    assert otel_spans["inner"].parent is not None
    assert otel_spans["inner"].parent.span_id == otel_spans["outer"].context.span_id
