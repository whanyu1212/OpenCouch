"""Tests for decorator-style tracing helpers."""

from __future__ import annotations

import pytest

from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.decorators import trace_event, trace_span
from agent.observability.recorder import CompositeTraceRecorder, InMemoryTraceRecorder


def test_trace_span_records_sync_function() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span(
        "test.sync",
        attrs={"kind": "sync", "user_message": "secret"},
        result_attrs=lambda result: {"result_count": len(result)},
    )
    def run() -> list[str]:
        trace_event("test.event", {"value": "ok"})
        return ["ok"]

    with use_trace_context(context, recorder):
        assert run() == ["ok"]

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.name == "test.sync"
    assert span.status == "ok"
    assert span.duration_ms is not None
    assert span.attributes["kind"] == "sync"
    assert span.attributes["user_message"] == "[redacted]"
    assert span.attributes["result_count"] == 1
    assert recorder.events[0].span_id == span.span_id


@pytest.mark.asyncio
async def test_trace_span_records_async_function() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.async")
    async def run() -> str:
        return "ok"

    with use_trace_context(context, recorder):
        assert await run() == "ok"

    assert len(recorder.completed_spans) == 1
    assert recorder.completed_spans[0].name == "test.async"
    assert recorder.completed_spans[0].status == "ok"


def test_trace_span_records_error_and_reraises_original_exception() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.error")
    def run() -> None:
        raise RuntimeError("boom")

    with (
        use_trace_context(context, recorder),
        pytest.raises(RuntimeError, match="boom"),
    ):
        run()

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.status == "error"
    assert span.error_type == "RuntimeError"
    assert span.error_message == "boom"


def test_trace_span_can_suppress_error_message() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.safe_error", record_error_message=False)
    def run() -> None:
        raise RuntimeError("raw user supplied failure detail")

    with (
        use_trace_context(context, recorder),
        pytest.raises(RuntimeError, match="raw user supplied failure detail"),
    ):
        run()

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.status == "error"
    assert span.error_type is None
    assert span.error_message is None
    assert span.attributes["error_type"] == "RuntimeError"


def test_trace_span_is_noop_without_enabled_context() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1")

    @trace_span("test.disabled")
    def run() -> str:
        trace_event("test.event")
        return "ok"

    with use_trace_context(context, recorder):
        assert run() == "ok"

    assert recorder.completed_spans == []
    assert recorder.events == []


def test_nested_trace_spans_record_parent_span_id() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("outer")
    def outer() -> None:
        inner()

    @trace_span("inner")
    def inner() -> None:
        return None

    with use_trace_context(context, recorder):
        outer()

    spans = {span.name: span for span in recorder.completed_spans}
    assert spans["inner"].parent_span_id == spans["outer"].span_id


@pytest.mark.asyncio
async def test_trace_span_records_async_generator_iteration() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.stream")
    async def stream() -> object:
        trace_event("stream.started")
        yield "first"
        trace_event("stream.finished")
        yield "second"

    with use_trace_context(context, recorder):
        assert [item async for item in stream()] == ["first", "second"]

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.name == "test.stream"
    assert span.status == "ok"
    assert [event.name for event in recorder.events] == [
        "stream.started",
        "stream.finished",
    ]
    assert {event.span_id for event in recorder.events} == {span.span_id}


@pytest.mark.asyncio
async def test_trace_span_closes_async_generator_span_on_early_close() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.stream.closed")
    async def stream() -> object:
        trace_event("stream.started")
        yield "first"
        trace_event("stream.unreachable")
        yield "second"

    with use_trace_context(context, recorder):
        generator = stream()
        assert await generator.__anext__() == "first"
        await generator.aclose()

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.name == "test.stream.closed"
    assert span.status == "closed"
    assert [event.name for event in recorder.events] == ["stream.started"]
    assert recorder.events[0].span_id == span.span_id


@pytest.mark.asyncio
async def test_trace_span_records_async_generator_errors() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("test.stream.error")
    async def stream() -> object:
        trace_event("stream.started")
        yield "first"
        raise RuntimeError("stream failed")

    with (
        use_trace_context(context, recorder),
        pytest.raises(RuntimeError, match="stream failed"),
    ):
        _ = [item async for item in stream()]

    assert len(recorder.completed_spans) == 1
    span = recorder.completed_spans[0]
    assert span.status == "error"
    assert span.error_type == "RuntimeError"
    assert recorder.events[0].span_id == span.span_id


def test_composite_recorder_isolates_child_failures() -> None:
    class FailingRecorder(InMemoryTraceRecorder):
        def event(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("telemetry failed")

    good_recorder = InMemoryTraceRecorder()
    recorder = CompositeTraceRecorder([FailingRecorder(), good_recorder])
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        trace_event("test.event", {"value": "ok"})

    assert len(good_recorder.events) == 1
    assert good_recorder.events[0].name == "test.event"


def test_composite_recorder_preserves_child_span_ids() -> None:
    first_recorder = InMemoryTraceRecorder()
    second_recorder = InMemoryTraceRecorder()
    recorder = CompositeTraceRecorder([first_recorder, second_recorder])
    context = TraceContext(trace_id="trace-1", config=TraceConfig(enabled=True))

    @trace_span("outer")
    def outer() -> None:
        trace_event("outer.event")
        inner()

    @trace_span("inner")
    def inner() -> None:
        trace_event("inner.event")

    with use_trace_context(context, recorder):
        outer()

    first_spans = {span.name: span for span in first_recorder.completed_spans}
    second_spans = {span.name: span for span in second_recorder.completed_spans}

    assert first_recorder.events[0].span_id == first_spans["outer"].span_id
    assert second_recorder.events[0].span_id == second_spans["outer"].span_id
    assert first_recorder.events[1].span_id == first_spans["inner"].span_id
    assert second_recorder.events[1].span_id == second_spans["inner"].span_id
    assert first_spans["inner"].parent_span_id == first_spans["outer"].span_id
    assert second_spans["inner"].parent_span_id == second_spans["outer"].span_id
