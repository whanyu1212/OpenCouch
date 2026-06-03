"""Tests for trace context propagation."""

from __future__ import annotations

from agent.observability.config import TraceConfig
from agent.observability.context import (
    TraceContext,
    get_current_span_id,
    get_current_trace_context,
    get_current_trace_recorder,
    use_parent_span,
    use_trace_context,
)
from agent.observability.recorder import InMemoryTraceRecorder, NoopTraceRecorder


def test_trace_context_is_absent_by_default() -> None:
    assert get_current_trace_context() is None
    assert get_current_span_id() is None
    assert isinstance(get_current_trace_recorder(), NoopTraceRecorder)


def test_use_trace_context_sets_and_restores_context() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(
        trace_id="trace-1",
        session_id="session-1",
        config=TraceConfig(enabled=True),
    )

    with use_trace_context(context, recorder):
        assert get_current_trace_context() == context
        assert get_current_trace_recorder() is recorder

    assert get_current_trace_context() is None
    assert isinstance(get_current_trace_recorder(), NoopTraceRecorder)


def test_disabled_trace_context_uses_noop_recorder() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-1")

    with use_trace_context(context, recorder):
        assert isinstance(get_current_trace_recorder(), NoopTraceRecorder)


def test_nested_context_and_parent_span_are_restored() -> None:
    outer = TraceContext(trace_id="outer", config=TraceConfig(enabled=True))
    inner = TraceContext(trace_id="inner", config=TraceConfig(enabled=True))

    with use_trace_context(outer, InMemoryTraceRecorder()):
        with use_parent_span("outer-span"):
            assert get_current_span_id() == "outer-span"
            with use_trace_context(inner, InMemoryTraceRecorder()):
                assert get_current_trace_context() == inner
                assert get_current_span_id() is None
            assert get_current_trace_context() == outer
            assert get_current_span_id() == "outer-span"
