"""Trace context propagation for agent observability."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from agent.observability.config import TraceConfig
from agent.observability.recorder import NoopTraceRecorder, TraceRecorder


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Identity and runtime controls for one traceable session or turn."""

    trace_id: str
    session_id: str | None = None
    turn_id: str | None = None
    runtime_mode: Literal["text", "voice"] | None = None
    user_id_hash: str | None = None
    config: TraceConfig = TraceConfig()

    @property
    def enabled(self) -> bool:
        """Return whether this context should emit trace data."""

        return self.config.enabled


_current_trace_context: ContextVar[TraceContext | None] = ContextVar(
    "agent_observability_trace_context",
    default=None,
)
_current_trace_recorder: ContextVar[TraceRecorder | None] = ContextVar(
    "agent_observability_trace_recorder",
    default=None,
)
_current_span_id: ContextVar[str | None] = ContextVar(
    "agent_observability_current_span_id",
    default=None,
)


def get_current_trace_context() -> TraceContext | None:
    """Return the active trace context, if any."""

    return _current_trace_context.get()


def get_current_trace_recorder() -> TraceRecorder:
    """Return the active recorder or a no-op recorder when tracing is unavailable."""

    context = _current_trace_context.get()
    recorder = _current_trace_recorder.get()
    if context is None or not context.enabled or recorder is None:
        return NoopTraceRecorder()
    return recorder


def get_current_span_id() -> str | None:
    """Return the active parent span identifier, if any."""

    return _current_span_id.get()


@contextmanager
def use_trace_context(
    context: TraceContext,
    recorder: TraceRecorder | None = None,
) -> Iterator[None]:
    """Set trace context and recorder for the current async context."""

    context_token = _current_trace_context.set(context)
    recorder_token = _current_trace_recorder.set(recorder)
    span_token = _current_span_id.set(None)
    try:
        yield
    finally:
        _current_span_id.reset(span_token)
        _current_trace_recorder.reset(recorder_token)
        _current_trace_context.reset(context_token)


@contextmanager
def use_parent_span(span_id: str | None) -> Iterator[None]:
    """Temporarily set the active parent span identifier."""

    token = _current_span_id.set(span_id)
    try:
        yield
    finally:
        _current_span_id.reset(token)
