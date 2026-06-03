"""Factory helpers for named trace exporter recorders."""

from __future__ import annotations

from agent.observability.config import TraceConfig
from agent.observability.exporters.logging import StructuredLogRecorder
from agent.observability.exporters.opentelemetry import OpenTelemetryTraceRecorder
from agent.observability.exporters.state import StateDiagnosticsRecorder
from agent.observability.recorder import (
    CompositeTraceRecorder,
    NoopTraceRecorder,
    TraceRecorder,
)


def build_trace_recorder(config: TraceConfig) -> TraceRecorder:
    """Build a trace recorder from named exporters in ``TraceConfig``."""

    if not config.enabled or not config.exporters:
        return NoopTraceRecorder()

    recorders: list[TraceRecorder] = []
    for exporter in config.exporters:
        recorders.append(_build_named_recorder(exporter))

    if len(recorders) == 1:
        return recorders[0]
    return CompositeTraceRecorder(recorders)


def _build_named_recorder(name: str) -> TraceRecorder:
    normalized = name.strip().lower()
    if normalized == "logging":
        return StructuredLogRecorder()
    if normalized == "state":
        return StateDiagnosticsRecorder()
    if normalized in {"opentelemetry", "otel", "otlp"}:
        return OpenTelemetryTraceRecorder()
    msg = f"Unknown trace exporter: {name!r}"
    raise ValueError(msg)


__all__ = ["build_trace_recorder"]
