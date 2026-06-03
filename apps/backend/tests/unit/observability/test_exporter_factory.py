"""Tests for trace exporter factory helpers."""

from __future__ import annotations

import pytest

from agent.observability.config import TraceConfig
from agent.observability.exporters.factory import build_trace_recorder
from agent.observability.exporters.logging import StructuredLogRecorder
from agent.observability.exporters.state import StateDiagnosticsRecorder
from agent.observability.recorder import (
    CompositeTraceRecorder,
    InMemoryTraceRecorder,
    NoopTraceRecorder,
)


def test_build_trace_recorder_returns_noop_when_disabled() -> None:
    recorder = build_trace_recorder(
        TraceConfig(enabled=False, exporters=("logging",)),
    )

    assert isinstance(recorder, NoopTraceRecorder)


def test_build_trace_recorder_returns_noop_without_exporters() -> None:
    recorder = build_trace_recorder(TraceConfig(enabled=True))

    assert isinstance(recorder, NoopTraceRecorder)


def test_build_trace_recorder_builds_named_recorders() -> None:
    assert isinstance(
        build_trace_recorder(TraceConfig(enabled=True, exporters=("logging",))),
        StructuredLogRecorder,
    )
    assert isinstance(
        build_trace_recorder(TraceConfig(enabled=True, exporters=("state",))),
        StateDiagnosticsRecorder,
    )


def test_build_trace_recorder_aliases_opentelemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.observability.exporters.factory.OpenTelemetryTraceRecorder",
        InMemoryTraceRecorder,
    )

    recorder = build_trace_recorder(TraceConfig(enabled=True, exporters=("otlp",)))

    assert isinstance(recorder, InMemoryTraceRecorder)


def test_build_trace_recorder_composes_multiple_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.observability.exporters.factory.OpenTelemetryTraceRecorder",
        InMemoryTraceRecorder,
    )

    recorder = build_trace_recorder(
        TraceConfig(enabled=True, exporters=("state", "otlp")),
    )

    assert isinstance(recorder, CompositeTraceRecorder)


def test_build_trace_recorder_rejects_unknown_exporter() -> None:
    with pytest.raises(ValueError, match="Unknown trace exporter"):
        build_trace_recorder(TraceConfig(enabled=True, exporters=("missing",)))
