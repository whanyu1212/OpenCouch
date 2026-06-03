"""Bounded AgentState diagnostics exporter for trace records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.observability.recorder import (
    InMemoryTraceRecorder,
    SpanHandle,
    TraceEventRecord,
    TraceSpanRecord,
)
from agent.observability.redaction import sanitize_attributes

TRACE_DIAGNOSTICS_KEY = "trace"


class StateDiagnosticsRecorder(InMemoryTraceRecorder):
    """Recorder that exposes compact diagnostics for AgentState."""

    def __init__(
        self,
        *,
        max_spans: int = 25,
        max_events: int = 50,
        max_attribute_length: int = 120,
    ) -> None:
        super().__init__()
        self._max_spans = max_spans
        self._max_events = max_events
        self._max_attribute_length = max_attribute_length

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record a bounded sanitized event."""

        if len(self.events) >= self._max_events:
            return
        self.events.append(
            TraceEventRecord(
                name=name,
                attributes=sanitize_attributes(
                    attributes,
                    max_string_length=self._max_attribute_length,
                ),
                span_id=span_id,
            )
        )

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start a span using the base in-memory recorder."""

        return super().start_span(
            name,
            sanitize_attributes(
                attributes, max_string_length=self._max_attribute_length
            ),
            parent_span_id=parent_span_id,
        )

    def _complete_span(self, record: TraceSpanRecord) -> None:
        if len(self.completed_spans) >= self._max_spans:
            return
        record.attributes = sanitize_attributes(
            record.attributes,
            max_string_length=self._max_attribute_length,
        )
        self.completed_spans.append(record)

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a compact diagnostics payload suitable for AgentState."""

        return {
            TRACE_DIAGNOSTICS_KEY: {
                "spans": [
                    {
                        "name": span.name,
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id,
                        "status": span.status,
                        "duration_ms": span.duration_ms,
                        "attributes": span.attributes,
                        "error_type": span.error_type,
                    }
                    for span in self.completed_spans[: self._max_spans]
                ],
                "events": [
                    {
                        "name": event.name,
                        "span_id": event.span_id,
                        "attributes": event.attributes,
                    }
                    for event in self.events[: self._max_events]
                ],
            }
        }
