"""Structured logging exporter for agent trace records."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent.observability.recorder import (
    InMemoryTraceRecorder,
    SpanHandle,
    TraceSpanRecord,
)
from agent.observability.redaction import sanitize_attributes

logger = logging.getLogger(__name__)


class StructuredLogRecorder(InMemoryTraceRecorder):
    """Recorder that emits sanitized span/event summaries to Python logging."""

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start a span and keep the in-memory summary for completion logging."""

        return super().start_span(
            name,
            sanitize_attributes(attributes),
            parent_span_id=parent_span_id,
        )

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Log a sanitized event."""

        payload = {
            "type": "trace_event",
            "name": name,
            "span_id": span_id,
            "attributes": sanitize_attributes(attributes),
        }
        logger.debug("agent trace event", extra={"agent_trace": payload})
        super().event(name, attributes, span_id=span_id)

    def _complete_span(self, record: TraceSpanRecord) -> None:
        super()._complete_span(record)
        payload = {
            "type": "trace_span",
            "name": record.name,
            "span_id": record.span_id,
            "parent_span_id": record.parent_span_id,
            "status": record.status,
            "duration_ms": record.duration_ms,
            "attributes": record.attributes,
            "error_type": record.error_type,
        }
        logger.debug("agent trace span", extra={"agent_trace": payload})
