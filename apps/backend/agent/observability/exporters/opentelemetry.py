"""OpenTelemetry exporter adapter for agent trace records."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode

from agent.observability.recorder import SpanHandle
from agent.observability.redaction import sanitize_attributes


_DEFAULT_SERVICE_NAME = "opencouch-agent"


class OpenTelemetryTraceRecorder:
    """Trace recorder that exports agent spans to OpenTelemetry."""

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        span_exporter: SpanExporter | None = None,
        service_name: str = _DEFAULT_SERVICE_NAME,
        tracer_name: str = "opencouch.agent",
        enable_default_otlp_exporter: bool = True,
    ) -> None:
        """Initialize the OpenTelemetry recorder.

        Args:
            tracer_provider: Optional SDK tracer provider. Tests can inject a
                provider with an in-memory exporter.
            span_exporter: Optional exporter to attach with a simple processor.
            service_name: Resource service name when creating a provider.
            tracer_name: Instrumentation scope name.
            enable_default_otlp_exporter: Attach an OTLP/HTTP exporter when no
                provider or explicit exporter is supplied.
        """

        self._owns_provider = tracer_provider is None
        if tracer_provider is None:
            tracer_provider = TracerProvider(
                resource=Resource.create({"service.name": service_name})
            )
            if span_exporter is not None:
                tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
            elif enable_default_otlp_exporter:
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter())
                )
        elif span_exporter is not None:
            tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        self._tracer_provider = tracer_provider
        self._tracer = tracer_provider.get_tracer(tracer_name)
        self._active_spans: dict[str, Span] = {}

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start an OpenTelemetry span and return an agent span handle."""

        span_id = uuid.uuid4().hex
        parent_span = self._active_spans.get(parent_span_id or "")
        parent_context = (
            trace.set_span_in_context(parent_span) if parent_span is not None else None
        )
        span = self._tracer.start_span(
            name,
            context=parent_context,
            attributes=_otel_attributes(attributes),
            start_time=_time_ns(),
        )
        self._active_spans[span_id] = span
        return OpenTelemetrySpanHandle(self, span_id, span)

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record a sanitized event on an active span or as a standalone span."""

        sanitized = _otel_attributes(attributes)
        span = self._active_spans.get(span_id or "")
        if span is not None:
            span.add_event(name, sanitized, timestamp=_time_ns())
            return

        standalone = self._tracer.start_span(
            f"event.{name}",
            attributes={
                "agent.event.name": name,
                "agent.event.standalone": True,
                **sanitized,
            },
            start_time=_time_ns(),
        )
        standalone.end(end_time=_time_ns())

    def error(
        self,
        name: str,
        error: BaseException,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record an error event without exporting raw exception messages."""

        payload = {
            "error.type": type(error).__name__,
            **_otel_attributes(attributes),
        }
        self.event(name, payload, span_id=span_id)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush pending spans when the underlying provider supports it."""

        return bool(self._tracer_provider.force_flush(timeout_millis=timeout_millis))

    def shutdown(self) -> None:
        """Shut down the owned tracer provider."""

        if self._owns_provider:
            self._tracer_provider.shutdown()

    def _end_span(
        self,
        span_id: str,
        span: Span,
        *,
        status: str,
        error: BaseException | None,
        attributes: Mapping[str, Any] | None,
    ) -> None:
        sanitized = _otel_attributes(attributes)
        if sanitized:
            span.set_attributes(sanitized)

        if error is not None:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR))
        elif status == "error":
            span.set_status(Status(StatusCode.ERROR))
        elif status == "ok":
            span.set_status(Status(StatusCode.OK))
        elif status == "closed":
            span.set_attribute("agent.span.closed", True)

        self._active_spans.pop(span_id, None)
        span.end(end_time=_time_ns())


class OpenTelemetrySpanHandle:
    """Mutable handle for one OpenTelemetry span."""

    def __init__(
        self,
        recorder: OpenTelemetryTraceRecorder,
        span_id: str,
        span: Span,
    ) -> None:
        self._recorder = recorder
        self._span_id = span_id
        self._span = span
        self._ended = False

    @property
    def span_id(self) -> str:
        """Return the agent-local span identifier."""

        return self._span_id

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach a sanitized event to this span."""

        if self._ended:
            return
        self._span.add_event(
            name,
            _otel_attributes(attributes),
            timestamp=_time_ns(),
        )

    def end(
        self,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Complete this span exactly once."""

        if self._ended:
            return
        self._ended = True
        self._recorder._end_span(
            self._span_id,
            self._span,
            status=status,
            error=error,
            attributes=attributes,
        )


def _otel_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    sanitized = sanitize_attributes(attributes)
    return {
        key: _otel_attribute_value(value)
        for key, value in sanitized.items()
        if value is not None
    }


def _otel_attribute_value(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        primitive_values = [
            item for item in value if isinstance(item, str | bool | int | float)
        ]
        if len(primitive_values) == len(value):
            return primitive_values
    return "[complex]"


def _time_ns() -> int:
    return int(time.time() * 1_000_000_000)


__all__ = [
    "OpenTelemetrySpanHandle",
    "OpenTelemetryTraceRecorder",
]
