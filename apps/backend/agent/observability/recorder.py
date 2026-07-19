"""Vendor-neutral trace recorder primitives for agent observability."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.observability.redaction import sanitize_attributes


@dataclass(slots=True)
class TraceEventRecord:
    """One semantic trace event."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    span_id: str | None = None


@dataclass(slots=True)
class TraceSpanRecord:
    """One completed trace span."""

    name: str
    span_id: str
    parent_span_id: str | None
    attributes: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    status: str = "running"
    error_type: str | None = None
    error_message: str | None = None
    events: list[TraceEventRecord] = field(default_factory=list)


class SpanHandle(Protocol):
    """Mutable handle returned by trace recorders for active spans."""

    @property
    def span_id(self) -> str:
        """Return this span's identifier."""

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach an event to this active span."""

    def end(
        self,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Complete this active span."""


class TraceRecorder(Protocol):
    """Vendor-neutral recorder interface."""

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start a trace span."""

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record a semantic trace event."""

    def error(
        self,
        name: str,
        error: BaseException,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record an error event."""


class NoopSpanHandle:
    """Span handle that intentionally records nothing."""

    span_id = ""

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Ignore an event."""

    def end(
        self,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Ignore span completion."""


class NoopTraceRecorder:
    """Recorder used when tracing is disabled."""

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Return a no-op span handle."""

        return NoopSpanHandle()

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Ignore an event."""

    def error(
        self,
        name: str,
        error: BaseException,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Ignore an error."""


class InMemorySpanHandle:
    """Active span handle for ``InMemoryTraceRecorder``."""

    def __init__(
        self, recorder: InMemoryTraceRecorder, record: TraceSpanRecord
    ) -> None:
        self._recorder = recorder
        self._record = record

    @property
    def span_id(self) -> str:
        """Return this span's identifier."""

        return self._record.span_id

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach a sanitized event to this span."""

        self._record.events.append(
            TraceEventRecord(
                name=name,
                attributes=sanitize_attributes(attributes),
                span_id=self._record.span_id,
            )
        )

    def end(
        self,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Complete this span."""

        if self._record.ended_at is not None:
            return

        if attributes:
            self._record.attributes.update(sanitize_attributes(attributes))
        ended_at = time.time()
        self._record.ended_at = ended_at
        self._record.duration_ms = round((ended_at - self._record.started_at) * 1000, 2)
        self._record.status = status
        if error is not None:
            self._record.status = "error"
            self._record.error_type = type(error).__name__
            self._record.error_message = sanitize_attributes(
                {"message": str(error)}
            ).get("message")
        self._recorder._complete_span(self._record)


class InMemoryTraceRecorder:
    """Recorder useful for tests and local inspection."""

    def __init__(self) -> None:
        self.completed_spans: list[TraceSpanRecord] = []
        self.events: list[TraceEventRecord] = []
        self.errors: list[TraceEventRecord] = []

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start an in-memory span."""

        record = TraceSpanRecord(
            name=name,
            span_id=uuid.uuid4().hex,
            parent_span_id=parent_span_id,
            attributes=sanitize_attributes(attributes),
            started_at=time.time(),
        )
        return InMemorySpanHandle(self, record)

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record a sanitized event."""

        self.events.append(
            TraceEventRecord(
                name=name,
                attributes=sanitize_attributes(attributes),
                span_id=span_id,
            )
        )

    def error(
        self,
        name: str,
        error: BaseException,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Record a sanitized error event."""

        payload = dict(attributes or {})
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        self.errors.append(
            TraceEventRecord(
                name=name,
                attributes=sanitize_attributes(payload),
                span_id=span_id,
            )
        )

    def _complete_span(self, record: TraceSpanRecord) -> None:
        self.completed_spans.append(record)


class CompositeTraceRecorder:
    """Recorder that forwards operations to multiple recorders safely."""

    def __init__(self, recorders: Sequence[TraceRecorder]) -> None:
        self._recorders = tuple(recorders)

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        parent_span_id: str | None = None,
    ) -> SpanHandle:
        """Start a composite span across all child recorders."""

        handles: list[tuple[int, SpanHandle]] = []
        parent_span_ids = _decode_composite_span_id(parent_span_id)
        for index, recorder in enumerate(self._recorders):
            try:
                child_parent_span_id = (
                    parent_span_ids[index]
                    if parent_span_ids is not None and index < len(parent_span_ids)
                    else parent_span_id
                )
                handles.append(
                    (
                        index,
                        recorder.start_span(
                            name,
                            attributes,
                            parent_span_id=child_parent_span_id,
                        ),
                    )
                )
            except Exception:
                continue
        return CompositeSpanHandle(handles)

    def event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Forward an event to all child recorders."""

        span_ids = _decode_composite_span_id(span_id)
        for index, recorder in enumerate(self._recorders):
            try:
                child_span_id = (
                    span_ids[index]
                    if span_ids is not None and index < len(span_ids)
                    else span_id
                )
                recorder.event(name, attributes, span_id=child_span_id)
            except Exception:
                continue

    def error(
        self,
        name: str,
        error: BaseException,
        attributes: Mapping[str, Any] | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        """Forward an error to all child recorders."""

        span_ids = _decode_composite_span_id(span_id)
        for index, recorder in enumerate(self._recorders):
            try:
                child_span_id = (
                    span_ids[index]
                    if span_ids is not None and index < len(span_ids)
                    else span_id
                )
                recorder.error(name, error, attributes, span_id=child_span_id)
            except Exception:
                continue


class CompositeSpanHandle:
    """Span handle that completes multiple child spans safely."""

    def __init__(self, handles: Sequence[tuple[int, SpanHandle]]) -> None:
        self._handles = tuple(handles)

    @property
    def span_id(self) -> str:
        """Return a composite span id carrying each child span id."""

        if not self._handles:
            return ""
        span_ids = ["" for _ in range(max(index for index, _ in self._handles) + 1)]
        for index, handle in self._handles:
            span_ids[index] = handle.span_id
        return _encode_composite_span_id(span_ids)

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Forward an event to all child spans."""

        for _, handle in self._handles:
            try:
                handle.add_event(name, attributes)
            except Exception:
                continue

    def end(
        self,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Complete all child spans."""

        for _, handle in self._handles:
            try:
                handle.end(status=status, error=error, attributes=attributes)
            except Exception:
                continue


_COMPOSITE_SPAN_PREFIX = "composite:"


def _encode_composite_span_id(span_ids: Sequence[str]) -> str:
    return f"{_COMPOSITE_SPAN_PREFIX}{'|'.join(span_ids)}"


def _decode_composite_span_id(span_id: str | None) -> tuple[str, ...] | None:
    if not span_id or not span_id.startswith(_COMPOSITE_SPAN_PREFIX):
        return None
    return tuple(span_id.removeprefix(_COMPOSITE_SPAN_PREFIX).split("|"))
