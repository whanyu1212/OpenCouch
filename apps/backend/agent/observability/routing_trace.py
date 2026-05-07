"""Structured routing trace helpers for graph diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

ROUTING_TRACE_KEY = "routing_trace"


class RoutingTraceEntry(TypedDict, total=False):
    """One compact routing decision for CLI/API observability."""

    stage: str
    decision: str
    source: str
    reason: str
    confidence: str


def append_routing_trace(
    existing_diagnostics: Mapping[str, Any] | None,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a diagnostics delta with one routing-trace entry appended.

    Args:
        existing_diagnostics (Mapping[str, Any] | None): Current diagnostics
            mapping from state. Only the existing routing trace is read.
        entry (Mapping[str, Any]): New routing trace entry.

    Returns:
        dict[str, Any]: Diagnostics delta containing only ``routing_trace``.
    """

    trace: list[RoutingTraceEntry] = []
    existing_trace = (existing_diagnostics or {}).get(ROUTING_TRACE_KEY)
    if isinstance(existing_trace, list):
        for item in existing_trace:
            normalized = _normalize_trace_entry(item)
            if normalized is not None:
                trace.append(normalized)

    normalized_entry = _normalize_trace_entry(entry)
    if normalized_entry is not None:
        trace.append(normalized_entry)

    return {ROUTING_TRACE_KEY: trace}


def routing_trace_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> tuple[RoutingTraceEntry, ...]:
    """Return normalized routing trace entries from diagnostics.

    Args:
        diagnostics (Mapping[str, Any] | None): Agent diagnostics mapping.

    Returns:
        tuple[RoutingTraceEntry, ...]: Clean routing trace entries.
    """

    raw_trace = (diagnostics or {}).get(ROUTING_TRACE_KEY)
    if not isinstance(raw_trace, list):
        return ()

    entries: list[RoutingTraceEntry] = []
    for item in raw_trace:
        normalized = _normalize_trace_entry(item)
        if normalized is not None:
            entries.append(normalized)
    return tuple(entries)


def _normalize_trace_entry(value: Any) -> RoutingTraceEntry | None:
    """Normalize one routing trace item.

    Args:
        value (Any): Raw trace entry.

    Returns:
        RoutingTraceEntry | None: Clean entry, or None when required fields
            are absent.
    """

    if not isinstance(value, Mapping):
        return None

    stage = _clean_trace_value(value.get("stage"), max_length=48)
    decision = _clean_trace_value(value.get("decision"), max_length=64)
    if not stage or not decision:
        return None

    entry: RoutingTraceEntry = {
        "stage": stage,
        "decision": decision,
    }
    source = _clean_trace_value(value.get("source"), max_length=64)
    reason = _clean_trace_value(value.get("reason"), max_length=180)
    confidence = _clean_trace_value(value.get("confidence"), max_length=24)
    if source:
        entry["source"] = source
    if reason:
        entry["reason"] = reason
    if confidence:
        entry["confidence"] = confidence
    return entry


def _clean_trace_value(value: Any, *, max_length: int) -> str:
    """Return a compact single-line string for routing trace display.

    Args:
        value (Any): Raw value.
        max_length (int): Maximum display length.

    Returns:
        str: Cleaned string, possibly truncated.
    """

    if value is None:
        return ""
    cleaned = " ".join(str(value).strip().split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 1]}..."
