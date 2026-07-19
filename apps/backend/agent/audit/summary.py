"""Aggregate helpers for safety audit records."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from agent.audit.models import (
    CrisisLogAggregate,
    CrisisLogLevelCounts,
    CrisisLogPathCounts,
    CrisisLogRecord,
)


def summarize_crisis_log_records(
    day: date,
    records: Iterable[CrisisLogRecord],
) -> CrisisLogAggregate:
    """Summarize one day's crisis log records for operator review."""

    level_counts = CrisisLogLevelCounts()
    path_counts = CrisisLogPathCounts()
    events_total = 0
    completed_total = 0
    llm_failures_total = 0
    tool_fallbacks_total = 0
    response_llm_overrides_total = 0
    voice_missed_crises_total = 0

    for record in records:
        events_total += 1
        if record.response_node_completed:
            completed_total += 1
        if record.llm_failure_occurred:
            llm_failures_total += 1
        if record.response_path == "sdk_tool_fallback":
            tool_fallbacks_total += 1
        if record.response_path == "response_llm_override":
            response_llm_overrides_total += 1
        if record.event_type == "voice_missed_crisis":
            voice_missed_crises_total += 1

        level_field = f"level_{record.level}"
        setattr(level_counts, level_field, getattr(level_counts, level_field) + 1)

        classifier_path = record.classifier_path
        setattr(path_counts, classifier_path, getattr(path_counts, classifier_path) + 1)

    completion_rate = 1.0 if events_total == 0 else completed_total / events_total
    return CrisisLogAggregate(
        date=day.isoformat(),
        events_total=events_total,
        events_by_level=level_counts,
        events_by_classifier_path=path_counts,
        llm_failures_total=llm_failures_total,
        tool_fallbacks_total=tool_fallbacks_total,
        response_llm_overrides_total=response_llm_overrides_total,
        voice_missed_crises_total=voice_missed_crises_total,
        response_node_completion_rate=completion_rate,
    )


__all__ = ["summarize_crisis_log_records"]
