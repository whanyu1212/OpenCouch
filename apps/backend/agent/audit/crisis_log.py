"""Crisis safety ledger record writers and backends.

The crisis ledger is separate from prompt memory and writes regardless of
memory mode. Runtime code captures only minimal structured safety events through
``agent.audit.capture`` after response/state finalization; retention purges,
exports, summaries, and review are operator- or maintenance-driven and never
happen during normal turn processing.

Concrete backends share the same async protocol:

- ``InMemoryCrisisLogBackend`` is ephemeral and used by tests and
  incognito runtimes.
- ``PostgresCrisisLogBackend`` is the supported durable implementation.
- ``NullCrisisLogBackend`` is reserved for explicit test fixtures.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any, Mapping, Protocol, cast, get_args
from uuid import uuid4

from agent.audit.models import (
    CrisisLogRecord,
    CrisisResourceLookupStatus,
    CrisisResponsePath,
)
from agent.memory.hashing import hash_session_id
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.models import CrisisAssessment
from agent.observability.context import get_current_trace_context
from agent.observability.decorators import trace_event
from agent.observability.events import AUDIT_CRISIS_LOG_APPEND

logger = logging.getLogger(__name__)

# Derived from the canonical Literal so the runtime guard can never drift from
# the type. Adding a status to CrisisResourceLookupStatus extends this set too.
_VALID_RESOURCE_LOOKUP_STATUSES: frozenset[str] = frozenset(
    get_args(CrisisResourceLookupStatus)
)


class CrisisLogBackend(Protocol):
    """Protocol that any crisis-log backend must implement.

    Crisis-response side effects write records, debugging and audit code
    reads them back by date, CLI status reports the total count, retention
    purging deletes expired records, and the runtime lifecycle owns
    closing.
    """

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one crisis event record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Persists the record in the backend.
        """
        ...

    async def aappend_once(self, record: CrisisLogRecord) -> bool:
        """Append a record unless its deterministic ID already exists."""

        ...

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """
        ...

    async def arecord_count(self) -> int:
        """Count crisis records across the backend.

        Returns:
            int: Total crisis-log record count.
        """
        ...

    async def apurge_before(self, cutoff: date) -> int:
        """Purge crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Returns:
            None: Closes the backend.
        """
        ...


async def write_crisis_log(
    state: Mapping[str, Any],
    context: Any,
    *,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Build and append one minimal crisis safety event record.

    Args:
        state: Finalized or in-progress turn state.
        context: Runtime context exposing the crisis ledger backend.
        raise_on_failure: When ``True``, re-raise write failures after logging so
            bounded capture callers can report an accurate failed status. The
            default preserves the legacy best-effort/no-crash writer contract.
    """

    crisis = state.get("crisis")
    needs_crisis_response = (
        bool(crisis.get("needs_crisis_response"))
        if isinstance(crisis, Mapping)
        else bool(getattr(crisis, "needs_crisis_response", False))
    )
    if crisis is None or not needs_crisis_response:
        logger.debug("crisis log called on non-crisis turn; skipping write")
        return {}

    # Guard the whole write, not just the append: record construction reads
    # several state fields and validates a Pydantic model, and the backend
    # handle is resolved here too. A crisis audit write is a safety side-channel
    # -- if any of it raises, it must degrade to a logged error and never
    # propagate out to break the crisis response or the turn lifecycle.
    try:
        backend = context.crisis_log_backend
        crisis_audit = state.get("crisis_audit", {})
        override_kind = crisis_audit.get("crisis_override_kind", "none")
        classifier_path = crisis_audit.get("crisis_classifier_path", "llm_primary")
        llm_failure_occurred = crisis_audit.get("crisis_llm_failure_occurred", False)

        if "crisis_classifier_path" not in crisis_audit:
            logger.debug(
                "crisis log: no classifier_path in crisis_audit state; "
                "using default 'llm_primary'"
            )

        user_id = (
            None
            if context.memory_mode == MemoryMode.INCOGNITO
            else state.get("user_id")
        )
        level = crisis.get("level", 0) if isinstance(crisis, Mapping) else crisis.level
        reason = (
            crisis.get("reason", "") if isinstance(crisis, Mapping) else crisis.reason
        )
        diagnostics = state.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        response_path = _response_path_from_diagnostics(diagnostics)
        trace_context = get_current_trace_context()
        enabled_trace_context = (
            trace_context
            if trace_context is not None and trace_context.enabled
            else None
        )
        deterministic_audit_id = diagnostics.get("voice_crisis_audit_id")
        record = CrisisLogRecord(
            id=(
                deterministic_audit_id
                if isinstance(deterministic_audit_id, str) and deterministic_audit_id
                else str(uuid4())
            ),
            session_id_opaque=hash_session_id(state.get("session_id")),
            user_id_or_null=user_id,
            detected_at=iso_now(),
            level=level,
            override_kind=override_kind,
            classifier_path=classifier_path,
            # CrisisAssessment.reason is uncapped LLM output; CrisisLogRecord.reason
            # enforces max_length=500. Truncate here so an over-long reason can never
            # raise ValidationError and silently drop the whole crisis audit record.
            reason=(reason or "")[:500],
            response_node_completed=True,
            llm_failure_occurred=llm_failure_occurred,
            response_style=str(state.get("response_style") or "crisis_response"),
            resource_lookup_status=_resource_lookup_status_from_state(state),
            resource_count=_resource_count_from_state(state),
            tool_calls=_tool_calls_from_diagnostics(diagnostics),
            response_path=response_path,
            fallback_reason=_fallback_reason_from_diagnostics(
                diagnostics,
                response_path=response_path,
            ),
            trace_id=enabled_trace_context.trace_id if enabled_trace_context else None,
            trace_session_id=(
                enabled_trace_context.session_id if enabled_trace_context else None
            ),
            trace_turn_id=enabled_trace_context.turn_id
            if enabled_trace_context
            else None,
            trace_runtime_mode=(
                enabled_trace_context.runtime_mode if enabled_trace_context else None
            ),
        )
        if isinstance(deterministic_audit_id, str) and deterministic_audit_id:
            audit_recorded = await backend.aappend_once(record)
        else:
            await backend.aappend(record)
            audit_recorded = True
        trace_event(
            AUDIT_CRISIS_LOG_APPEND,
            {
                "audit_recorded": audit_recorded,
                "event_type": record.event_type,
                "level": record.level,
                "classifier_path": record.classifier_path,
                "resource_lookup_status": record.resource_lookup_status,
                "resource_count": record.resource_count,
                "response_path": record.response_path,
                "runtime_mode": (
                    enabled_trace_context.runtime_mode
                    if enabled_trace_context
                    else None
                ),
                "trace_correlated": enabled_trace_context is not None,
            },
        )
    except Exception:
        logger.error(
            "crisis log failed to write record; audit trail lost for this event",
            exc_info=True,
        )
        if raise_on_failure:
            raise

    return {}


async def record_crisis_outcome(
    state: Mapping[str, Any],
    context: Any,
    *,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Write the minimal crisis event record for a finalized turn.

    This lower-level writer is channel-neutral but not timeout-bounded. Runtime
    callers should prefer ``agent.audit.capture.capture_crisis_outcome`` so the
    conversation path only waits for the configured best-effort capture window.
    """

    return await write_crisis_log(
        state,
        context,
        raise_on_failure=raise_on_failure,
    )


async def record_voice_missed_crisis(
    state: Mapping[str, Any],
    context: Any,
    *,
    assessment: CrisisAssessment,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Write a post-turn audit record for a voice crisis miss.

    The voice post-turn classifier runs after the Realtime response has already
    been spoken and persisted. When it detects level-2/3 risk but the Realtime
    path did not route crisis, record a distinct audit event instead of
    pretending a crisis response completed.
    """

    if not assessment.needs_crisis_response:
        logger.debug("voice missed-crisis audit called for non-crisis assessment")
        return {}

    try:
        backend = context.crisis_log_backend
        user_id = (
            None
            if context.memory_mode == MemoryMode.INCOGNITO
            else state.get("user_id")
        )
        diagnostics = state.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        deterministic_audit_id = diagnostics.get("voice_missed_crisis_audit_id")
        trace_context = get_current_trace_context()
        enabled_trace_context = (
            trace_context
            if trace_context is not None and trace_context.enabled
            else None
        )
        record = CrisisLogRecord(
            id=(
                deterministic_audit_id
                if isinstance(deterministic_audit_id, str) and deterministic_audit_id
                else str(uuid4())
            ),
            event_type="voice_missed_crisis",
            session_id_opaque=hash_session_id(state.get("session_id")),
            user_id_or_null=user_id,
            detected_at=iso_now(),
            level=assessment.level,
            override_kind="none",
            classifier_path="voice_post_turn",
            reason=(assessment.reason or "voice_post_turn_classifier")[:500],
            response_node_completed=False,
            llm_failure_occurred=False,
            response_style=str(state.get("response_style") or "voice"),
            resource_lookup_status="not_attempted",
            resource_count=0,
            tool_calls=_voice_tool_calls_from_diagnostics(diagnostics),
            response_path="not_routed",
            fallback_reason="voice_realtime_crisis_tool_not_called",
            trace_id=enabled_trace_context.trace_id if enabled_trace_context else None,
            trace_session_id=(
                enabled_trace_context.session_id if enabled_trace_context else None
            ),
            trace_turn_id=enabled_trace_context.turn_id
            if enabled_trace_context
            else None,
            trace_runtime_mode=(
                enabled_trace_context.runtime_mode if enabled_trace_context else "voice"
            ),
        )
        if isinstance(deterministic_audit_id, str) and deterministic_audit_id:
            audit_recorded = await backend.aappend_once(record)
        else:
            await backend.aappend(record)
            audit_recorded = True
        trace_event(
            AUDIT_CRISIS_LOG_APPEND,
            {
                "audit_recorded": audit_recorded,
                "event_type": record.event_type,
                "level": record.level,
                "classifier_path": record.classifier_path,
                "resource_lookup_status": record.resource_lookup_status,
                "resource_count": record.resource_count,
                "response_path": record.response_path,
                "runtime_mode": record.trace_runtime_mode,
                "trace_correlated": enabled_trace_context is not None,
            },
        )
    except Exception:
        logger.error(
            "voice missed-crisis audit failed to write record; audit trail lost for this event",
            exc_info=True,
        )
        if raise_on_failure:
            raise

    return {}


def _resource_lookup_status_from_state(
    state: Mapping[str, Any],
) -> CrisisResourceLookupStatus:
    status = str(state.get("resource_lookup_status") or "not_attempted")
    if status in _VALID_RESOURCE_LOOKUP_STATUSES:
        return cast(CrisisResourceLookupStatus, status)
    return "not_attempted"


def _resource_count_from_state(state: Mapping[str, Any]) -> int:
    resources = state.get("found_resources")
    if not isinstance(resources, list):
        return 0
    return len(resources)


def _tool_calls_from_diagnostics(diagnostics: Mapping[str, Any]) -> list[str]:
    tool_calls = diagnostics.get("openai_crisis_tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [str(tool_name) for tool_name in tool_calls]


def _voice_tool_calls_from_diagnostics(diagnostics: Mapping[str, Any]) -> list[str]:
    tool_calls = diagnostics.get("voice_tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [str(tool_name) for tool_name in tool_calls]


def _response_path_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> CrisisResponsePath:
    # Voice crisis turns answer in the Realtime model's live reply, not via the
    # text SDK tool loop, so none of the text fallback keys below are set. Map
    # them to "sdk" up front; otherwise every voice record would read "unknown".
    if diagnostics.get("voice_turn_outcome") == "safety_interrupted":
        return "safety_overlay"
    if diagnostics.get("voice_runtime") == "openai_realtime":
        return "sdk"
    if diagnostics.get("openai_response_llm_override") is True:
        return "response_llm_override"
    if diagnostics.get("openai_crisis_tool_fallback") is True:
        return "sdk_tool_fallback"
    if "openai_crisis_tool_fallback" in diagnostics:
        return "sdk"
    return "unknown"


def _fallback_reason_from_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    response_path: CrisisResponsePath,
) -> str | None:
    explicit_reason = diagnostics.get("openai_sdk_fallback_reason")
    if explicit_reason:
        return str(explicit_reason)[:200]
    if response_path == "response_llm_override":
        return "response_llm_override"
    if response_path == "sdk_tool_fallback":
        return "crisis_resource_tool_not_called"
    return None


class InMemoryCrisisLogBackend:
    """In-memory crisis log backend for tests and incognito runtimes.

    Stores records in a per-instance dict keyed by ``datetime.date``.
    Nothing persists across runtime restarts.

    NOT thread-safe. Each runtime instance should own its own backend.
    """

    def __init__(self) -> None:
        """Initialize the in-memory crisis backend.

        Returns:
            None: Creates an empty in-memory record store.
        """

        self._records_by_date: dict[date, list[CrisisLogRecord]] = defaultdict(list)
        self._closed = False

    def _ensure_open(self) -> None:
        """Raise when the backend has already been closed.

        Raises:
            RuntimeError: If the backend is closed.

        Returns:
            None: The backend is open.
        """

        if self._closed:
            raise RuntimeError("InMemoryCrisisLogBackend is closed.")

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one in-memory crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Stores the record in memory.
        """

        self._ensure_open()
        day = date.fromisoformat(record.detected_at.split("T", 1)[0])
        self._records_by_date[day].append(record)

    async def aappend_once(self, record: CrisisLogRecord) -> bool:
        """Append a record unless its deterministic ID already exists."""

        self._ensure_open()
        if any(
            existing.id == record.id
            for records in self._records_by_date.values()
            for existing in records
        ):
            return False
        await self.aappend(record)
        return True

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List in-memory crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        self._ensure_open()
        return list(self._records_by_date.get(day, []))

    async def aclose(self) -> None:
        """Close the in-memory crisis backend.

        Returns:
            None: Marks the backend closed and clears in-memory data.
        """

        if self._closed:
            return
        self._closed = True
        self._records_by_date.clear()

    async def arecord_count(self) -> int:
        """Count in-memory crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        if self._closed:
            return 0
        return sum(len(records) for records in self._records_by_date.values())

    async def apurge_before(self, cutoff: date) -> int:
        """Purge in-memory crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        if self._closed:
            return 0

        stale_dates = [day for day in self._records_by_date.keys() if day < cutoff]
        deleted = 0
        for day in stale_dates:
            deleted += len(self._records_by_date[day])
            del self._records_by_date[day]
        return deleted


class NullCrisisLogBackend:
    """No-op crisis log backend.

    Reserved for test fixtures that want to assert "no crisis events
    were logged" or want to explicitly disable logging. NOT a valid
    production backend — the always-on crisis log promise requires
    a real backend in all three memory modes.
    """

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Discard a crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to ignore.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def aappend_once(self, record: CrisisLogRecord) -> bool:
        """Discard an idempotent crisis record."""

        return False

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Always an empty list.
        """

        return []

    async def aclose(self) -> None:
        """Close the null crisis backend.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def arecord_count(self) -> int:
        """Count crisis records in the null backend.

        Returns:
            int: Always ``0``.
        """

        return 0

    async def apurge_before(self, cutoff: date) -> int:  # noqa: ARG002
        """Purge crisis records from the null backend.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Always ``0``.
        """

        return 0


if TYPE_CHECKING:
    _in_memory_backend: CrisisLogBackend = InMemoryCrisisLogBackend()
    _null_backend: CrisisLogBackend = NullCrisisLogBackend()
