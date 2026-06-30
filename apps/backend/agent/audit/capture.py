"""Bounded runtime capture seam for safety audit events.

The conversation runtime should only capture minimal, structured safety facts. Any
operator review, daily summaries, exports, or retention purges should run later
through scripts/jobs over the persisted ledger.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from agent.audit.crisis_log import record_crisis_outcome, record_voice_missed_crisis
from agent.models import CrisisAssessment
from agent.observability.decorators import trace_event
from agent.observability.events import (
    AUDIT_SAFETY_EVENT_CAPTURE_COMPLETED,
    AUDIT_SAFETY_EVENT_CAPTURE_FAILED,
    AUDIT_SAFETY_EVENT_CAPTURE_SKIPPED,
    AUDIT_SAFETY_EVENT_CAPTURE_TIMEOUT,
)

logger = logging.getLogger(__name__)

DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS = 0.25
MIN_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS = 0.001

SafetyEventCaptureKind = Literal["crisis_response", "voice_missed_crisis"]
SafetyEventCaptureStatus = Literal["captured", "skipped", "timeout", "failed"]


@dataclass(frozen=True, slots=True)
class SafetyEventCaptureResult:
    """Observable result for one best-effort safety-event capture attempt."""

    kind: SafetyEventCaptureKind
    status: SafetyEventCaptureStatus
    reason: str | None = None
    timeout_seconds: float | None = None

    @property
    def captured(self) -> bool:
        """Return whether the capture operation completed within the bound."""

        return self.status == "captured"


async def capture_crisis_outcome(
    state: Mapping[str, Any],
    context: Any,
    *,
    timeout_seconds: float = DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS,
) -> SafetyEventCaptureResult:
    """Best-effort capture for a finalized crisis-response turn.

    This is the runtime-facing seam: it skips non-crisis turns, bounds latency for
    crisis turns, and delegates record construction/persistence to the existing
    crisis-log writer.
    """

    if not crisis_outcome_capture_required(state):
        return _skipped("crisis_response", reason="not_crisis_response")

    return await _capture_with_timeout(
        "crisis_response",
        lambda: record_crisis_outcome(state, context),
        timeout_seconds=timeout_seconds,
    )


async def capture_voice_missed_crisis(
    state: Mapping[str, Any],
    context: Any,
    *,
    assessment: CrisisAssessment,
    timeout_seconds: float = DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS,
) -> SafetyEventCaptureResult:
    """Best-effort capture for a post-turn voice missed-crisis event."""

    if not assessment.needs_crisis_response:
        return _skipped("voice_missed_crisis", reason="not_crisis_response")

    return await _capture_with_timeout(
        "voice_missed_crisis",
        lambda: record_voice_missed_crisis(
            state,
            context,
            assessment=assessment,
        ),
        timeout_seconds=timeout_seconds,
    )


def crisis_outcome_capture_required(state: Mapping[str, Any]) -> bool:
    """Return whether a finalized state should emit a crisis safety event."""

    crisis = state.get("crisis")
    if crisis is None:
        return False
    if isinstance(crisis, Mapping):
        return bool(crisis.get("needs_crisis_response"))
    return bool(getattr(crisis, "needs_crisis_response", False))


async def _capture_with_timeout(
    kind: SafetyEventCaptureKind,
    operation: Callable[[], Awaitable[dict[str, Any]]],
    *,
    timeout_seconds: float,
) -> SafetyEventCaptureResult:
    bounded_timeout = max(
        MIN_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS,
        float(timeout_seconds),
    )
    try:
        await asyncio.wait_for(operation(), timeout=bounded_timeout)
    except TimeoutError:
        logger.warning(
            "safety event capture timed out",
            extra={"event_kind": kind, "timeout_seconds": bounded_timeout},
        )
        trace_event(
            AUDIT_SAFETY_EVENT_CAPTURE_TIMEOUT,
            {
                "event_kind": kind,
                "timeout_seconds": bounded_timeout,
            },
        )
        return SafetyEventCaptureResult(
            kind=kind,
            status="timeout",
            reason="timeout",
            timeout_seconds=bounded_timeout,
        )
    except Exception:
        logger.warning(
            "safety event capture failed",
            extra={"event_kind": kind},
            exc_info=True,
        )
        trace_event(
            AUDIT_SAFETY_EVENT_CAPTURE_FAILED,
            {"event_kind": kind},
        )
        return SafetyEventCaptureResult(
            kind=kind,
            status="failed",
            reason="exception",
            timeout_seconds=bounded_timeout,
        )

    trace_event(
        AUDIT_SAFETY_EVENT_CAPTURE_COMPLETED,
        {
            "event_kind": kind,
            "timeout_seconds": bounded_timeout,
        },
    )
    return SafetyEventCaptureResult(
        kind=kind,
        status="captured",
        timeout_seconds=bounded_timeout,
    )


def _skipped(
    kind: SafetyEventCaptureKind,
    *,
    reason: str,
) -> SafetyEventCaptureResult:
    trace_event(
        AUDIT_SAFETY_EVENT_CAPTURE_SKIPPED,
        {"event_kind": kind, "reason": reason},
    )
    return SafetyEventCaptureResult(kind=kind, status="skipped", reason=reason)


__all__ = [
    "DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS",
    "SafetyEventCaptureKind",
    "SafetyEventCaptureResult",
    "SafetyEventCaptureStatus",
    "capture_crisis_outcome",
    "capture_voice_missed_crisis",
    "crisis_outcome_capture_required",
]
