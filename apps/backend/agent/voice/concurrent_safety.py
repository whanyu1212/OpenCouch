"""Observation-only safety assessment for in-progress Realtime voice turns."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

from agent.guardrails.service import CrisisRiskResult, CrisisRiskService
from agent.models import Channel, CrisisAssessment
from agent.observability.timing import elapsed_ms
from agent.state import AgentState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

VoiceConcurrentSafetyStatus = Literal["completed", "skipped", "timeout", "failed"]

_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_MAX_CONCURRENCY = 2


@dataclass(frozen=True, slots=True)
class VoiceConcurrentSafetyResult:
    """Immutable outcome of one fail-open voice safety observation."""

    status: VoiceConcurrentSafetyStatus
    reason: str | None
    assessment: CrisisAssessment | None
    duration_ms: float


class VoiceConcurrentSafetyService:
    """Run bounded crisis assessments without changing voice runtime state."""

    def __init__(
        self,
        *,
        service: CrisisRiskService | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self._service = service or CrisisRiskService()
        self._timeout_seconds = max(0.001, float(timeout_seconds))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def assess_turn(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        prior_transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
    ) -> VoiceConcurrentSafetyResult:
        """Classify one voice turn and fail open on timeout or provider errors."""

        start = time.monotonic()
        text = user_text.strip()
        if llm_client is None:
            return VoiceConcurrentSafetyResult(
                status="skipped",
                reason="no_llm_client",
                assessment=None,
                duration_ms=round(elapsed_ms(start), 2),
            )
        if not text:
            return VoiceConcurrentSafetyResult(
                status="skipped",
                reason="empty_user_text",
                assessment=None,
                duration_ms=round(elapsed_ms(start), 2),
            )

        classifier_state = cast(
            AgentState,
            {
                "message": text,
                "channel": Channel.VOICE,
                "user_id": user_id,
                "session_id": thread_id,
                "transcript": copy.deepcopy(prior_transcript),
            },
        )

        async def run_assessment() -> CrisisRiskResult:
            async with self._semaphore:
                return await self._service.assess_turn(
                    classifier_state,
                    llm_client=llm_client,
                )

        try:
            result = await asyncio.wait_for(
                run_assessment(),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return VoiceConcurrentSafetyResult(
                status="timeout",
                reason="timeout",
                assessment=None,
                duration_ms=round(elapsed_ms(start), 2),
            )
        except Exception as exc:
            logger.warning(
                "voice concurrent safety assessment failed: %s",
                type(exc).__name__,
            )
            return VoiceConcurrentSafetyResult(
                status="failed",
                reason="exception",
                assessment=None,
                duration_ms=round(elapsed_ms(start), 2),
            )

        return VoiceConcurrentSafetyResult(
            status="completed",
            reason=None,
            assessment=result.assessment,
            duration_ms=round(elapsed_ms(start), 2),
        )


__all__ = [
    "VoiceConcurrentSafetyResult",
    "VoiceConcurrentSafetyService",
    "VoiceConcurrentSafetyStatus",
]
