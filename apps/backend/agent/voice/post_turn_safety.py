"""Post-turn safety auditing for OpenAI Realtime voice turns.

The voice runtime is speech-to-speech, so this module deliberately runs after a
voice response has already been spoken and persisted. It is an audit/safety-net
path, not a pre-response gate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, cast

from agent.audit.capture import capture_voice_missed_crisis
from agent.guardrails.service import CrisisRiskService
from agent.models import Channel
from agent.observability.decorators import trace_event
from agent.observability.events import (
    VOICE_POST_TURN_SAFETY_COMPLETED,
    VOICE_POST_TURN_SAFETY_FAILED,
    VOICE_POST_TURN_SAFETY_MISSED_CRISIS,
    VOICE_POST_TURN_SAFETY_SCHEDULED,
    VOICE_POST_TURN_SAFETY_SKIPPED,
)
from agent.observability.timing import elapsed_ms
from agent.state import AgentState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_CLOSE_DRAIN_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_CONCURRENCY = 2
_DEFAULT_MAX_PENDING_TASKS = 100
_TRANSIENT_SKIP_REASONS = {"no_llm_client", "task_limit_reached"}


@dataclass(frozen=True, slots=True)
class VoicePostTurnSafetyCheck:
    """Immutable inputs for one post-turn voice safety audit."""

    thread_id: str
    user_id: str | None
    user_text: str
    realtime_route: str
    response_style: str
    state: AgentState
    prior_state: AgentState | None
    context: Any
    llm_client: BaseLLMClient | None
    turn_instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class VoicePostTurnSafetyScheduleResult:
    """Observable outcome of attempting to enqueue a post-turn safety check."""

    scheduled: bool
    reason: str | None = None
    pending_count: int = 0

    @property
    def status(self) -> str:
        """Return a stable API-facing status string."""

        return "scheduled" if self.scheduled else "skipped"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable diagnostic payload."""

        return {
            "scheduled": self.scheduled,
            "status": self.status,
            "reason": self.reason,
            "pending_count": self.pending_count,
        }


class VoicePostTurnSafetyAuditor:
    """Bounded in-process task runner for post-turn voice safety checks."""

    def __init__(
        self,
        *,
        service: CrisisRiskService | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        close_drain_timeout_seconds: float = _DEFAULT_CLOSE_DRAIN_TIMEOUT_SECONDS,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_pending_tasks: int = _DEFAULT_MAX_PENDING_TASKS,
    ) -> None:
        self._service = service or CrisisRiskService()
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._close_drain_timeout_seconds = max(
            0.0,
            float(close_drain_timeout_seconds),
        )
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._max_pending_tasks = max(1, int(max_pending_tasks))
        self._tasks: set[asyncio.Task[None]] = set()
        self._schedule_results: dict[
            tuple[str, str], VoicePostTurnSafetyScheduleResult
        ] = {}
        self._closed = False

    @property
    def pending_count(self) -> int:
        """Return the number of not-yet-finished audit tasks."""

        self._discard_finished_tasks()
        return len(self._tasks)

    def forget_schedule_result(
        self,
        *,
        thread_id: str,
        turn_instance_id: str | None,
    ) -> None:
        """Release retry state after the turn receipt is durable."""

        if turn_instance_id is None:
            return
        self._schedule_results.pop((thread_id, turn_instance_id), None)

    def schedule_check(
        self,
        check: VoicePostTurnSafetyCheck,
    ) -> VoicePostTurnSafetyScheduleResult:
        """Schedule one background audit check when it is useful and possible.

        Returns:
            VoicePostTurnSafetyScheduleResult: Whether work was scheduled and,
            when skipped, the reason. The caller can surface this immediately
            because later classifier completion/failure happens asynchronously.
        """

        if check.turn_instance_id is not None:
            previous = self._schedule_results.get(
                (check.thread_id, check.turn_instance_id)
            )
            if previous is not None:
                return previous

        if self._closed:
            return self._remember_schedule_result(
                check,
                self._skip_check(check, reason="closed"),
            )
        if check.llm_client is None:
            return self._remember_schedule_result(
                check,
                self._skip_check(check, reason="no_llm_client"),
            )
        if not check.user_text.strip():
            return self._remember_schedule_result(
                check,
                self._skip_check(check, reason="empty_user_text"),
            )
        if check.realtime_route == "crisis":
            return self._remember_schedule_result(
                check,
                self._skip_check(check, reason="already_crisis_routed"),
            )

        self._discard_finished_tasks()
        if len(self._tasks) >= self._max_pending_tasks:
            return self._remember_schedule_result(
                check,
                self._skip_check(check, reason="task_limit_reached"),
            )

        task = asyncio.create_task(
            self._run_check(check),
            name=f"voice-post-turn-safety:{check.thread_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(lambda done: self._tasks.discard(done))
        pending_count = len(self._tasks)
        trace_event(
            VOICE_POST_TURN_SAFETY_SCHEDULED,
            {
                "voice_runtime": "openai_realtime",
                "route": check.realtime_route,
                "response_style": check.response_style,
                "pending_count": pending_count,
            },
        )
        return self._remember_schedule_result(
            check,
            VoicePostTurnSafetyScheduleResult(
                scheduled=True,
                pending_count=pending_count,
            ),
        )

    async def drain(self, timeout_seconds: float | None = None) -> int:
        """Wait for scheduled checks to finish.

        Args:
            timeout_seconds: Optional maximum wait. ``None`` waits until all
                current tasks finish.

        Returns:
            int: Number of tasks still pending after the drain attempt.
        """

        self._discard_finished_tasks()
        pending = tuple(self._tasks)
        if not pending:
            return 0

        if timeout_seconds is None:
            await asyncio.gather(*pending, return_exceptions=True)
            self._discard_finished_tasks()
            return 0

        done, still_pending = await asyncio.wait(
            pending,
            timeout=max(0.0, float(timeout_seconds)),
        )
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        self._discard_finished_tasks()
        return len(still_pending)

    async def aclose(self) -> None:
        """Drain then cancel remaining checks during runtime shutdown."""

        self._closed = True
        remaining = await self.drain(self._close_drain_timeout_seconds)
        if remaining == 0:
            return

        pending = tuple(task for task in self._tasks if not task.done())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._discard_finished_tasks()

    async def _run_check(self, check: VoicePostTurnSafetyCheck) -> None:
        try:
            async with self._semaphore:
                await asyncio.wait_for(
                    self._classify_and_audit(check),
                    timeout=self._timeout_seconds,
                )
        except TimeoutError:
            logger.warning("voice post-turn safety check timed out")
            trace_event(
                VOICE_POST_TURN_SAFETY_FAILED,
                {
                    "voice_runtime": "openai_realtime",
                    "reason": "timeout",
                    "route": check.realtime_route,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("voice post-turn safety check failed", exc_info=True)
            trace_event(
                VOICE_POST_TURN_SAFETY_FAILED,
                {
                    "voice_runtime": "openai_realtime",
                    "reason": "exception",
                    "route": check.realtime_route,
                },
            )

    async def _classify_and_audit(self, check: VoicePostTurnSafetyCheck) -> None:
        start = time.monotonic()
        classifier_state = _classifier_state_for_check(check)
        result = await self._service.assess_turn(
            classifier_state,
            llm_client=check.llm_client,
        )
        assessment = result.assessment
        missed_crisis = assessment.level >= 2 and assessment.needs_crisis_response

        if missed_crisis:
            await capture_voice_missed_crisis(
                check.state,
                check.context,
                assessment=assessment,
            )
            trace_event(
                VOICE_POST_TURN_SAFETY_MISSED_CRISIS,
                {
                    "voice_runtime": "openai_realtime",
                    "level": assessment.level,
                    "classifier_path": "voice_post_turn",
                    "source_classifier_path": result.classifier_path,
                    "route": check.realtime_route,
                    "response_style": check.response_style,
                },
            )

        trace_event(
            VOICE_POST_TURN_SAFETY_COMPLETED,
            {
                "voice_runtime": "openai_realtime",
                "level": assessment.level,
                "needs_crisis_response": assessment.needs_crisis_response,
                "needs_clarification": assessment.needs_clarification,
                "missed_crisis": missed_crisis,
                "classifier_path": "voice_post_turn",
                "source_classifier_path": result.classifier_path,
                "route": check.realtime_route,
                "duration_ms": round(elapsed_ms(start), 2),
            },
        )

    def _skip_check(
        self,
        check: VoicePostTurnSafetyCheck,
        *,
        reason: str,
    ) -> VoicePostTurnSafetyScheduleResult:
        self._discard_finished_tasks()
        pending_count = len(self._tasks)
        trace_event(
            VOICE_POST_TURN_SAFETY_SKIPPED,
            {
                "voice_runtime": "openai_realtime",
                "reason": reason,
                "route": check.realtime_route,
                "response_style": check.response_style,
                "pending_count": pending_count,
            },
        )
        return VoicePostTurnSafetyScheduleResult(
            scheduled=False,
            reason=reason,
            pending_count=pending_count,
        )

    def _discard_finished_tasks(self) -> None:
        self._tasks = {task for task in self._tasks if not task.done()}

    def _remember_schedule_result(
        self,
        check: VoicePostTurnSafetyCheck,
        result: VoicePostTurnSafetyScheduleResult,
    ) -> VoicePostTurnSafetyScheduleResult:
        turn_instance_id = check.turn_instance_id
        if (
            turn_instance_id is None
            or not result.scheduled
            and result.reason in _TRANSIENT_SKIP_REASONS
        ):
            return result
        self._schedule_results[(check.thread_id, turn_instance_id)] = result
        return result


def _classifier_state_for_check(check: VoicePostTurnSafetyCheck) -> AgentState:
    """Build a text-gate-compatible state for the voice user's latest turn."""

    classifier_state = cast(AgentState, dict(check.state))
    classifier_state["message"] = check.user_text.strip()
    classifier_state["channel"] = Channel.VOICE
    classifier_state["user_id"] = check.user_id
    classifier_state["session_id"] = check.thread_id
    classifier_state["transcript"] = (
        list(check.prior_state.get("transcript", []) or [])
        if check.prior_state is not None
        else []
    )
    return classifier_state


__all__ = [
    "VoicePostTurnSafetyAuditor",
    "VoicePostTurnSafetyCheck",
    "VoicePostTurnSafetyScheduleResult",
]
