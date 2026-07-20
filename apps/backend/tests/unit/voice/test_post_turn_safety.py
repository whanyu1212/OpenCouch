"""Unit tests for the non-blocking voice post-turn safety auditor."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.guardrails.service import CrisisRiskResult, CrisisRiskService
from agent.memory.modes import MemoryMode
from agent.models import CrisisAssessment
from agent.state import AgentState
from agent.voice.post_turn_safety import (
    VoicePostTurnSafetyAuditor,
    VoicePostTurnSafetyCheck,
)
from llm.base import BaseLLMClient
from tests.support.persistence import FakeCrossRestartLLM


class _SlowCrisisRiskService(CrisisRiskService):
    def __init__(self, *, delay_seconds: float = 0.05, fail: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.fail = fail
        self.calls = 0

    async def assess_turn(
        self,
        state: object,
        *,
        llm_client: BaseLLMClient | None,
    ) -> CrisisRiskResult:
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("classifier exploded")
        return CrisisRiskResult(
            assessment=CrisisAssessment(
                level=0,
                confidence="high",
                reason="safe test turn",
                needs_crisis_response=False,
                needs_clarification=False,
            ),
            classifier_path="llm_primary",
        )


def _check(
    *,
    thread_id: str = "voice-thread",
    llm_client: BaseLLMClient | None = None,
    turn_instance_id: str | None = None,
) -> VoicePostTurnSafetyCheck:
    return VoicePostTurnSafetyCheck(
        thread_id=thread_id,
        user_id="user-1",
        user_text="I had a rough day.",
        realtime_route="therapeutic",
        response_style="supportive",
        state=cast(
            AgentState,
            {
                "session_id": thread_id,
                "user_id": "user-1",
                "response_style": "supportive",
                "diagnostics": {"voice_runtime": "openai_realtime"},
            },
        ),
        prior_state=None,
        context=SimpleNamespace(
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        ),
        llm_client=llm_client,
        turn_instance_id=turn_instance_id,
    )


@pytest.mark.asyncio
async def test_schedule_check_reports_no_llm_skip() -> None:
    auditor = VoicePostTurnSafetyAuditor()

    result = auditor.schedule_check(_check(llm_client=None))

    assert result.as_dict() == {
        "scheduled": False,
        "status": "skipped",
        "reason": "no_llm_client",
        "pending_count": 0,
    }


@pytest.mark.asyncio
async def test_schedule_check_reports_queue_limit_skip() -> None:
    auditor = VoicePostTurnSafetyAuditor(
        service=_SlowCrisisRiskService(delay_seconds=0.05),
        max_pending_tasks=1,
    )
    llm = FakeCrossRestartLLM()

    first = auditor.schedule_check(_check(thread_id="voice-one", llm_client=llm))
    second = auditor.schedule_check(_check(thread_id="voice-two", llm_client=llm))
    pending = await auditor.drain(timeout_seconds=1.0)

    assert first.scheduled is True
    assert first.pending_count == 1
    assert second.as_dict() == {
        "scheduled": False,
        "status": "skipped",
        "reason": "task_limit_reached",
        "pending_count": 1,
    }
    assert pending == 0


@pytest.mark.asyncio
async def test_schedule_check_reuses_result_for_same_turn_instance() -> None:
    service = _SlowCrisisRiskService(delay_seconds=0.0)
    auditor = VoicePostTurnSafetyAuditor(service=service)
    check = _check(
        llm_client=FakeCrossRestartLLM(),
        turn_instance_id="same-voice-turn",
    )

    first = auditor.schedule_check(check)
    second = auditor.schedule_check(check)
    pending = await auditor.drain(timeout_seconds=1.0)

    assert first.scheduled is True
    assert second == first
    assert pending == 0
    assert service.calls == 1


@pytest.mark.asyncio
async def test_schedule_cache_scopes_results_to_turn_instance() -> None:
    service = _SlowCrisisRiskService(delay_seconds=0.0)
    auditor = VoicePostTurnSafetyAuditor(service=service)
    llm = FakeCrossRestartLLM()

    first = auditor.schedule_check(
        _check(
            thread_id="voice-thread",
            llm_client=llm,
            turn_instance_id="turn-one",
        )
    )
    second = auditor.schedule_check(
        _check(
            thread_id="voice-thread",
            llm_client=llm,
            turn_instance_id="turn-two",
        )
    )
    pending = await auditor.drain(timeout_seconds=1.0)

    assert first.scheduled is True
    assert second.scheduled is True
    assert pending == 0
    assert service.calls == 2


@pytest.mark.asyncio
async def test_timeout_failure_is_logged_and_drained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    auditor = VoicePostTurnSafetyAuditor(
        service=_SlowCrisisRiskService(delay_seconds=0.2),
        timeout_seconds=0.01,
    )

    with caplog.at_level(logging.WARNING):
        result = auditor.schedule_check(_check(llm_client=FakeCrossRestartLLM()))
        pending = await auditor.drain(timeout_seconds=1.0)

    assert result.scheduled is True
    assert pending == 0
    assert "voice post-turn safety check timed out" in caplog.text


@pytest.mark.asyncio
async def test_exception_failure_is_logged_and_drained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    auditor = VoicePostTurnSafetyAuditor(
        service=_SlowCrisisRiskService(delay_seconds=0.0, fail=True),
    )

    with caplog.at_level(logging.WARNING):
        result = auditor.schedule_check(_check(llm_client=FakeCrossRestartLLM()))
        pending = await auditor.drain(timeout_seconds=1.0)

    assert result.scheduled is True
    assert pending == 0
    assert "voice post-turn safety check failed" in caplog.text
