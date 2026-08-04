"""Unit tests for observation-only concurrent voice safety assessment."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from agent.guardrails.service import CrisisRiskResult, CrisisRiskService
from agent.models import Channel, CrisisAssessment
from agent.voice.concurrent_safety import VoiceConcurrentSafetyService
from llm.base import BaseLLMClient
from tests.support.persistence import FakeCrossRestartLLM


class _RecordingRiskService(CrisisRiskService):
    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.fail = fail
        self.states: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def assess_turn(
        self,
        state: Any,
        *,
        llm_client: BaseLLMClient | None,
    ) -> CrisisRiskResult:
        assert llm_client is not None
        self.states.append(state)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_seconds)
            if self.fail:
                raise RuntimeError("provider failed with sensitive details")
            return CrisisRiskResult(
                assessment=CrisisAssessment(
                    level=2,
                    confidence="high",
                    reason="classifier-only reason",
                    needs_crisis_response=True,
                    needs_clarification=False,
                ),
                classifier_path="llm_primary",
            )
        finally:
            self.active -= 1


def _assess(
    service: VoiceConcurrentSafetyService,
    *,
    user_text: str = "I might hurt myself.",
    llm_client: BaseLLMClient | None = None,
    prior_transcript: list[dict[str, Any]] | None = None,
):
    return service.assess_turn(
        thread_id="voice-thread",
        user_id="user-1",
        user_text=user_text,
        prior_transcript=prior_transcript or [],
        llm_client=llm_client,
    )


@pytest.mark.asyncio
async def test_completed_result_uses_isolated_voice_classifier_state() -> None:
    risk_service = _RecordingRiskService()
    service = VoiceConcurrentSafetyService(service=risk_service)
    transcript = [{"role": "user", "content": "earlier", "metadata": {"nested": True}}]

    result = await _assess(
        service,
        user_text="  I might hurt myself.  ",
        llm_client=FakeCrossRestartLLM(),
        prior_transcript=transcript,
    )

    assert result.status == "completed"
    assert result.reason is None
    assert result.assessment is not None
    assert result.assessment.level == 2
    assert result.duration_ms >= 0
    assert risk_service.states == [
        {
            "message": "I might hurt myself.",
            "channel": Channel.VOICE,
            "user_id": "user-1",
            "session_id": "voice-thread",
            "transcript": transcript,
        }
    ]
    assert risk_service.states[0]["transcript"] is not transcript
    assert risk_service.states[0]["transcript"][0] is not transcript[0]
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_text", "llm_client", "reason"),
    [
        ("current text", None, "no_llm_client"),
        ("   ", FakeCrossRestartLLM(), "empty_user_text"),
    ],
)
async def test_missing_inputs_skip_without_calling_classifier(
    user_text: str,
    llm_client: BaseLLMClient | None,
    reason: str,
) -> None:
    risk_service = _RecordingRiskService()
    service = VoiceConcurrentSafetyService(service=risk_service)

    result = await _assess(
        service,
        user_text=user_text,
        llm_client=llm_client,
    )

    assert result.status == "skipped"
    assert result.reason == reason
    assert result.assessment is None
    assert risk_service.states == []


@pytest.mark.asyncio
async def test_timeout_returns_fail_open_result() -> None:
    service = VoiceConcurrentSafetyService(
        service=_RecordingRiskService(delay_seconds=0.05),
        timeout_seconds=0.005,
    )

    result = await _assess(service, llm_client=FakeCrossRestartLLM())

    assert result.status == "timeout"
    assert result.reason == "timeout"
    assert result.assessment is None


@pytest.mark.asyncio
async def test_provider_exception_returns_fail_open_result() -> None:
    service = VoiceConcurrentSafetyService(
        service=_RecordingRiskService(fail=True),
    )

    result = await _assess(service, llm_client=FakeCrossRestartLLM())

    assert result.status == "failed"
    assert result.reason == "exception"
    assert result.assessment is None


@pytest.mark.asyncio
async def test_assessments_respect_instance_concurrency_limit() -> None:
    risk_service = _RecordingRiskService(delay_seconds=0.01)
    service = VoiceConcurrentSafetyService(
        service=risk_service,
        max_concurrency=2,
    )
    llm = cast(BaseLLMClient, FakeCrossRestartLLM())

    results = await asyncio.gather(
        *(_assess(service, llm_client=llm) for _ in range(6))
    )

    assert all(result.status == "completed" for result in results)
    assert risk_service.max_active == 2


@pytest.mark.asyncio
async def test_timeout_includes_waiting_for_concurrency_slot() -> None:
    service = VoiceConcurrentSafetyService(
        service=_RecordingRiskService(delay_seconds=0.05),
        timeout_seconds=0.08,
        max_concurrency=1,
    )
    llm = cast(BaseLLMClient, FakeCrossRestartLLM())

    first, second = await asyncio.gather(
        _assess(service, llm_client=llm),
        _assess(service, llm_client=llm),
    )

    assert first.status == "completed"
    assert second.status == "timeout"
