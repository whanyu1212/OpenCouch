from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.modes import MemoryMode
from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import (
    VOICE_CONCURRENT_SAFETY_ASSESSED,
    VOICE_CONCURRENT_SAFETY_TURN_OBSERVED,
)
from agent.observability.recorder import InMemoryTraceRecorder
from agent.runtime import PersistentAgentRuntime
from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection
from tests.support.persistence import (
    FakeCrossRestartLLM,
    in_memory_audit_feedback_dependencies,
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)


class _ConcurrentSafetyLLM(FakeCrossRestartLLM):
    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        if response_schema.__name__ == "CrisisAssessmentSchema":
            return response_schema(
                level=2,
                confidence="high",
                reason="private classifier reason about current transcript",
                needs_crisis_response=True,
                needs_clarification=False,
            )
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


def _runtime() -> PersistentAgentRuntime:
    return PersistentAgentRuntime(
        dependencies=in_memory_audit_feedback_dependencies(),
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )


def _app(
    monkeypatch: pytest.MonkeyPatch,
    runtime: PersistentAgentRuntime,
    llm_client: FakeCrossRestartLLM | None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: llm_client
    return app


@pytest.mark.asyncio
async def test_safety_endpoint_returns_status_and_privacy_safe_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    app = _app(monkeypatch, runtime, _ConcurrentSafetyLLM())
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-safety", config=TraceConfig(enabled=True)
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with use_trace_context(trace_context, recorder):
                response = await client.post(
                    "/api/voice/realtime/safety/check",
                    json={
                        "thread_id": "private-thread-id",
                        "user_id": "private-user-id",
                        "memory_mode": "incognito",
                        "client_turn_id": "private-client-turn-id",
                        "user_text": "I might hurt myself.",
                        "prior_message_count": 0,
                    },
                )

        state = await runtime.get_state("private-thread-id")

    assert response.status_code == 200
    assert response.json() == {
        "client_turn_id": "private-client-turn-id",
        "status": "completed",
        "reason": None,
    }
    assert state is None
    event = next(
        event
        for event in recorder.events
        if event.name == VOICE_CONCURRENT_SAFETY_ASSESSED
    )
    assert event.attributes["mode"] == "observe"
    assert event.attributes["status"] == "completed"
    assert event.attributes["memory_mode"] == "incognito"
    assert event.attributes["level"] == 2
    assert event.attributes["confidence"] == "high"
    assert event.attributes["needs_crisis_response"] is True
    assert event.attributes["needs_clarification"] is False
    assert len(event.attributes["correlation_hash"]) == 64
    rendered = str(event.attributes)
    assert "private-thread-id" not in rendered
    assert "private-user-id" not in rendered
    assert "private-client-turn-id" not in rendered
    assert "I might hurt myself" not in rendered
    assert "private classifier reason" not in rendered


@pytest.mark.asyncio
async def test_safety_endpoint_reports_missing_llm_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    app = _app(monkeypatch, runtime, None)

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/safety/check",
                json={
                    "thread_id": "voice-thread",
                    "client_turn_id": "turn-1",
                    "user_text": "current text",
                    "prior_message_count": 0,
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "client_turn_id": "turn-1",
        "status": "skipped",
        "reason": "no_llm_client",
    }


@pytest.mark.asyncio
async def test_turn_event_correlates_and_observes_crisis_tools_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    app = _app(monkeypatch, runtime, _ConcurrentSafetyLLM())
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-correlation",
        config=TraceConfig(enabled=True),
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with use_trace_context(trace_context, recorder):
                safety_response = await client.post(
                    "/api/voice/realtime/safety/check",
                    json={
                        "thread_id": "private-thread-id",
                        "client_turn_id": "private-client-turn-id",
                        "user_text": "I might hurt myself.",
                        "prior_message_count": 0,
                    },
                )
                turn_response = await client.post(
                    "/api/voice/realtime/turn",
                    json={
                        "thread_id": "private-thread-id",
                        "client_turn_id": "private-client-turn-id",
                        "user_text": "I might hurt myself.",
                        "assistant_text": "Please contact someone nearby now.",
                        "tool_calls": [
                            {
                                "tool_name": "get_crisis_support_template",
                                "status": "completed",
                            },
                            {
                                "tool_name": "lookup_crisis_resources",
                                "status": "failed",
                                "error": "not available",
                            },
                        ],
                    },
                )

        state = await runtime.get_state("private-thread-id")

    assert safety_response.status_code == 200
    assert turn_response.status_code == 200
    assert state is not None
    assert "client_turn_id" not in state
    safety_event = next(
        event
        for event in recorder.events
        if event.name == VOICE_CONCURRENT_SAFETY_ASSESSED
    )
    turn_event = next(
        event
        for event in recorder.events
        if event.name == VOICE_CONCURRENT_SAFETY_TURN_OBSERVED
    )
    assert turn_event.attributes == {
        "voice_runtime": "openai_realtime",
        "correlation_hash": safety_event.attributes["correlation_hash"],
        "route": "crisis",
        "completed_tool_names": ["get_crisis_support_template"],
        "completed_tool_count": 1,
        "failed_tool_names": ["lookup_crisis_resources"],
        "failed_tool_count": 1,
        "crisis_tool_observed": True,
    }
    rendered = str(turn_event.attributes)
    assert "private-thread-id" not in rendered
    assert "private-client-turn-id" not in rendered


@pytest.mark.asyncio
async def test_turn_endpoint_omits_observation_event_without_client_turn_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    app = _app(monkeypatch, runtime, None)
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-turn", config=TraceConfig(enabled=True)
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with use_trace_context(trace_context, recorder):
                response = await client.post(
                    "/api/voice/realtime/turn",
                    json={
                        "thread_id": "voice-thread",
                        "user_text": "I had a hard day.",
                        "assistant_text": "I am listening.",
                    },
                )

    assert response.status_code == 200
    assert not any(
        event.name == VOICE_CONCURRENT_SAFETY_TURN_OBSERVED for event in recorder.events
    )
