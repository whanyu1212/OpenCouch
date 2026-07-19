from __future__ import annotations

import asyncio
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
    VOICE_SAFETY_INTERRUPTION_DECIDED,
    VOICE_SAFETY_RESOURCES_RESOLVED,
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
    body = response.json()
    assert {key: body[key] for key in ("client_turn_id", "status", "reason")} == {
        "client_turn_id": "private-client-turn-id",
        "status": "completed",
        "reason": None,
    }
    assert body["action"] == "interrupt"
    assert body["risk_level"] == 2
    assert isinstance(body["interruption_token"], str)
    assert body["interruption_token"]
    assert set(body["support"]) == {"headline", "validation", "immediate_step"}
    assert all(body["support"].values())
    assert "private classifier reason" not in str(body)
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
    decision_event = next(
        event
        for event in recorder.events
        if event.name == VOICE_SAFETY_INTERRUPTION_DECIDED
    )
    assert decision_event.attributes["action"] == "interrupt"
    assert decision_event.attributes["risk_level"] == 2
    assert "private classifier reason" not in str(decision_event.attributes)


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
        "action": "continue",
        "risk_level": None,
        "support": None,
        "interruption_token": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lookup_status", "location", "resources"),
    [
        (
            "found",
            "Singapore",
            [
                {
                    "name": "Verified Line",
                    "phone": "123",
                    "url": "https://example.test",
                    "region": "Singapore",
                }
            ],
        ),
        ("no_location", "", []),
        ("location_refused", "", []),
        ("no_verified_results", "Singapore", []),
        ("lookup_error", "Singapore", []),
    ],
)
async def test_resource_endpoint_contract_uses_snapshot_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    lookup_status: str,
    location: str,
    resources: list[dict[str, str]],
) -> None:
    runtime = _runtime()
    observed: dict[str, Any] = {}

    async def fake_lookup(request, *, llm_client):
        observed["message"] = request.current_user_message
        observed["transcript"] = list(request.transcript)
        assert llm_client is not None
        return location, resources, lookup_status

    monkeypatch.setattr(
        "agent.voice.runtime_facade.find_crisis_resources_for_request",
        fake_lookup,
    )
    app = _app(monkeypatch, runtime, _ConcurrentSafetyLLM())
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-resources", config=TraceConfig(enabled=True)
    )

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="private-resource-thread",
            user_id=None,
            user_text="Earlier user text.",
            assistant_text="Earlier assistant text.",
            llm_client=None,
        )
        before = await runtime.get_state("private-resource-thread")
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with use_trace_context(trace_context, recorder):
                response = await client.post(
                    "/api/voice/realtime/safety/resources",
                    json={
                        "thread_id": "private-resource-thread",
                        "client_turn_id": "private-resource-turn",
                        "user_text": "Current private text.",
                        "prior_message_count": 1,
                        "pending_prior_transcript": [
                            {"role": "assistant", "content": "Pending prior text."}
                        ],
                    },
                )
        after = await runtime.get_state("private-resource-thread")

    assert response.status_code == 200
    body = response.json()
    assert body["client_turn_id"] == "private-resource-turn"
    assert body["status"] == lookup_status
    assert body["inferred_location"] == ""
    assert body["resources"] == (resources if lookup_status == "found" else [])
    assert body["message"]
    assert observed == {
        "message": "Current private text.",
        "transcript": [
            {"role": "user", "content": "Earlier user text."},
            {
                "role": "assistant",
                "content": "Earlier assistant text.",
                "response_style": "voice",
            },
            {"role": "assistant", "content": "Pending prior text."},
        ],
    }
    assert after == before
    event = next(
        event
        for event in recorder.events
        if event.name == VOICE_SAFETY_RESOURCES_RESOLVED
    )
    assert event.attributes["status"] == lookup_status
    assert event.attributes["resource_count"] == len(body["resources"])
    rendered = str(event.attributes)
    assert "private-resource-thread" not in rendered
    assert "private-resource-turn" not in rendered
    assert "Current private text" not in rendered
    assert location not in rendered or not location
    assert all(resource["name"] not in rendered for resource in resources)


@pytest.mark.asyncio
async def test_resource_endpoint_timeout_maps_to_lookup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    async def slow_lookup(request, *, llm_client):
        del request, llm_client
        await asyncio.sleep(0.05)
        return "Never returned", [], "no_verified_results"

    monkeypatch.setattr(
        "agent.voice.runtime_facade.find_crisis_resources_for_request",
        slow_lookup,
    )
    monkeypatch.setattr(
        "agent.voice.runtime_facade._SAFETY_RESOURCE_LOOKUP_TIMEOUT_SECONDS",
        0.005,
    )
    app = _app(monkeypatch, runtime, _ConcurrentSafetyLLM())

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/safety/resources",
                json={
                    "thread_id": "voice-timeout",
                    "client_turn_id": "turn-timeout",
                    "user_text": "Current text.",
                    "prior_message_count": 0,
                },
            )
        state = await runtime.get_state("voice-timeout")

    assert response.status_code == 200
    assert response.json()["status"] == "lookup_error"
    assert response.json()["resources"] == []
    assert state is None


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
