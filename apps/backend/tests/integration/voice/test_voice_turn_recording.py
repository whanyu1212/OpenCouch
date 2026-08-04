from __future__ import annotations

from dataclasses import replace

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.hashing import hash_session_id
from agent.memory.modes import MemoryMode
from agent.models import CrisisAssessment, MessageRole
from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import (
    VOICE_PENDING_TURNS_RETIRED,
    VOICE_RESPONSE_FINALIZED,
    VOICE_SAFETY_INTERRUPTED_TURN_RECORDED,
    VOICE_TURN_COMPLETION_METADATA_PERSIST_FAILED,
)
from agent.observability.recorder import InMemoryTraceRecorder
from agent.runtime import PersistentAgentRuntime, RuntimeBehaviorConfig
from agent.voice.runtime_facade import (
    VoicePendingTurnCapacityError,
    VoicePendingTurnHandleBusyError,
    VoicePendingTurnRetiredError,
)
from agent.voice.safety_proof import VoiceSafetyInterruptionProofService
from api.dependencies import get_llm_client
from api.models import VoiceTurnRecordRequest
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection
from tests.support.persistence import (
    FakeCrossRestartLLM,
    in_memory_audit_feedback_dependencies,
    in_memory_runtime_storage_paths,
    postgres_thread_persistence_config,
    runtime_persistence_config,
    runtime_storage_paths,
)
from tests.support.safety_capture import (
    CRISIS_VOICE_RESPONSE_TEXT,
    CRISIS_VOICE_USER_TEXT,
    utc_crisis_records,
    voice_crisis_lookup_tool_call,
)


def _runtime(**kwargs) -> PersistentAgentRuntime:
    return PersistentAgentRuntime(
        dependencies=in_memory_audit_feedback_dependencies(),
        **kwargs,
    )


class _VoiceCrisisAuditLLM(FakeCrossRestartLLM):
    def __init__(
        self,
        *,
        level: int,
        reason: str = "voice post-turn classifier detected crisis risk",
        delay_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self.level = level
        self.reason = reason
        self.delay_seconds = delay_seconds

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        if response_schema.__name__ == "CrisisAssessmentSchema":
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            self.crisis_calls += 1
            return response_schema(
                level=self.level,
                confidence="high",
                reason=self.reason,
                needs_crisis_response=self.level >= 2,
                needs_clarification=self.level == 1,
            )
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


@pytest.mark.asyncio
async def test_record_voice_turn_persists_thread_history() -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-thread",
            user_id=None,
            user_text="I feel overwhelmed.",
            assistant_text="That sounds like a lot to carry.",
            response_style="supportive",
            llm_client=None,
        )
        history = await runtime.get_history("voice-thread")

    assert [message.role for message in history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert history[0].content == "I feel overwhelmed."
    assert history[1].content == "That sounds like a lot to carry."
    assert history[1].response_style == "supportive"


@pytest.mark.asyncio
async def test_voice_turn_endpoint_records_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "voice-thread",
                    "memory_mode": "persistent",
                    "user_text": "I feel overwhelmed.",
                    "assistant_text": "That sounds like a lot to carry.",
                    "response_style": "supportive",
                },
            )

        history = await runtime.get_history("voice-thread")
        persisted_state = await runtime.get_state("voice-thread")

    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is True
    assert body["message_count"] == 2
    expected_safety = {
        "scheduled": False,
        "status": "skipped",
        "reason": "no_llm_client",
        "pending_count": 0,
    }
    assert body["post_turn_safety"] == expected_safety
    assert persisted_state is not None
    assert persisted_state["diagnostics"]["voice_post_turn_safety"] == expected_safety
    assert [message.content for message in history] == [
        "I feel overwhelmed.",
        "That sounds like a lot to carry.",
    ]


@pytest.mark.asyncio
async def test_voice_turn_endpoint_infers_route_and_tool_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "voice-thread",
                    "user_id": "user-1",
                    "memory_mode": "persistent",
                    "user_text": "What is the latest guidance?",
                    "assistant_text": "I found a verified answer.",
                    "tool_calls": [
                        {
                            "tool_name": "answer_grounded_lookup",
                            "status": "completed",
                            "output": {
                                "grounded_lookup": {
                                    "query": "latest guidance",
                                    "status": "answered",
                                }
                            },
                        }
                    ],
                },
            )

        state = await runtime.get_state("voice-thread")

    assert response.status_code == 200
    assert state is not None
    assert state["route"] == "grounded_lookup"
    assert state["response_style"] == "grounded_lookup"
    assert state["grounded_lookup"] == {
        "query": "latest guidance",
        "status": "answered",
    }
    assert state["transcript"][-1]["response_style"] == "grounded_lookup"
    assert state["diagnostics"]["voice_tool_calls"] == ["answer_grounded_lookup"]


@pytest.mark.asyncio
async def test_safety_interrupted_turn_persists_user_and_settled_tools_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=3)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: llm
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-interrupted", config=TraceConfig(enabled=True)
    )

    async with runtime:
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id="private-interrupted-thread",
            user_id="user-1",
            client_turn_id="private-interrupted-turn",
            inferred_location="",
            found_resources=[],
            resource_lookup_status="lookup_error",
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with use_trace_context(trace_context, recorder):
                response = await client.post(
                    "/api/voice/realtime/turn",
                    json={
                        "thread_id": "private-interrupted-thread",
                        "client_turn_id": "private-interrupted-turn",
                        "interruption_token": voice_routes._VOICE_SAFETY_PROOFS.issue(
                            thread_id="private-interrupted-thread",
                            client_turn_id="private-interrupted-turn",
                            user_text="I might hurt myself.",
                            user_id="user-1",
                            memory_mode="persistent",
                            risk_level=3,
                        ),
                        "user_id": "user-1",
                        "outcome": "safety_interrupted",
                        "user_text": "I might hurt myself.",
                        "assistant_text": "Partial assistant audio must disappear.",
                        "route": "crisis",
                        "response_style": "crisis_response",
                        "tool_calls": [
                            {
                                "tool_name": "lookup_crisis_resources",
                                "status": "completed",
                                "output": {"resource_lookup_status": "lookup_error"},
                            },
                            {
                                "tool_name": "settled_failed_tool",
                                "status": "failed",
                                "error": "settled diagnostic",
                            },
                            {
                                "tool_name": "unsettled_tool",
                                "status": "started",
                                "output": {"partial": True},
                            },
                        ],
                    },
                )
        state = await runtime.get_state("private-interrupted-thread")
        history = await runtime.get_history("private-interrupted-thread")
        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    assert response.status_code == 200
    assert response.json()["message_count"] == 1
    assert response.json()["post_turn_safety"] == {
        "scheduled": False,
        "status": "skipped",
        "reason": "safety_interruption_verified",
        "pending_count": 0,
    }
    assert state is not None
    assert state["route"] == "voice_safety_interrupted"
    assert state["response_style"] == "voice_safety_interrupted"
    assert state["response_text"] == ""
    assert state["transcript"] == [{"role": "user", "content": "I might hurt myself."}]
    outcomes = state["diagnostics"]["voice_tool_call_outcomes"]
    assert [outcome["tool_name"] for outcome in outcomes] == [
        "lookup_crisis_resources",
        "settled_failed_tool",
    ]
    assert [outcome["status"] for outcome in outcomes] == ["completed", "failed"]
    assert state["diagnostics"]["voice_tool_calls"] == [
        "lookup_crisis_resources",
        "settled_failed_tool",
    ]
    assert state["resource_lookup_status"] == "lookup_error"
    assert "Partial assistant audio" not in str(state)
    assert [message.role for message in history] == [MessageRole.USER]
    assert llm.crisis_calls == 0
    assert runtime.voice.post_turn_safety_pending_count == 0
    assert len(audit_records) == 1
    audit_record = audit_records[0]
    assert audit_record.event_type == "crisis_response"
    assert audit_record.classifier_path == "voice_concurrent"
    assert audit_record.response_node_completed is True
    assert audit_record.response_path == "safety_overlay"
    assert audit_record.fallback_reason is None
    assert audit_record.resource_lookup_status == "lookup_error"
    assert audit_record.tool_calls == [
        "lookup_crisis_resources",
        "settled_failed_tool",
    ]
    assert state["crisis"].level == 3
    event = next(
        event
        for event in recorder.events
        if event.name == VOICE_SAFETY_INTERRUPTED_TURN_RECORDED
    )
    assert event.attributes["route"] == "voice_safety_interrupted"
    assert len(event.attributes["correlation_hash"]) == 64
    assert "private-interrupted" not in str(event.attributes)
    assert not any(event.name == VOICE_RESPONSE_FINALIZED for event in recorder.events)


@pytest.mark.asyncio
async def test_overlapping_voice_turns_keep_crisis_resources_correlated() -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    thread_id = "overlapping-crisis-resource-thread"
    turn_a = "overlapping-turn-a"
    turn_b = "overlapping-turn-b"
    hash_a = hashlib.sha256(f"{thread_id}\0{turn_a}".encode()).hexdigest()
    hash_b = hashlib.sha256(f"{thread_id}\0{turn_b}".encode()).hexdigest()
    resources_a = [
        {"name": "Turn A support", "phone": "111", "url": "https://a.example"}
    ]
    resources_b = [
        {"name": "Turn B support", "phone": "222", "url": "https://b.example"}
    ]
    tool_calls = [
        {
            "tool_name": "lookup_crisis_resources",
            "status": "completed",
            "output": {"resource_lookup_status": "found"},
        }
    ]

    async with runtime:
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id=thread_id,
            user_id="user-1",
            client_turn_id=turn_a,
            inferred_location="Location A",
            found_resources=resources_a,
            resource_lookup_status="found",
        )
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id=thread_id,
            user_id="user-1",
            client_turn_id=turn_b,
            inferred_location="Location B",
            found_resources=resources_b,
            resource_lookup_status="found",
        )

        context_a = await runtime.voice.build_voice_tool_context(
            thread_id=thread_id,
            user_id="user-1",
            current_user_message="Turn A",
            transcript=[],
            client_turn_id=turn_a,
        )
        lookup_a = context_a.latest_crisis_resource_tool_result()
        assert lookup_a is not None
        assert lookup_a.inferred_location == "Location A"
        assert lookup_a.found_resources == resources_a

        state_a = await runtime.voice.record_voice_turn(
            thread_id=thread_id,
            user_id="user-1",
            user_text="Turn A crisis request",
            assistant_text="Turn A response",
            tool_calls=tool_calls,
            correlation_hash=hash_a,
            request_hash="request-a",
            llm_client=None,
        )
        assert state_a["found_resources"] == resources_a
        assert set(state_a["diagnostics"]["voice_crisis_resource_lookups"]) == {hash_b}

        context_b = await runtime.voice.build_voice_tool_context(
            thread_id=thread_id,
            user_id="user-1",
            current_user_message="Turn B",
            transcript=[],
            client_turn_id=turn_b,
        )
        lookup_b = context_b.latest_crisis_resource_tool_result()
        assert lookup_b is not None
        assert lookup_b.inferred_location == "Location B"
        assert lookup_b.found_resources == resources_b

        state_b = await runtime.voice.record_voice_turn(
            thread_id=thread_id,
            user_id="user-1",
            user_text="Turn B crisis request",
            assistant_text="Turn B response",
            tool_calls=tool_calls,
            correlation_hash=hash_b,
            request_hash="request-b",
            llm_client=None,
        )

    assert state_b["found_resources"] == resources_b
    assert "voice_crisis_resource_lookups" not in state_b["diagnostics"]


@pytest.mark.asyncio
async def test_safety_interrupted_turn_requires_server_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=0)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: llm

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "unverified-interrupted-thread",
                    "client_turn_id": "unverified-interrupted-turn",
                    "interruption_token": "invalid-proof",
                    "outcome": "safety_interrupted",
                    "user_text": "This is not a crisis.",
                },
            )
        state = await runtime.get_state("unverified-interrupted-thread")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == (
        "voice_safety_interruption_proof_invalid"
    )
    assert state is None


@pytest.mark.asyncio
async def test_safety_interrupted_turn_retry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=3)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: llm
    payload = {
        "thread_id": "idempotent-interrupted-thread",
        "client_turn_id": "idempotent-interrupted-turn",
        "outcome": "safety_interrupted",
        "user_text": "I might hurt myself.",
    }
    payload["interruption_token"] = voice_routes._VOICE_SAFETY_PROOFS.issue(
        thread_id=payload["thread_id"],
        client_turn_id=payload["client_turn_id"],
        user_text=payload["user_text"],
        user_id=None,
        memory_mode="persistent",
        risk_level=3,
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first, second = await asyncio.gather(
                client.post("/api/voice/realtime/turn", json=payload),
                client.post("/api/voice/realtime/turn", json=payload),
            )
            await runtime.voice.record_voice_turn(
                thread_id="idempotent-interrupted-thread",
                user_id=None,
                user_text="Later user turn.",
                assistant_text="Later assistant turn.",
                llm_client=None,
            )
            delayed_retry = await client.post("/api/voice/realtime/turn", json=payload)
            conflict = await client.post(
                "/api/voice/realtime/turn",
                json={**payload, "user_text": "Different text."},
            )
        history = await runtime.get_history("idempotent-interrupted-thread")
        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert delayed_retry.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == (
        "voice_realtime_turn_idempotency_conflict"
    )
    assert [message.content for message in history] == [
        "I might hurt myself.",
        "Later user turn.",
        "Later assistant turn.",
    ]
    assert len(audit_records) == 1
    assert llm.crisis_calls == 0


@pytest.mark.asyncio
async def test_safety_interrupted_pending_retry_accepts_expired_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    sdk_attempts = 0

    async def fail_sdk_history_once(*args: Any, **kwargs: Any) -> None:
        nonlocal sdk_attempts
        sdk_attempts += 1
        if sdk_attempts == 1:
            raise RuntimeError("simulated retryable SDK history failure")

    monkeypatch.setattr(
        runtime.voice,
        "_collaboration",
        replace(
            runtime.voice._collaboration,
            ensure_sdk_turn_recorded=fail_sdk_history_once,
        ),
    )
    now = 100.0
    proofs = VoiceSafetyInterruptionProofService(
        secret=b"expired-retry-test",
        ttl_seconds=1,
        clock=lambda: now,
    )
    monkeypatch.setattr(voice_routes, "_VOICE_SAFETY_PROOFS", proofs)
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    app.dependency_overrides[get_llm_client] = lambda: None
    payload = {
        "thread_id": "expired-proof-retry-thread",
        "client_turn_id": "expired-proof-retry-turn",
        "outcome": "safety_interrupted",
        "user_text": "I might hurt myself.",
    }
    payload["interruption_token"] = proofs.issue(
        thread_id=payload["thread_id"],
        client_turn_id=payload["client_turn_id"],
        user_text=payload["user_text"],
        user_id=None,
        memory_mode="persistent",
        risk_level=3,
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first = await client.post("/api/voice/realtime/turn", json=payload)
            now = 102.0
            retry = await client.post("/api/voice/realtime/turn", json=payload)
            conflict = await client.post(
                "/api/voice/realtime/turn",
                json={**payload, "user_text": "Different text."},
            )
        history = await runtime.get_history(payload["thread_id"])

    assert first.status_code == 500
    assert retry.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == (
        "voice_realtime_turn_idempotency_conflict"
    )
    assert sdk_attempts == 2
    assert [message.content for message in history] == [payload["user_text"]]


@pytest.mark.asyncio
async def test_safety_interrupted_turn_requires_user_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "voice-invalid-interrupted",
                    "outcome": "safety_interrupted",
                    "user_text": "   ",
                    "assistant_text": "Must not make this valid.",
                },
            )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_first_voice_crisis_lookup_does_not_advance_turn_count() -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id="voice-first-crisis",
            user_id="user-1",
            inferred_location="Singapore",
            found_resources=[],
            resource_lookup_status="lookup_error",
        )
        placeholder = await runtime.get_state("voice-first-crisis")

        await runtime.voice.record_voice_turn(
            thread_id="voice-first-crisis",
            user_id="user-1",
            user_text=CRISIS_VOICE_USER_TEXT,
            assistant_text=CRISIS_VOICE_RESPONSE_TEXT,
            tool_calls=[
                voice_crisis_lookup_tool_call(resource_lookup_status="lookup_error")
            ],
            llm_client=None,
        )
        recorded = await runtime.get_state("voice-first-crisis")

    assert placeholder is not None
    assert placeholder["session_progress"]["turn_count"] == 0
    assert recorded is not None
    assert recorded["session_progress"]["turn_count"] == 1


@pytest.mark.asyncio
async def test_voice_crisis_turn_writes_one_audit_record() -> None:
    """A voice turn that called lookup_crisis_resources is audited like text.

    The live Realtime crisis route is prompt/tool driven, so this in-turn
    audit record is based on the model's crisis tool call. Missed non-crisis
    routes are checked separately by the post-turn safety auditor. The runtime
    must still write exactly one ``CrisisLogRecord`` here so a crisis over voice
    is as auditable as one over text, and the verified resource status must
    thread into the record.
    """

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id="voice-crisis-thread",
            user_id="user-1",
            inferred_location="Singapore",
            found_resources=[
                {
                    "name": "Samaritans of Singapore",
                    "phone": "1767",
                    "url": "https://www.sos.org.sg",
                    "region": "Singapore",
                }
            ],
            resource_lookup_status="found",
        )
        await runtime.voice.record_voice_turn(
            thread_id="voice-crisis-thread",
            user_id="user-1",
            user_text=CRISIS_VOICE_USER_TEXT,
            assistant_text=CRISIS_VOICE_RESPONSE_TEXT,
            tool_calls=[
                voice_crisis_lookup_tool_call(
                    found_resources=[
                        {
                            "name": "Samaritans of Singapore",
                            "phone": "1767",
                            "url": "https://www.sos.org.sg",
                            "region": "Singapore",
                        }
                    ],
                )
            ],
            llm_client=None,
        )

        records = await utc_crisis_records(runtime.crisis_log_backend)

    assert len(records) == 1
    record = records[0]
    assert record.level == 2
    assert record.reason == "voice_crisis_tool_call"
    assert record.classifier_path == "llm_primary"
    assert record.override_kind == "none"
    assert record.resource_lookup_status == "found"
    assert record.resource_count == 1
    assert record.tool_calls == ["lookup_crisis_resources"]


@pytest.mark.asyncio
async def test_voice_non_crisis_followup_does_not_reaudit_prior_crisis() -> None:
    """A later ordinary voice turn must not re-audit stale prior crisis state."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-crisis-followup",
            user_id="user-1",
            user_text=CRISIS_VOICE_USER_TEXT,
            assistant_text="Your safety matters. Let's get immediate support.",
            tool_calls=[
                voice_crisis_lookup_tool_call(resource_lookup_status="lookup_error")
            ],
            llm_client=None,
        )
        await runtime.voice.record_voice_turn(
            thread_id="voice-crisis-followup",
            user_id="user-1",
            user_text="I made tea and feel steadier now.",
            assistant_text="I'm glad you're feeling a little steadier.",
            response_style="supportive",
            llm_client=None,
        )
        records = await utc_crisis_records(runtime.crisis_log_backend)

    assert len(records) == 1
    assert records[0].event_type == "crisis_response"


@pytest.mark.asyncio
async def test_voice_crisis_capture_runs_before_sdk_history_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved voice crisis turn should still audit if SDK bookkeeping fails."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )

    async with runtime:

        async def fail_sdk_history(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated voice SDK history failure")

        monkeypatch.setattr(
            runtime.voice,
            "_collaboration",
            replace(
                runtime.voice._collaboration,
                ensure_sdk_turn_recorded=fail_sdk_history,
            ),
        )

        with pytest.raises(RuntimeError, match="simulated voice SDK history failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-crisis-sdk-failure",
                user_id="user-1",
                user_text=CRISIS_VOICE_USER_TEXT,
                assistant_text=CRISIS_VOICE_RESPONSE_TEXT,
                route="crisis",
                response_style="crisis_response",
                tool_calls=[voice_crisis_lookup_tool_call()],
                llm_client=None,
            )

        assert await runtime.crisis_log_backend.arecord_count() == 1
        state = await runtime.get_state("voice-crisis-sdk-failure")
        assert state is not None
        assert state["route"] == "crisis"


@pytest.mark.asyncio
async def test_voice_turn_retry_resumes_failed_lifecycle_without_duplicate_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    sdk_attempts = 0

    async def fail_sdk_history_once(*args: Any, **kwargs: Any) -> None:
        nonlocal sdk_attempts
        sdk_attempts += 1
        if sdk_attempts == 1:
            raise RuntimeError("simulated retryable SDK history failure")

    monkeypatch.setattr(
        runtime.voice,
        "_collaboration",
        replace(
            runtime.voice._collaboration,
            ensure_sdk_turn_recorded=fail_sdk_history_once,
        ),
    )
    correlation_hash = "retryable-voice-turn"
    request_hash = "stable-request"

    async with runtime:
        with pytest.raises(RuntimeError, match="retryable SDK history failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-lifecycle-retry",
                user_id="user-1",
                user_text="I had a difficult day.",
                assistant_text="I'm here with you.",
                correlation_hash=correlation_hash,
                request_hash=request_hash,
                llm_client=None,
            )

        pending_state = await runtime.get_state("voice-lifecycle-retry")
        assert pending_state is not None
        pending_diagnostics = pending_state.get("diagnostics", {})
        assert correlation_hash not in pending_diagnostics.get(
            "voice_recorded_turn_hashes", []
        )
        assert correlation_hash in pending_diagnostics["voice_pending_turns"]
        assert (
            await runtime.voice.recorded_voice_turn_receipt(
                thread_id="voice-lifecycle-retry",
                correlation_hash=correlation_hash,
                request_hash=request_hash,
            )
            is None
        )

        state = await runtime.voice.record_voice_turn(
            thread_id="voice-lifecycle-retry",
            user_id="user-1",
            user_text="I had a difficult day.",
            assistant_text="I'm here with you.",
            correlation_hash=correlation_hash,
            request_hash=request_hash,
            llm_client=None,
        )
        history = await runtime.get_history("voice-lifecycle-retry")
        receipt = await runtime.voice.recorded_voice_turn_receipt(
            thread_id="voice-lifecycle-retry",
            correlation_hash=correlation_hash,
            request_hash=request_hash,
        )

    assert sdk_attempts == 2
    assert [message.content for message in history] == [
        "I had a difficult day.",
        "I'm here with you.",
    ]
    assert receipt is not None
    assert receipt.message_count == 2
    assert correlation_hash in state["diagnostics"]["voice_recorded_turn_hashes"]
    assert "voice_pending_turns" not in state["diagnostics"]


@pytest.mark.asyncio
async def test_pending_voice_turn_capacity_preserves_existing_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state

    async def fail_completion_receipt_save(thread_id: str, state: Any) -> None:
        diagnostics = state.get("diagnostics", {})
        if diagnostics.get("voice_recorded_turn_receipts"):
            raise RuntimeError("simulated pending voice turn failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_save,
    )

    async with runtime:
        for index in range(32):
            with pytest.raises(RuntimeError, match="pending voice turn failure"):
                await runtime.voice.record_voice_turn(
                    thread_id="voice-pending-capacity",
                    user_id="user-1",
                    user_text=f"user turn {index}",
                    assistant_text=f"assistant turn {index}",
                    correlation_hash=f"pending-correlation-{index}",
                    request_hash=f"pending-request-{index}",
                    retry_handle_id=f"retry-handle-{index}",
                    llm_client=None,
                )

        with pytest.raises(VoicePendingTurnCapacityError):
            await runtime.voice.record_voice_turn(
                thread_id="voice-pending-capacity",
                user_id="user-1",
                user_text="overflow user turn",
                assistant_text="overflow assistant turn",
                correlation_hash="pending-correlation-overflow",
                request_hash="pending-request-overflow",
                retry_handle_id="retry-handle-overflow",
                llm_client=None,
            )

        pending_state = await runtime.get_state("voice-pending-capacity")
        assert pending_state is not None
        assert len(pending_state["diagnostics"]["voice_pending_turns"]) == 32

        monkeypatch.setattr(runtime._state_store, "save_state", original_save_state)
        await runtime.voice.record_voice_turn(
            thread_id="voice-pending-capacity",
            user_id="user-1",
            user_text="user turn 0",
            assistant_text="assistant turn 0",
            correlation_hash="pending-correlation-0",
            request_hash="pending-request-0",
            retry_handle_id="retry-handle-0",
            llm_client=None,
        )
        state = await runtime.get_state("voice-pending-capacity")

    assert state is not None
    assert len(state["diagnostics"]["voice_pending_turns"]) == 31
    assert "pending-correlation-0" not in state["diagnostics"]["voice_pending_turns"]


@pytest.mark.asyncio
async def test_pending_voice_turn_retry_cannot_claim_another_handles_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state

    async def fail_completion_receipt_save(thread_id: str, state: Any) -> None:
        diagnostics = state.get("diagnostics", {})
        if diagnostics.get("voice_recorded_turn_receipts"):
            raise RuntimeError("simulated pending voice turn failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_save,
    )

    async with runtime:
        for correlation_hash, request_hash, retry_handle_id in (
            ("first-pending-correlation", "first-pending-request", "first-handle"),
            ("second-pending-correlation", "second-pending-request", "second-handle"),
        ):
            with pytest.raises(RuntimeError, match="pending voice turn failure"):
                await runtime.voice.record_voice_turn(
                    thread_id="voice-pending-handle-ownership",
                    user_id="user-1",
                    user_text=correlation_hash,
                    assistant_text="assistant turn",
                    correlation_hash=correlation_hash,
                    request_hash=request_hash,
                    retry_handle_id=retry_handle_id,
                    llm_client=None,
                )

        with pytest.raises(VoicePendingTurnHandleBusyError):
            await runtime.voice.record_voice_turn(
                thread_id="voice-pending-handle-ownership",
                user_id="user-1",
                user_text="first-pending-correlation",
                assistant_text="assistant turn",
                correlation_hash="first-pending-correlation",
                request_hash="first-pending-request",
                retry_handle_id="second-handle",
                llm_client=None,
            )

        state = await runtime.get_state("voice-pending-handle-ownership")

    assert state is not None
    pending_turns = state["diagnostics"]["voice_pending_turns"]
    assert (
        pending_turns["first-pending-correlation"]["retry_handle_id"] == "first-handle"
    )
    assert (
        pending_turns["second-pending-correlation"]["retry_handle_id"]
        == "second-handle"
    )


@pytest.mark.asyncio
async def test_expired_pending_voice_turn_is_retired_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state
    completion_receipt_attempts = 0

    async def fail_completion_receipt_save_once(thread_id: str, state: Any) -> None:
        nonlocal completion_receipt_attempts
        diagnostics = state.get("diagnostics", {})
        if diagnostics.get("voice_recorded_turn_receipts"):
            completion_receipt_attempts += 1
            if completion_receipt_attempts <= 2:
                raise RuntimeError("simulated stale pending turn failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_save_once,
    )
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-pending-turn-retirement",
        config=TraceConfig(enabled=True),
    )

    async with runtime:
        with pytest.raises(RuntimeError, match="stale pending turn failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-pending-retirement",
                user_id="user-1",
                user_text="expired user turn",
                assistant_text="expired assistant turn",
                correlation_hash="expired-correlation",
                request_hash="expired-request",
                llm_client=None,
            )

        pending_state = await runtime.get_state("voice-pending-retirement")
        assert pending_state is not None
        pending_turn = pending_state["diagnostics"]["voice_pending_turns"][
            "expired-correlation"
        ]
        pending_turn["last_attempted_at"] = "2000-01-01T00:00:00Z"
        await runtime._state_store.save_state("voice-pending-retirement", pending_state)

        with use_trace_context(trace_context, recorder):
            await runtime.voice.record_voice_turn(
                thread_id="voice-pending-retirement",
                user_id="user-1",
                user_text="current user turn",
                assistant_text="current assistant turn",
                correlation_hash="current-correlation",
                request_hash="current-request",
                llm_client=None,
            )
        state = await runtime.get_state("voice-pending-retirement")

        with pytest.raises(VoicePendingTurnRetiredError):
            await runtime.voice.record_voice_turn(
                thread_id="voice-pending-retirement",
                user_id="user-1",
                user_text="expired user turn",
                assistant_text="expired assistant turn",
                correlation_hash="expired-correlation",
                request_hash="expired-request",
                llm_client=None,
            )

    assert state is not None
    diagnostics = state["diagnostics"]
    assert "expired-correlation" not in diagnostics.get("voice_pending_turns", {})
    tombstone = diagnostics["voice_retired_pending_turns"]["expired-correlation"]
    assert tombstone["request_hash"] == "expired-request"
    assert isinstance(tombstone["retired_at"], str)
    retirement_event = next(
        event for event in recorder.events if event.name == VOICE_PENDING_TURNS_RETIRED
    )
    assert retirement_event.attributes["retired_count"] == 1
    assert retirement_event.attributes["retired_safety_count"] == 0
    assert retirement_event.attributes["retired_non_safety_count"] == 1
    assert retirement_event.attributes["max_age_seconds"] > 0
    assert "expired user turn" not in str(retirement_event.attributes)


@pytest.mark.asyncio
async def test_pending_voice_retry_survives_postgres_runtime_restart(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_paths = runtime_storage_paths(tmp_path)
    persistence_config = postgres_thread_persistence_config()
    thread_id = "voice-pending-restart"
    correlation_hash = "restart-correlation"
    request_hash = "restart-request"

    async with PersistentAgentRuntime(
        storage_paths=storage_paths,
        persistence_config=persistence_config,
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    ) as runtime_a:
        original_save_state = runtime_a._state_store.save_state

        async def fail_completion_receipt_save(thread_id: str, state: Any) -> None:
            diagnostics = state.get("diagnostics", {})
            if diagnostics.get("voice_recorded_turn_receipts"):
                raise RuntimeError("simulated restart receipt failure")
            await original_save_state(thread_id, state)

        monkeypatch.setattr(
            runtime_a._state_store,
            "save_state",
            fail_completion_receipt_save,
        )

        with pytest.raises(RuntimeError, match="restart receipt failure"):
            await runtime_a.voice.record_voice_turn(
                thread_id=thread_id,
                user_id="user-1",
                user_text="restart user turn",
                assistant_text="restart assistant turn",
                correlation_hash=correlation_hash,
                request_hash=request_hash,
                retry_handle_id="restart-handle",
                llm_client=None,
            )
        state = await runtime_a.get_state(thread_id)
        assert state is not None
        assert correlation_hash in state["diagnostics"]["voice_pending_turns"]

    async with PersistentAgentRuntime(
        storage_paths=storage_paths,
        persistence_config=persistence_config,
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    ) as runtime_b:
        state = await runtime_b.voice.record_voice_turn(
            thread_id=thread_id,
            user_id="user-1",
            user_text="restart user turn",
            assistant_text="restart assistant turn",
            correlation_hash=correlation_hash,
            request_hash=request_hash,
            retry_handle_id="restart-handle",
            llm_client=None,
        )
        history = await runtime_b.get_history(thread_id)

    assert "voice_pending_turns" not in state["diagnostics"]
    assert [message.content for message in history] == [
        "restart user turn",
        "restart assistant turn",
    ]


@pytest.mark.asyncio
async def test_retry_handle_heartbeat_protects_pending_voice_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state

    async def fail_completion_receipt_save(thread_id: str, state: Any) -> None:
        diagnostics = state.get("diagnostics", {})
        if diagnostics.get("voice_recorded_turn_receipts"):
            raise RuntimeError("simulated live retry failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_save,
    )
    stale_timestamp = "2000-01-01T00:00:00Z"

    async with runtime:
        with pytest.raises(RuntimeError, match="live retry failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-retry-handle",
                user_id="user-1",
                user_text="live user turn",
                assistant_text="live assistant turn",
                correlation_hash="live-correlation",
                request_hash="live-request",
                retry_handle_id="live-handle",
                llm_client=None,
            )

        pending_state = await runtime.get_state("voice-retry-handle")
        assert pending_state is not None
        pending_turn = pending_state["diagnostics"]["voice_pending_turns"][
            "live-correlation"
        ]
        pending_turn["last_attempted_at"] = stale_timestamp
        pending_turn["retry_handle_seen_at"] = stale_timestamp
        await runtime._state_store.save_state("voice-retry-handle", pending_state)
        await runtime.voice.touch_pending_voice_retry_handle(
            thread_id="voice-retry-handle",
            retry_handle_id="live-handle",
        )

        with pytest.raises(VoicePendingTurnHandleBusyError):
            await runtime.voice.record_voice_turn(
                thread_id="voice-retry-handle",
                user_id="user-1",
                user_text="competing user turn",
                assistant_text="competing assistant turn",
                correlation_hash="competing-correlation",
                request_hash="competing-request",
                retry_handle_id="live-handle",
                llm_client=None,
            )

        with pytest.raises(RuntimeError, match="live retry failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-retry-handle",
                user_id="user-1",
                user_text="new user turn",
                assistant_text="new assistant turn",
                correlation_hash="new-correlation",
                request_hash="new-request",
                retry_handle_id="new-handle",
                llm_client=None,
            )
        state = await runtime.get_state("voice-retry-handle")

    assert state is not None
    diagnostics = state["diagnostics"]
    assert len(diagnostics["voice_pending_turns"]) == 2
    assert "live-correlation" not in diagnostics.get("voice_retired_pending_turns", {})


@pytest.mark.asyncio
async def test_safety_pending_voice_turn_survives_extended_retry_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state

    async def fail_completion_receipt_save(thread_id: str, state: Any) -> None:
        diagnostics = state.get("diagnostics", {})
        if diagnostics.get("voice_recorded_turn_receipts"):
            raise RuntimeError("simulated safety retry failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_save,
    )
    assessment = CrisisAssessment(
        level=3,
        confidence="high",
        reason="verified safety interruption",
        needs_crisis_response=True,
        needs_clarification=False,
    )
    extended_outage = (
        (datetime.now(timezone.utc) - timedelta(days=3))
        .isoformat()
        .replace("+00:00", "Z")
    )

    async with runtime:
        with pytest.raises(RuntimeError, match="safety retry failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-safety-retention",
                user_id="user-1",
                user_text="I might hurt myself.",
                assistant_text="",
                outcome="safety_interrupted",
                correlation_hash="safety-correlation",
                request_hash="safety-request",
                retry_handle_id="safety-handle",
                safety_assessment=assessment,
                llm_client=None,
            )

        pending_state = await runtime.get_state("voice-safety-retention")
        assert pending_state is not None
        safety_pending_turn = pending_state["diagnostics"]["voice_pending_turns"][
            "safety-correlation"
        ]
        safety_pending_turn["created_at"] = extended_outage
        safety_pending_turn["last_attempted_at"] = extended_outage
        await runtime._state_store.save_state("voice-safety-retention", pending_state)

        with pytest.raises(RuntimeError, match="safety retry failure"):
            await runtime.voice.record_voice_turn(
                thread_id="voice-safety-retention",
                user_id="user-1",
                user_text="ordinary user turn",
                assistant_text="ordinary assistant turn",
                correlation_hash="ordinary-correlation",
                request_hash="ordinary-request",
                retry_handle_id="ordinary-handle",
                llm_client=None,
            )
        state = await runtime.get_state("voice-safety-retention")

    assert state is not None
    diagnostics = state["diagnostics"]
    assert "safety-correlation" in diagnostics["voice_pending_turns"]
    assert "safety-correlation" not in diagnostics.get(
        "voice_retired_pending_turns", {}
    )


@pytest.mark.asyncio
async def test_turn_endpoint_rejects_a_retired_pending_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    payload = {
        "thread_id": "voice-retired-retry-endpoint",
        "client_turn_id": "expired-client-turn",
        "memory_mode": "persistent",
        "user_text": "expired user turn",
        "assistant_text": "expired assistant turn",
    }
    request = VoiceTurnRecordRequest.model_validate(payload)
    correlation_hash = voice_routes._voice_turn_correlation_hash(
        thread_id=request.thread_id,
        client_turn_id=request.client_turn_id or "",
    )
    request_hash = voice_routes._voice_turn_request_hash(request)
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        await runtime._state_store.save_state(
            request.thread_id,
            {
                "diagnostics": {
                    "voice_retired_pending_turns": {
                        correlation_hash: {
                            "request_hash": request_hash,
                            "retired_at": "2000-01-01T00:00:00Z",
                        }
                    }
                }
            },
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/voice/realtime/turn", json=payload)

    assert response.status_code == 410
    assert response.json()["detail"] == {
        "code": "voice_realtime_turn_retry_expired",
        "message": "The retry window for this voice turn has expired. Start a new turn instead.",
    }


@pytest.mark.asyncio
async def test_legacy_pending_turn_retries_reuse_post_turn_safety_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state
    sdk_attempts = 0
    llm = _VoiceCrisisAuditLLM(level=3)

    async def fail_sdk_history_once(*args: Any, **kwargs: Any) -> None:
        nonlocal sdk_attempts
        sdk_attempts += 1
        if sdk_attempts == 1:
            raise RuntimeError("simulated pre-receipt lifecycle failure")

    monkeypatch.setattr(
        runtime.voice,
        "_collaboration",
        replace(
            runtime.voice._collaboration,
            ensure_sdk_turn_recorded=fail_sdk_history_once,
        ),
    )
    correlation_hash = "legacy-pending-correlation"
    request_hash = "legacy-pending-request"
    turn = {
        "thread_id": "voice-legacy-pending-retry",
        "user_id": "user-1",
        "user_text": "I had a difficult day.",
        "assistant_text": "I'm here with you.",
        "correlation_hash": correlation_hash,
        "request_hash": request_hash,
        "llm_client": llm,
    }

    async with runtime:
        with pytest.raises(RuntimeError, match="pre-receipt lifecycle failure"):
            await runtime.voice.record_voice_turn(**turn)

        pending_state = await runtime.get_state("voice-legacy-pending-retry")
        assert pending_state is not None
        pending_turn = pending_state["diagnostics"]["voice_pending_turns"][
            correlation_hash
        ]
        pending_turn.pop("turn_instance_id")
        await original_save_state("voice-legacy-pending-retry", pending_state)

        async def fail_completion_receipt_saves(
            thread_id: str,
            state: Any,
        ) -> None:
            diagnostics = state.get("diagnostics", {})
            if diagnostics.get("voice_recorded_turn_receipts"):
                raise RuntimeError("persistent completion receipt failure")
            await original_save_state(thread_id, state)

        monkeypatch.setattr(
            runtime._state_store,
            "save_state",
            fail_completion_receipt_saves,
        )

        for _ in range(2):
            with pytest.raises(
                RuntimeError,
                match="persistent completion receipt failure",
            ):
                await runtime.voice.record_voice_turn(**turn)
            assert await runtime.voice.drain_post_turn_safety_checks() == 0

        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    assert llm.crisis_calls == 1
    missed_crises = [
        record for record in audit_records if record.event_type == "voice_missed_crisis"
    ]
    assert len(missed_crises) == 1


@pytest.mark.asyncio
async def test_safety_interruption_retry_deduplicates_audit_after_receipt_save_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state
    save_attempts = 0

    async def fail_receipt_save_once(thread_id: str, state: Any) -> None:
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 2:
            raise RuntimeError("simulated receipt save failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(runtime._state_store, "save_state", fail_receipt_save_once)
    assessment = CrisisAssessment(
        level=3,
        confidence="high",
        reason="verified voice safety interruption",
        needs_crisis_response=True,
        needs_clarification=False,
    )
    turn = {
        "thread_id": "voice-safety-audit-retry",
        "user_id": "user-1",
        "user_text": "I might hurt myself.",
        "assistant_text": "",
        "outcome": "safety_interrupted",
        "correlation_hash": "stable-safety-correlation",
        "request_hash": "stable-safety-request",
        "safety_assessment": assessment,
        "llm_client": None,
    }
    recorder = InMemoryTraceRecorder()
    trace_context = TraceContext(
        trace_id="voice-completion-metadata-failure",
        config=TraceConfig(enabled=True),
    )

    async with runtime:
        with use_trace_context(trace_context, recorder):
            first_state = await runtime.voice.record_voice_turn(**turn)
        assert await runtime.crisis_log_backend.arecord_count() == 1
        persisted_state = await runtime.get_state("voice-safety-audit-retry")
        assert persisted_state is not None
        assert "voice_pending_turns" not in persisted_state["diagnostics"]

        state = await runtime.voice.record_voice_turn(**turn)
        records = await utc_crisis_records(runtime.crisis_log_backend)
        history = await runtime.get_history("voice-safety-audit-retry")

    assert save_attempts == 3
    assert first_state["diagnostics"]["voice_post_turn_safety"] == {
        "scheduled": False,
        "status": "skipped",
        "reason": "safety_interruption_verified",
        "pending_count": 0,
    }
    assert len(records) == 1
    assert records[0].id == ("voice-safety-interruption:stable-safety-correlation")
    assert [message.content for message in history] == ["I might hurt myself."]
    assert "voice_pending_turns" not in state["diagnostics"]
    failure_event = next(
        event
        for event in recorder.events
        if event.name == VOICE_TURN_COMPLETION_METADATA_PERSIST_FAILED
    )
    assert failure_event.attributes == {
        "voice_runtime": "openai_realtime",
        "outcome": "safety_interrupted",
        "route": "voice_safety_interrupted",
        "memory_mode": "local",
        "error_type": "RuntimeError",
        "correlation_hash": "stable-safety-correlation",
    }
    assert "hurt myself" not in str(failure_event.attributes)


@pytest.mark.asyncio
async def test_turn_endpoint_succeeds_when_completion_metadata_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state
    save_attempts = 0

    async def fail_completion_metadata_save_once(thread_id: str, state: Any) -> None:
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 2:
            raise RuntimeError("simulated completion metadata save failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_metadata_save_once,
    )
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    llm = _VoiceCrisisAuditLLM(level=3)
    app.dependency_overrides[get_llm_client] = lambda: llm
    payload = {
        "thread_id": "voice-completion-metadata-api",
        "client_turn_id": "voice-completion-metadata-turn",
        "user_id": "user-1",
        "user_text": "I had a difficult day.",
        "assistant_text": "I'm here with you.",
        "outcome": "completed",
    }

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first = await client.post("/api/voice/realtime/turn", json=payload)
            persisted_state = await runtime.get_state("voice-completion-metadata-api")
            assert await runtime.voice.drain_post_turn_safety_checks() == 0
            retry = await client.post("/api/voice/realtime/turn", json=payload)
            conflict = await client.post(
                "/api/voice/realtime/turn",
                json={**payload, "user_text": "Different text."},
            )
        history = await runtime.get_history("voice-completion-metadata-api")
        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    assert first.status_code == 200
    assert first.json() == {
        "recorded": True,
        "thread_id": "voice-completion-metadata-api",
        "message_count": 2,
        "post_turn_safety": {
            "scheduled": True,
            "status": "scheduled",
            "reason": None,
            "pending_count": 1,
        },
    }
    assert persisted_state is not None
    persisted_diagnostics = persisted_state["diagnostics"]
    assert len(persisted_diagnostics["voice_recorded_turn_hashes"]) == 1
    assert "voice_pending_turns" not in persisted_diagnostics
    receipt = next(iter(persisted_diagnostics["voice_recorded_turn_receipts"].values()))
    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == (
        "voice_realtime_turn_idempotency_conflict"
    )
    assert save_attempts == 3
    assert [message.content for message in history] == [
        "I had a difficult day.",
        "I'm here with you.",
    ]
    assert llm.crisis_calls == 1
    assert len(audit_records) == 1
    assert audit_records[0].event_type == "voice_missed_crisis"
    thread_hash = hashlib.sha256(b"voice-completion-metadata-api").hexdigest()
    turn_instance_id = receipt["turn_instance_id"]
    assert audit_records[0].id == (
        f"voice-missed-crisis:{thread_hash}:{turn_instance_id}"
    )


@pytest.mark.asyncio
async def test_reused_correlation_after_receipt_eviction_gets_new_audit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.voice.runtime_facade._MAX_RECORDED_VOICE_TURN_HASHES",
        1,
    )
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=3)
    common = {
        "thread_id": "voice-reused-correlation",
        "user_id": "user-1",
        "assistant_text": "I'm here with you.",
    }

    async with runtime:
        await runtime.voice.record_voice_turn(
            **common,
            user_text="First accepted turn.",
            correlation_hash="recycled-correlation",
            request_hash="first-request",
            llm_client=llm,
        )
        assert await runtime.voice.drain_post_turn_safety_checks() == 0
        await runtime.voice.record_voice_turn(
            **common,
            user_text="Intervening turn.",
            correlation_hash="intervening-correlation",
            request_hash="intervening-request",
            llm_client=None,
        )
        await runtime.voice.record_voice_turn(
            **common,
            user_text="Second accepted turn.",
            correlation_hash="recycled-correlation",
            request_hash="second-request",
            llm_client=llm,
        )
        assert await runtime.voice.drain_post_turn_safety_checks() == 0
        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    missed_crises = [
        record for record in audit_records if record.event_type == "voice_missed_crisis"
    ]
    assert llm.crisis_calls == 2
    assert len(missed_crises) == 2
    assert len({record.id for record in missed_crises}) == 2


@pytest.mark.asyncio
async def test_turn_endpoint_fails_when_completion_receipt_cannot_be_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )
    original_save_state = runtime._state_store.save_state
    save_attempts = 0

    async def fail_completion_receipt_saves(thread_id: str, state: Any) -> None:
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts >= 2:
            raise RuntimeError("persistent completion receipt failure")
        await original_save_state(thread_id, state)

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_completion_receipt_saves,
    )
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "voice-persistent-receipt-failure",
                    "client_turn_id": "voice-persistent-receipt-failure-turn",
                    "user_text": "I had a difficult day.",
                    "assistant_text": "I'm here with you.",
                },
            )
        state = await runtime.get_state("voice-persistent-receipt-failure")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "voice_realtime_turn_record_failed",
        "message": "persistent completion receipt failure",
    }
    assert save_attempts == 3
    assert state is not None
    assert len(state["transcript"]) == 2
    assert len(state["diagnostics"]["voice_pending_turns"]) == 1


@pytest.mark.asyncio
async def test_reset_thread_reuse_gets_new_post_turn_safety_identity() -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=3)
    turn = {
        "thread_id": "voice-reset-reuse",
        "user_id": "user-1",
        "user_text": "A difficult turn.",
        "assistant_text": "I'm here with you.",
        "correlation_hash": "reused-after-reset",
        "request_hash": "same-request-after-reset",
        "llm_client": llm,
    }

    async with runtime:
        await runtime.voice.record_voice_turn(**turn)
        assert await runtime.voice.drain_post_turn_safety_checks() == 0
        await runtime.end_session("voice-reset-reuse")
        await runtime.reset_thread("voice-reset-reuse")
        await runtime.voice.record_voice_turn(**turn)
        assert await runtime.voice.drain_post_turn_safety_checks() == 0
        audit_records = await utc_crisis_records(runtime.crisis_log_backend)

    missed_crises = [
        record for record in audit_records if record.event_type == "voice_missed_crisis"
    ]
    assert llm.crisis_calls == 2
    assert len(missed_crises) == 2
    assert len({record.id for record in missed_crises}) == 2


@pytest.mark.asyncio
async def test_turn_endpoint_still_fails_when_canonical_state_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )

    async def fail_canonical_state_save(thread_id: str, state: Any) -> None:
        del thread_id, state
        raise RuntimeError("simulated canonical state save failure")

    monkeypatch.setattr(
        runtime._state_store,
        "save_state",
        fail_canonical_state_save,
    )
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/turn",
                json={
                    "thread_id": "voice-canonical-save-failure",
                    "client_turn_id": "voice-canonical-save-failure-turn",
                    "user_text": "I had a difficult day.",
                    "assistant_text": "I'm here with you.",
                },
            )
        state = await runtime.get_state("voice-canonical-save-failure")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "voice_realtime_turn_record_failed",
        "message": "simulated canonical state save failure",
    }
    assert state is None


@pytest.mark.asyncio
async def test_non_crisis_voice_turn_writes_no_audit_record() -> None:
    """An ordinary voice turn must not produce a crisis audit record."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
        state = await runtime.voice.record_voice_turn(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="I had a rough day at work.",
            assistant_text="That sounds draining. Want to talk it through?",
            response_style="supportive",
            llm_client=None,
        )

        pending = await runtime.voice.drain_post_turn_safety_checks()
        count = await runtime.crisis_log_backend.arecord_count()

    assert pending == 0
    assert count == 0
    assert state["diagnostics"]["voice_post_turn_safety"] == {
        "scheduled": False,
        "status": "skipped",
        "reason": "no_llm_client",
        "pending_count": 0,
    }


@pytest.mark.asyncio
async def test_post_turn_voice_classifier_writes_missed_crisis_audit_record() -> None:
    """A post-turn voice classifier miss is audited without changing latency path."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(
        level=2,
        reason="User explicitly described suicidal ideation over voice.",
    )

    async with runtime:
        state = await runtime.voice.record_voice_turn(
            thread_id="voice-missed-crisis",
            user_id="user-1",
            user_text="I've been thinking about ending it all.",
            assistant_text="That sounds incredibly heavy. I'm here with you.",
            response_style="supportive",
            llm_client=llm,
        )
        pending = await runtime.voice.drain_post_turn_safety_checks()
        records = await runtime.crisis_log_backend.alist_by_date(
            datetime.now(timezone.utc).date()
        )

    assert pending == 0
    assert state["diagnostics"]["voice_post_turn_safety"] == {
        "scheduled": True,
        "status": "scheduled",
        "reason": None,
        "pending_count": 1,
    }
    assert llm.crisis_calls == 1
    assert len(records) == 1
    record = records[0]
    assert record.event_type == "voice_missed_crisis"
    assert record.level == 2
    assert record.reason == "User explicitly described suicidal ideation over voice."
    assert record.classifier_path == "voice_post_turn"
    assert record.response_node_completed is False
    assert record.response_style == "supportive"
    assert record.resource_lookup_status == "not_attempted"
    assert record.resource_count == 0
    assert record.response_path == "not_routed"
    assert record.fallback_reason == "voice_realtime_crisis_tool_not_called"
    assert record.trace_runtime_mode == "voice"


@pytest.mark.asyncio
async def test_post_turn_voice_classifier_writes_no_record_for_safe_turn() -> None:
    """A safe post-turn classifier result must not create audit noise."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = FakeCrossRestartLLM()

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-safe-post-turn",
            user_id="user-1",
            user_text="I had a rough day at work.",
            assistant_text="That sounds draining. Want to talk it through?",
            response_style="supportive",
            llm_client=llm,
        )
        pending = await runtime.voice.drain_post_turn_safety_checks()
        count = await runtime.crisis_log_backend.arecord_count()

    assert pending == 0
    assert llm.crisis_calls == 1
    assert count == 0


@pytest.mark.asyncio
async def test_post_turn_voice_classifier_skips_existing_crisis_route() -> None:
    """Realtime crisis routing already produces the only audit record."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=3)

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-existing-crisis",
            user_id="user-1",
            user_text="I might hurt myself tonight.",
            assistant_text="Your safety matters. Let's get immediate support.",
            tool_calls=[
                {
                    "tool_name": "lookup_crisis_resources",
                    "status": "completed",
                    "output": {
                        "inferred_location": "Singapore",
                        "found_resources": [],
                        "resource_lookup_status": "lookup_error",
                    },
                }
            ],
            llm_client=llm,
        )
        pending = await runtime.voice.drain_post_turn_safety_checks()
        records = await runtime.crisis_log_backend.alist_by_date(
            datetime.now(timezone.utc).date()
        )

    assert pending == 0
    assert llm.crisis_calls == 0
    assert len(records) == 1
    assert records[0].event_type == "crisis_response"
    assert records[0].classifier_path == "llm_primary"
    assert records[0].response_node_completed is True


@pytest.mark.asyncio
async def test_post_turn_safety_drain_waits_for_background_classifier() -> None:
    """The voice facade exposes a deterministic drain for scheduled checks."""

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    llm = _VoiceCrisisAuditLLM(level=2, delay_seconds=0.05)

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-drain-post-turn",
            user_id="user-1",
            user_text="I want to die.",
            assistant_text="I'm sorry it feels this painful.",
            response_style="supportive",
            llm_client=llm,
        )
        assert runtime.voice.post_turn_safety_pending_count == 1
        pending = await runtime.voice.drain_post_turn_safety_checks(timeout_seconds=1.0)
        count = await runtime.crisis_log_backend.arecord_count()

    assert pending == 0
    assert count == 1


@pytest.mark.asyncio
async def test_voice_crisis_turn_records_lookup_error_status() -> None:
    """A crisis turn whose lookup hit an outage audits lookup_error, not empty.

    Distinguishing a transient lookup failure from a true "no resources"
    result is the whole point of the ``lookup_error`` status; the audit log
    must preserve that distinction for review.
    """

    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id="voice-crisis-error",
            user_id="user-1",
            inferred_location="Singapore",
            found_resources=[],
            resource_lookup_status="lookup_error",
        )
        await runtime.voice.record_voice_turn(
            thread_id="voice-crisis-error",
            user_id="user-1",
            user_text="I don't think I can stay safe.",
            assistant_text="Your safety matters most. Let's get you immediate support.",
            tool_calls=[
                {
                    "tool_name": "lookup_crisis_resources",
                    "status": "completed",
                    "output": {
                        "inferred_location": "Singapore",
                        "found_resources": [],
                        "resource_lookup_status": "lookup_error",
                    },
                }
            ],
            llm_client=None,
        )

        records = await runtime.crisis_log_backend.alist_by_date(
            datetime.now(timezone.utc).date()
        )

    assert len(records) == 1
    assert records[0].resource_lookup_status == "lookup_error"
    assert records[0].resource_count == 0


@pytest.mark.asyncio
async def test_voice_end_endpoint_uses_runtime_session_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-thread",
            user_id=None,
            user_text="I feel overwhelmed.",
            assistant_text="That sounds like a lot to carry.",
            response_style="supportive",
            llm_client=None,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/end",
                json={"thread_id": "voice-thread", "memory_mode": "persistent"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["finalized"] is False
    assert data["summary"] is None
    assert (
        data["detail"]
        == "No summary produced (session too short, no LLM, or incognito mode)."
    )
    assert data["themes"] == []
    assert data["mood_opened"] is None
    assert data["mood_closed"] is None
    assert data["turn_count"] is None
    assert data["open_loops"] == []
    assert data["resolved_threads"] == []


@pytest.mark.asyncio
async def test_voice_end_endpoint_summarizes_persistent_voice_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    fake_llm = FakeCrossRestartLLM()

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            for user_text, assistant_text in [
                (
                    "I argued with Sarah and feel tense.",
                    "That sounds painful. Let's slow it down together.",
                ),
                (
                    "I want to remember that family conflict is a big trigger.",
                    "That is useful context to carry forward.",
                ),
            ]:
                turn_response = await client.post(
                    "/api/voice/realtime/turn",
                    json={
                        "thread_id": "voice-thread",
                        "user_id": "user-1",
                        "memory_mode": "persistent",
                        "user_text": user_text,
                        "assistant_text": assistant_text,
                        "response_style": "supportive",
                    },
                )
                assert turn_response.status_code == 200

            response = await client.post(
                "/api/voice/realtime/end",
                json={"thread_id": "voice-thread", "memory_mode": "persistent"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["finalized"] is True
    assert data["summary"] == "User mentioned their sister Sarah in passing."
    assert data["detail"] == "Session summary produced."
    assert isinstance(data["themes"], list)
    assert data["mood_opened"] is not None
    assert data["mood_closed"] is not None
    assert data["turn_count"] is not None
    assert isinstance(data["open_loops"], list)
    assert isinstance(data["resolved_threads"], list)
    assert fake_llm.summarization_calls == 1


@pytest.mark.asyncio
async def test_voice_end_with_positive_feedback_writes_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-feedback-thread",
            user_id="user-1",
            user_text="I want to wrap up for today.",
            assistant_text="We can close here and note what mattered.",
            response_style="supportive",
            llm_client=None,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/end",
                json={
                    "thread_id": "voice-feedback-thread",
                    "memory_mode": "persistent",
                    "feedback": "positive",
                },
            )

        records = await runtime.session_feedback_backend.alist_by_session(
            hash_session_id("voice-feedback-thread")
        )

    assert response.status_code == 200
    assert len(records) == 1
    assert records[0].label == "positive"
    assert records[0].source == "api_end"
    assert records[0].modality == "voice"


@pytest.mark.asyncio
async def test_incognito_voice_end_with_feedback_scrubs_user_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        await runtime.voice.record_voice_turn(
            thread_id="voice-incognito-feedback",
            user_id="private-user",
            user_text="I want to wrap up privately.",
            assistant_text="We can close here without saving durable memory.",
            response_style="supportive",
            llm_client=None,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/end",
                json={
                    "thread_id": "voice-incognito-feedback",
                    "memory_mode": "incognito",
                    "feedback": "positive",
                },
            )

        records = await runtime.session_feedback_backend.alist_by_session(
            hash_session_id("voice-incognito-feedback")
        )

    assert response.status_code == 200
    assert len(records) == 1
    assert records[0].label == "positive"
    assert records[0].source == "api_end"
    assert records[0].modality == "voice"
    assert records[0].user_id_or_null is None


@pytest.mark.asyncio
async def test_voice_end_rejects_invalid_feedback_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with runtime:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/voice/realtime/end",
                json={
                    "thread_id": "voice-feedback-thread",
                    "memory_mode": "persistent",
                    "feedback": "awesome",
                },
            )

        feedback_count = await runtime.session_feedback_backend.arecord_count()

    assert response.status_code == 422
    assert feedback_count == 0
