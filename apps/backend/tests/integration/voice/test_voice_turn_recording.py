from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.hashing import hash_session_id
from agent.memory.modes import MemoryMode
from agent.models import MessageRole
from agent.runtime import PersistentAgentRuntime, RuntimeBehaviorConfig
from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection
from tests.support.persistence import (
    FakeCrossRestartLLM,
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)
from tests.support.safety_capture import (
    CRISIS_VOICE_RESPONSE_TEXT,
    CRISIS_VOICE_USER_TEXT,
    utc_crisis_records,
    voice_crisis_lookup_tool_call,
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
async def test_voice_crisis_turn_writes_one_audit_record() -> None:
    """A voice turn that called lookup_crisis_resources is audited like text.

    The live Realtime crisis route is prompt/tool driven, so this in-turn
    audit record is based on the model's crisis tool call. Missed non-crisis
    routes are checked separately by the post-turn safety auditor. The runtime
    must still write exactly one ``CrisisLogRecord`` here so a crisis over voice
    is as auditable as one over text, and the verified resource status must
    thread into the record.
    """

    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
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

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
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
            runtime,
            "_ensure_openai_sdk_turn_recorded",
            fail_sdk_history,
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
async def test_non_crisis_voice_turn_writes_no_audit_record() -> None:
    """An ordinary voice turn must not produce a crisis audit record."""

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
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

    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )

    async with runtime:
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
    runtime = PersistentAgentRuntime(
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
