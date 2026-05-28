from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection


class _FakeSessionRuntime:
    def __init__(self) -> None:
        self.seen_memory_mode: str | None = None

    async def voice_session_memory_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        memory_mode: str,
    ) -> str:
        self.seen_memory_mode = memory_mode
        if memory_mode == "incognito":
            return ""
        return "PRIVATE SAVED MEMORY SHOULD APPEAR"


@pytest.mark.asyncio
async def test_incognito_voice_session_does_not_bootstrap_persistent_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeSessionRuntime()
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )

    async def fake_create_realtime_client_secret(
        *,
        session_config: dict[str, object],
        safety_identifier: str | None,
    ) -> str:
        return "ek_test_secret"

    monkeypatch.setattr(
        "agent.voice.realtime.create_realtime_client_secret",
        fake_create_realtime_client_secret,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/voice/realtime/session",
            json={
                "thread_id": "voice-thread",
                "user_id": "user-1",
                "memory_mode": "incognito",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert runtime.seen_memory_mode == "incognito"
    assert (
        "PRIVATE SAVED MEMORY SHOULD APPEAR"
        not in body["session_config"]["instructions"]
    )


@pytest.mark.asyncio
async def test_incognito_voice_turn_request_records_ephemeral_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
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
                    "thread_id": "voice-incognito-thread",
                    "user_id": "user-1",
                    "memory_mode": "incognito",
                    "user_text": "This should stay off the server record.",
                    "assistant_text": "I can stay with you in this moment.",
                },
            )
        state = await runtime.get_state("voice-incognito-thread")

    assert response.status_code == 200
    assert response.json()["recorded"] is True
    assert state is not None
    assert state["message"] == "This should stay off the server record."


@pytest.mark.asyncio
async def test_persistent_voice_turn_request_still_persists_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.LOCAL,
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
                    "thread_id": "voice-persistent-thread",
                    "user_id": "user-1",
                    "memory_mode": "persistent",
                    "user_text": "Please keep this in the thread.",
                    "assistant_text": "I will keep the session context available.",
                },
            )
        state = await runtime.get_state("voice-persistent-thread")

    assert response.status_code == 200
    assert response.json()["recorded"] is True
    assert state is not None
    assert state["message"] == "Please keep this in the thread."


class _FakeEndRuntime:
    def __init__(self) -> None:
        self.called = False
        self.feedback_called = False

    async def record_session_feedback(
        self, thread_id: str, *, label: str, source: str, modality: str = "text"
    ) -> None:
        self.feedback_called = True
        assert thread_id == "voice-thread"
        assert label == "positive"
        assert source == "api_end"
        assert modality == "voice"

    async def end_session(self, thread_id: str, *, llm_client: object | None) -> Any:
        self.called = True
        assert thread_id == "voice-thread"
        return None


@pytest.mark.asyncio
async def test_incognito_voice_end_request_records_feedback_before_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEndRuntime()
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/voice/realtime/end",
            json={
                "thread_id": "voice-thread",
                "memory_mode": "incognito",
                "feedback": "positive",
            },
        )

    assert response.status_code == 200
    assert runtime.called is True
    assert runtime.feedback_called is True
    data = response.json()
    assert data["finalized"] is False
    assert data["summary"] is None
    assert data["detail"] == "Incognito session ended without durable finalization."
    assert data["themes"] == []
    assert data["mood_opened"] is None
    assert data["mood_closed"] is None
    assert data["turn_count"] is None
    assert data["open_loops"] == []
    assert data["resolved_threads"] == []
