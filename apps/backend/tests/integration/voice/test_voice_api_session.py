from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.router import api_router
from api.routes import voice as voice_routes


class _FakeRuntime:
    async def voice_session_memory_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        memory_mode: str,
    ) -> str:
        return ""


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    runtime = _FakeRuntime()
    monkeypatch.setattr(voice_routes, "get_runtime_for_memory_mode", lambda _: runtime)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_create_voice_realtime_session_returns_client_secret(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_realtime_client_secret(
        *,
        session_config: dict[str, object],
        safety_identifier: str | None,
    ) -> str:
        captured["session_config"] = session_config
        captured["safety_identifier"] = safety_identifier
        return "ek_test_secret"

    monkeypatch.setattr(
        "agent.voice.realtime.create_realtime_client_secret",
        fake_create_realtime_client_secret,
    )

    response = await client.post(
        "/api/voice/realtime/session",
        json={
            "thread_id": "voice-thread",
            "user_id": "user-1",
            "memory_mode": "persistent",
            "assistant_voice": "cedar",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["client_secret"] == "ek_test_secret"
    assert body["thread_id"] == "voice-thread"
    assert body["user_id"] == "user-1"
    assert body["memory_mode"] == "persistent"
    assert body["session_config"]["model"] == "gpt-realtime-2"
    assert body["session_config"]["audio"]["output"]["voice"] == "cedar"
    assert captured["session_config"]["model"] == "gpt-realtime-2"
    assert captured["session_config"]["audio"]["output"]["voice"] == "cedar"
    assert captured["safety_identifier"]
    assert len(captured["safety_identifier"]) <= 64


@pytest.mark.asyncio
async def test_create_voice_realtime_session_defaults_to_persistent(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_realtime_client_secret(
        *,
        session_config: dict[str, object],
        safety_identifier: str | None,
    ) -> str:
        assert "persistent" in str(session_config["instructions"]).lower()
        assert safety_identifier
        return "ek_test_secret"

    monkeypatch.setattr(
        "agent.voice.realtime.create_realtime_client_secret",
        fake_create_realtime_client_secret,
    )

    response = await client.post(
        "/api/voice/realtime/session",
        json={"thread_id": "voice-thread"},
    )

    assert response.status_code == 200
    assert response.json()["memory_mode"] == "persistent"


@pytest.mark.asyncio
async def test_create_voice_realtime_session_includes_runtime_memory_context(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_context(
        self: object,
        *,
        thread_id: str,
        user_id: str | None,
        memory_mode: str,
    ) -> str:
        assert thread_id == "voice-thread"
        assert user_id == "user-1"
        assert memory_mode == "persistent"
        return "Saved preferences: use concise replies."

    async def fake_create_realtime_client_secret(
        *,
        session_config: dict[str, object],
        safety_identifier: str | None,
    ) -> str:
        captured["instructions"] = session_config["instructions"]
        captured["safety_identifier"] = safety_identifier
        return "ek_test_secret"

    monkeypatch.setattr(_FakeRuntime, "voice_session_memory_context", fake_context)
    monkeypatch.setattr(
        "agent.voice.realtime.create_realtime_client_secret",
        fake_create_realtime_client_secret,
    )

    response = await client.post(
        "/api/voice/realtime/session",
        json={
            "thread_id": "voice-thread",
            "user_id": "user-1",
            "memory_mode": "persistent",
        },
    )

    assert response.status_code == 200
    assert "Saved preferences: use concise replies." in captured["instructions"]
    assert captured["safety_identifier"]
