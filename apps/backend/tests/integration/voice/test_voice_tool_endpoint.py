from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection


class _FakeRuntime:
    pass


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    runtime = _FakeRuntime()

    monkeypatch.setattr(
        voice_routes,
        "get_runtime_selection",
        lambda mode: runtime_selection(runtime, mode),
    )
    app.dependency_overrides[get_llm_client] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_voice_tool_endpoint_dispatches_app_owned_tool(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_voice_tool_call(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"response_text": "Memory is enabled.", "side_effect": "none"}

    monkeypatch.setattr(
        "agent.voice.tools.execute_voice_tool_call",
        fake_execute_voice_tool_call,
    )

    response = await client.post(
        "/api/voice/realtime/tools",
        json={
            "thread_id": "voice-thread",
            "user_id": "user-1",
            "current_user_message": "What do you remember?",
            "transcript": [{"role": "user", "content": "What do you remember?"}],
            "memory_mode": "persistent",
            "tool_name": "show_memory_status",
            "arguments": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["output"]["response_text"] == "Memory is enabled."
    assert captured["thread_id"] == "voice-thread"
    assert captured["tool_name"] == "show_memory_status"
    assert captured["arguments"] == {}
    assert captured["memory_mode"] == "persistent"


@pytest.mark.asyncio
async def test_voice_tool_endpoint_bounds_execution_time(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def slow_execute_voice_tool_call(**kwargs: object) -> dict[str, object]:
        del kwargs
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    monkeypatch.setattr(
        "agent.voice.tools.execute_voice_tool_call",
        slow_execute_voice_tool_call,
    )
    monkeypatch.setattr(voice_routes, "_VOICE_TOOL_TIMEOUT_SECONDS", 0.001)

    response = await client.post(
        "/api/voice/realtime/tools",
        json={
            "thread_id": "voice-thread",
            "memory_mode": "persistent",
            "tool_name": "show_memory_status",
            "arguments": {},
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "voice_realtime_tool_failed"
    assert cancelled.is_set()
