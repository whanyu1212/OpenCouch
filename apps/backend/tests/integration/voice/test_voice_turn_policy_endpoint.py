from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection


@pytest.mark.asyncio
async def test_voice_turn_policy_endpoint_returns_route(
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
                "/api/voice/realtime/turn-policy",
                json={
                    "thread_id": "voice-thread",
                    "user_id": "user-1",
                    "user_text": "Can you look up the current 988 guidance?",
                    "memory_mode": "persistent",
                },
            )

    assert response.status_code == 200
    assert response.json()["route"] == "grounded_lookup"
    assert response.json()["response_style"] == "grounded_lookup"
    assert response.json()["required_tool_name"] == "answer_grounded_lookup"
