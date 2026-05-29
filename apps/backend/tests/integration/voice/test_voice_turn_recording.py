from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent.memory.hashing import hash_session_id
from agent.memory.modes import MemoryMode
from agent.models import MessageRole
from agent.runtime import PersistentAgentRuntime
from api.dependencies import get_llm_client
from api.router import api_router
from api.routes import voice as voice_routes
from tests.support.api_selection import runtime_selection
from tests.support.persistence import FakeCrossRestartLLM


@pytest.mark.asyncio
async def test_record_voice_turn_persists_thread_history() -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )

    async with runtime:
        await runtime.record_voice_turn(
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
                    "thread_id": "voice-thread",
                    "memory_mode": "persistent",
                    "user_text": "I feel overwhelmed.",
                    "assistant_text": "That sounds like a lot to carry.",
                    "response_style": "supportive",
                },
            )

        history = await runtime.get_history("voice-thread")

    assert response.status_code == 200
    assert response.json()["recorded"] is True
    assert response.json()["message_count"] == 2
    assert [message.content for message in history] == [
        "I feel overwhelmed.",
        "That sounds like a lot to carry.",
    ]


@pytest.mark.asyncio
async def test_voice_turn_endpoint_infers_route_and_tool_metadata(
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
async def test_voice_end_endpoint_uses_runtime_session_finalization(
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
        await runtime.record_voice_turn(
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
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.LOCAL,
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
        await runtime.record_voice_turn(
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
        await runtime.record_voice_turn(
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
