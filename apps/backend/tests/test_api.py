"""Tests for the FastAPI API layer.

Uses FastAPI's TestClient with a test-scoped lifespan that wires an
in-memory runtime (no SQLite on disk, no API keys needed). Each test
gets a clean runtime so there's no state bleed between tests.

The tests focus on the HTTP contract (status codes, response shapes,
error handling) rather than the agent's therapeutic quality — that's
what the eval harnesses are for. The API layer is a thin wrapper;
these tests verify the wrapping is correct.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from agent.persistence import PersistentAgentRuntime


@pytest.fixture
async def runtime():
    """Yield a fresh in-memory runtime for each test."""

    rt = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    )
    async with rt:
        yield rt


@pytest.fixture
async def client(runtime):
    """Yield an async HTTP client wired to a test FastAPI app.

    The app uses the same routes as the real app but with the
    runtime and LLM client injected from the test fixture rather
    than from the lifespan handler. This avoids needing real SQLite
    files or API keys for tests.
    """

    from fastapi import FastAPI

    from api.dependencies import get_llm_client, get_runtime
    from api.router import api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api")

    # Override dependencies with test instances
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[get_llm_client] = lambda: None  # deterministic mode

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Health ──────────────────────────────────────────────────────────


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client) -> None:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Chat ────────────────────────────────────────────────────────────


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_response(self, client) -> None:
        """A basic chat request should return a valid ChatResponse."""

        resp = await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "test-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response_text" in data
        assert len(data["response_text"]) > 0
        assert data["response_type"] in ("therapeutic", "crisis")
        assert "crisis" in data
        assert "diagnostics" in data

    @pytest.mark.asyncio
    async def test_chat_requires_message(self, client) -> None:
        """Missing message should return 422 validation error."""

        resp = await client.post(
            "/api/chat",
            json={"thread_id": "test-1"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_requires_thread_id(self, client) -> None:
        """Missing thread_id should return 422 validation error."""

        resp = await client.post(
            "/api/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_with_user_id(self, client) -> None:
        """user_id should be accepted and passed through."""

        resp = await client.post(
            "/api/chat",
            json={
                "message": "hello",
                "thread_id": "test-2",
                "user_id": "alice",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_continuity_across_turns(self, client) -> None:
        """Two turns on the same thread should maintain context."""

        resp1 = await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "test-cont"},
        )
        assert resp1.status_code == 200

        resp2 = await client.post(
            "/api/chat",
            json={"message": "how are you", "thread_id": "test-cont"},
        )
        assert resp2.status_code == 200
        # The second turn should succeed — the thread state persisted


# ── Threads ─────────────────────────────────────────────────────────


class TestThreads:
    @pytest.mark.asyncio
    async def test_list_threads_empty(self, client) -> None:
        """A fresh runtime should have no threads."""

        resp = await client.get("/api/threads")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_threads_after_chat(self, client) -> None:
        """After a chat turn, the thread should appear in the list."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "thread-list-test"},
        )
        resp = await client.get("/api/threads")
        assert resp.status_code == 200
        threads = resp.json()
        assert len(threads) >= 1
        assert any(t["thread_id"] == "thread-list-test" for t in threads)

    @pytest.mark.asyncio
    async def test_get_state_returns_dict(self, client) -> None:
        """State endpoint should return the raw state dict."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "state-test"},
        )
        resp = await client.get("/api/threads/state-test/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "message" in state
        assert "transcript" in state

    @pytest.mark.asyncio
    async def test_get_state_404_for_unknown_thread(self, client) -> None:
        """State for a non-existent thread should return 404."""

        resp = await client.get("/api/threads/nonexistent/state")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_history_returns_messages(self, client) -> None:
        """History should return transcript messages with role and content."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "hist-test"},
        )
        resp = await client.get("/api/threads/hist-test/history")
        assert resp.status_code == 200
        messages = resp.json()
        assert len(messages) >= 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["content"]) > 0

    @pytest.mark.asyncio
    async def test_get_history_empty_thread(self, client) -> None:
        """History for an unused thread returns an empty list."""

        resp = await client.get("/api/threads/no-history/history")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_history_includes_mode(self, client) -> None:
        """Assistant messages should carry the routing mode."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "mode-test"},
        )
        resp = await client.get("/api/threads/mode-test/history")
        messages = resp.json()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        # Mode should be a string (the therapeutic mode that shaped the reply)
        assert assistant_msgs[0]["mode"] is not None


# ── Memory ──────────────────────────────────────────────────────────


class TestMemory:
    @pytest.mark.asyncio
    async def test_memory_status_returns_counts(self, client) -> None:
        """Status should return namespace counts and recall toggle."""

        resp = await client.get(
            "/api/memory/status",
            params={"thread_id": "mem-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "counts" in data
        assert "semantic" in data["counts"]
        assert "episodic" in data["counts"]
        assert "procedural" in data["counts"]
        assert "proactive_recall_enabled" in data

    @pytest.mark.asyncio
    async def test_delete_fact_404_when_empty(self, client) -> None:
        """Deleting a fact when none exist should return 404."""

        resp = await client.delete(
            "/api/memory/facts/1",
            params={"thread_id": "empty-thread"},
        )
        assert resp.status_code == 404
        assert "No semantic facts" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_session_404_when_empty(self, client) -> None:
        """Deleting a session when none exist should return 404."""

        resp = await client.delete(
            "/api/memory/sessions/1",
            params={"thread_id": "empty-thread"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_rule_404_when_empty(self, client) -> None:
        """Deleting a rule when none exist should return 404."""

        resp = await client.delete(
            "/api/memory/rules/1",
            params={"thread_id": "empty-thread"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_fact_out_of_range(self, client, runtime) -> None:
        """Deleting a fact beyond the range should return 404."""

        # Seed one fact directly
        await runtime.memory_store.aput(
            ("seed-thread", "semantic"),
            "fact-1",
            {"evidence_quote": "test fact", "created_at": "2026-01-01T00:00:00Z"},
        )
        resp = await client.delete(
            "/api/memory/facts/99",
            params={"thread_id": "seed-thread"},
        )
        assert resp.status_code == 404
        assert "does not exist" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_fact_succeeds(self, client, runtime) -> None:
        """A valid fact index should delete and return success."""

        await runtime.memory_store.aput(
            ("del-test", "semantic"),
            "fact-to-delete",
            {"evidence_quote": "delete me", "created_at": "2026-01-01T00:00:00Z"},
        )
        resp = await client.delete(
            "/api/memory/facts/1",
            params={"thread_id": "del-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert "0 remaining" in data["detail"]

        # Verify it's actually gone
        count = await runtime.memory_store.arecord_count(("del-test", "semantic"))
        assert count == 0
