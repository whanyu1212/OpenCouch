"""Tests for the FastAPI API layer.

Uses FastAPI's TestClient with a test-scoped lifespan that wires an
in-memory runtime (no SQLite on disk, no API keys needed). Each test
gets a clean runtime so there's no state bleed between tests.

The tests focus on the HTTP contract (status codes, response shapes,
error handling) rather than the agent's therapeutic quality. The API layer is a
thin wrapper; these tests verify the wrapping is correct.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from agent.memory.models import (
    DispatchDecision,
    ExtractionResult,
    ProceduralExtractionResult,
    ProceduralProfile,
    ProceduralRule,
    SummarizationResult,
)
from agent.memory.procedural_profile import (
    aget_procedural_profile,
    aput_procedural_profile,
)
from agent.persistence import PersistentAgentRuntime
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeResponseTierLLM(BaseLLMClient):
    """Minimal text-only fake used to prove response-tier routing."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return self.text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield self.text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        raise AssertionError("structured generation should not be used in this test")


class _FakeAPILLM(BaseLLMClient):
    """LLM-shaped test double for API contract tests."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "api fake reply"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "api fake reply"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__

        if schema_name == "CrisisAssessmentSchema":
            from agent.gates.safety.service import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe API contract test turn",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

        if schema_name == "DispatchDecision":
            return cast(
                StructuredResponseT,
                DispatchDecision(
                    response_style="supportive",
                    therapeutic_approach="none",
                    reasoning="ordinary API contract test turn",
                    confidence="high",
                ),
            )

        if schema_name == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                reasoning="ordinary API contract test turn",
                confidence="high",
            )

        if schema_name == "ExtractionResult":
            return cast(
                StructuredResponseT,
                ExtractionResult(facts=[], reason="no API contract test facts"),
            )

        if schema_name == "ProceduralExtractionResult":
            return cast(
                StructuredResponseT,
                ProceduralExtractionResult(
                    rules=[],
                    reason="no API contract test rules",
                ),
            )

        if schema_name == "SummarizationResult":
            return cast(
                StructuredResponseT,
                SummarizationResult(arc=None, reason="API contract test session"),
            )

        raise RuntimeError(f"_FakeAPILLM: unexpected schema {schema_name}")


@pytest.fixture
async def runtime():
    """Yield a fresh in-memory runtime for each test."""

    llm = _FakeAPILLM()
    rt = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        default_llm_client=llm,
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

    from api.dependencies import get_llm_client, get_response_llm_clients, get_runtime
    from api.router import api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api")

    # Override dependencies with test instances
    llm = _FakeAPILLM()
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_response_llm_clients] = lambda: {}

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

    @pytest.mark.asyncio
    async def test_chat_can_use_response_tier_client(self, runtime) -> None:
        """Text response tier should affect only the final prose writer."""

        from fastapi import FastAPI

        from api.dependencies import (
            get_llm_client,
            get_response_llm_clients,
            get_runtime,
        )
        from api.router import api_router

        app = FastAPI()
        app.include_router(api_router, prefix="/api")
        llm = _FakeAPILLM()
        app.dependency_overrides[get_runtime] = lambda: runtime
        app.dependency_overrides[get_llm_client] = lambda: llm
        app.dependency_overrides[get_response_llm_clients] = lambda: {
            "quality": _FakeResponseTierLLM("quality-tier reply"),
        }

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            resp = await ac.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "thread_id": "tier-test",
                    "response_model_tier": "quality",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["response_text"] == "quality-tier reply"

    @pytest.mark.asyncio
    async def test_chat_defaults_to_fast_response_tier(self, runtime) -> None:
        """Omitted response tier should use the fast response client."""

        from fastapi import FastAPI

        from api.dependencies import (
            get_llm_client,
            get_response_llm_clients,
            get_runtime,
        )
        from api.router import api_router

        app = FastAPI()
        app.include_router(api_router, prefix="/api")
        llm = _FakeAPILLM()
        app.dependency_overrides[get_runtime] = lambda: runtime
        app.dependency_overrides[get_llm_client] = lambda: llm
        app.dependency_overrides[get_response_llm_clients] = lambda: {
            "fast": _FakeResponseTierLLM("fast-tier reply"),
        }

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            resp = await ac.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "thread_id": "tier-default-test",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["response_text"] == "fast-tier reply"


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
    async def test_session_status_tracks_active_session_lifecycle(self, client) -> None:
        """Session-status should flip on after a turn and off after /end."""

        resp = await client.get("/api/threads/status-thread/session-status")
        assert resp.status_code == 200
        assert resp.json() == {"has_active_session": False}

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "status-thread"},
        )
        resp = await client.get("/api/threads/status-thread/session-status")
        assert resp.status_code == 200
        assert resp.json() == {"has_active_session": True}

        await client.post("/api/threads/status-thread/end")
        resp = await client.get("/api/threads/status-thread/session-status")
        assert resp.status_code == 200
        assert resp.json() == {"has_active_session": False}

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
        assert assistant_msgs[0]["response_style"] is not None

    # ── v0.10 end-session feedback ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_end_session_without_body_writes_no_feedback(
        self, client, runtime
    ) -> None:
        """POST /end with no body should skip the feedback write.

        Backward-compat: clients that never update to POSTing a body
        must keep working. No feedback record should land in the
        store.
        """

        # Seed one turn so there's state to summarize.
        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "end-no-body"},
        )

        resp = await client.post("/api/threads/end-no-body/end")
        assert resp.status_code == 200

        # No feedback record.
        assert await runtime.session_feedback_backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_end_session_with_null_feedback_writes_no_record(
        self, client, runtime
    ) -> None:
        """Explicit ``{"feedback": null}`` is equivalent to no body.

        Forces us to cover the ``body.feedback is not None`` guard in
        the route handler — missing vs. explicitly-null should both
        short-circuit before calling ``record_session_feedback``.
        """

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "end-null-fb"},
        )

        resp = await client.post(
            "/api/threads/end-null-fb/end", json={"feedback": None}
        )
        assert resp.status_code == 200
        assert await runtime.session_feedback_backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_end_session_with_positive_feedback_writes_record(
        self, client, runtime
    ) -> None:
        """POST with ``{"feedback": "positive"}`` should write one
        record with ``source="api_end"`` BEFORE summarization runs.
        Summarization proceeds as usual (HTTP response shape unchanged).
        """

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "end-with-fb"},
        )

        resp = await client.post(
            "/api/threads/end-with-fb/end",
            json={"feedback": "positive"},
        )
        assert resp.status_code == 200
        # The response shape is unchanged — feedback write status is
        # NOT surfaced. Summarization may return None (incognito / no
        # LLM / thin session) which the handler converts to a plain
        # dict; either shape is accepted.
        data = resp.json()
        assert "summary" in data or "themes" in data

        # Exactly one feedback record in the store.
        from agent.memory.hashing import hash_session_id

        records = await runtime.session_feedback_backend.alist_by_session(
            hash_session_id("end-with-fb")
        )
        assert len(records) == 1
        assert records[0].label == "positive"
        assert records[0].source == "api_end"

    @pytest.mark.asyncio
    async def test_end_session_rejects_invalid_feedback_label(
        self, client, runtime
    ) -> None:
        """A feedback value that isn't ``positive``/``negative``/``skip``
        should be rejected at the Pydantic boundary with HTTP 422,
        BEFORE any graph or persistence code runs."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "end-bad-fb"},
        )

        resp = await client.post(
            "/api/threads/end-bad-fb/end",
            json={"feedback": "awesome"},
        )
        assert resp.status_code == 422
        # Nothing written on a 422.
        assert await runtime.session_feedback_backend.arecord_count() == 0


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
        # v0.10: feedback count surfaced alongside crisis_log_count.
        assert "crisis_log_count" in data
        assert "session_feedback_count" in data
        assert data["session_feedback_count"] == 0  # empty on fresh runtime

    @pytest.mark.asyncio
    async def test_memory_status_reflects_recorded_feedback(
        self, client, runtime
    ) -> None:
        """After POST /end with a feedback label, /memory/status should
        report session_feedback_count >= 1."""

        await client.post(
            "/api/chat",
            json={"message": "hello", "thread_id": "fb-status-test"},
        )
        await client.post(
            "/api/threads/fb-status-test/end",
            json={"feedback": "skip"},
        )

        resp = await client.get(
            "/api/memory/status",
            params={"thread_id": "fb-status-test"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_feedback_count"] >= 1

    @pytest.mark.asyncio
    async def test_update_memory_recall_toggles_owner_state(
        self, client, runtime
    ) -> None:
        """PATCH /memory/recall should persist the proactive recall toggle."""

        resp = await client.patch(
            "/api/memory/recall",
            params={"thread_id": "recall-thread", "user_id": "recall-owner"},
            json={"enabled": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["owner_id"] == "recall-owner"
        assert data["proactive_recall_enabled"] is True

        profile = await aget_procedural_profile(
            runtime.memory_store,
            user_id="recall-owner",
        )
        assert profile.proactive_recall_enabled is True

        status = await client.get(
            "/api/memory/status",
            params={"thread_id": "other-thread", "user_id": "recall-owner"},
        )
        assert status.status_code == 200
        assert status.json()["proactive_recall_enabled"] is True

    @pytest.mark.asyncio
    async def test_delete_fact_404_when_empty(self, client) -> None:
        """Deleting a fact when none exist should return 404."""

        resp = await client.delete(
            "/api/memory/facts/1",
            params={"thread_id": "empty-thread"},
        )
        assert resp.status_code == 404
        assert "No active semantic facts" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_memory_status_counts_active_facts_and_rules(
        self, client, runtime
    ) -> None:
        """Status should reflect active semantic facts and active rule count."""

        await runtime.memory_store.aput(
            ("status-owner", "semantic"),
            "fact-active",
            {
                "evidence_quote": "I have a sister named Sarah.",
                "created_at": "2026-01-01T00:00:00Z",
                "user_visible": True,
            },
        )
        await runtime.memory_store.aput(
            ("status-owner", "semantic"),
            "fact-dormant",
            {
                "evidence_quote": "old fact",
                "created_at": "2026-01-01T00:00:00Z",
                "dormant_at": "2026-01-02T00:00:00Z",
                "user_visible": True,
            },
        )
        await aput_procedural_profile(
            runtime.memory_store,
            user_id="status-owner",
            profile=ProceduralProfile(
                rules=[
                    ProceduralRule(
                        rule="Keep your replies shorter.",
                        evidence=["shorter replies"],
                        confidence="high",
                        added_at="2026-01-01T00:00:00Z",
                        source="explicit_user",
                    ),
                    ProceduralRule(
                        rule="Don't suggest meditation.",
                        evidence=["stop suggesting meditation"],
                        confidence="high",
                        added_at="2026-01-01T00:00:00Z",
                        source="explicit_user",
                    ),
                ]
            ),
        )

        resp = await client.get(
            "/api/memory/status",
            params={"thread_id": "status-owner"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["semantic"] == 1
        assert data["counts"]["procedural"] == 2
        assert data["counts"]["episodic"] == 0

    @pytest.mark.asyncio
    async def test_list_facts_hides_inactive_semantic_records(
        self, client, runtime
    ) -> None:
        """The facts endpoint should show only active semantic facts."""

        await runtime.memory_store.aput(
            ("fact-list-owner", "semantic"),
            "fact-active",
            {
                "evidence_quote": "My sister Sarah moved nearby.",
                "category": "relationship",
                "predicate": "KNOWS",
                "subject": {"identifier": "user"},
                "object": {"identifier": "Sarah"},
                "confidence": "high",
                "created_at": "2026-01-01T00:00:00Z",
                "user_visible": True,
            },
        )
        await runtime.memory_store.aput(
            ("fact-list-owner", "semantic"),
            "fact-superseded",
            {
                "evidence_quote": "stale",
                "category": "identity",
                "predicate": "IS",
                "subject": {"identifier": "user"},
                "object": {"identifier": "engineer"},
                "confidence": "medium",
                "created_at": "2026-01-01T00:00:00Z",
                "superseded_by": "fact-active",
                "user_visible": True,
            },
        )

        resp = await client.get(
            "/api/memory/facts",
            params={"thread_id": "fact-list-owner"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["key"] == "fact-active"

    @pytest.mark.asyncio
    async def test_delete_fact_indexes_only_active_semantic_records(
        self, client, runtime
    ) -> None:
        """Deleting fact #1 should target the first active fact, not dormant rows."""

        await runtime.memory_store.aput(
            ("delete-active-owner", "semantic"),
            "fact-dormant",
            {
                "evidence_quote": "old",
                "created_at": "2026-01-01T00:00:00Z",
                "dormant_at": "2026-01-02T00:00:00Z",
                "user_visible": True,
            },
        )
        await runtime.memory_store.aput(
            ("delete-active-owner", "semantic"),
            "fact-active",
            {
                "evidence_quote": "delete me",
                "created_at": "2026-01-01T00:00:00Z",
                "user_visible": True,
            },
        )

        resp = await client.delete(
            "/api/memory/facts/1",
            params={"thread_id": "delete-active-owner"},
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        remaining = await runtime.memory_store.asearch(
            ("delete-active-owner", "semantic"),
            query=None,
            limit=100,
        )
        assert len(remaining) == 1
        assert remaining[0].key == "fact-dormant"

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
