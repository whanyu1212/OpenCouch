"""Unit tests for the refactored load_memory_node and finalize_turn_node.

The pre-refactor version of ``run_load_memory_node`` wrote to seven state
keys (transcript, history, working_memory, memory, progress, routing,
response) including a deterministic bootstrap reply that was appended
to the transcript every turn. That behavior was wrong because the node
runs on the spine, not once per session — see the header comment in
``agent/nodes/load_memory.py`` for the full history.

The refactored node only writes ``working_memory`` and ``memory.summary``.
These tests pin that shape and verify the specific regressions the
refactor fixed:

1. No phantom assistant turns get appended to the transcript.
2. The node does NOT touch routing/response/progress.
3. Guest mode still short-circuits and returns an empty working memory.
4. Real retrieval still returns formatted memory snippets.
5. finalize_turn_node appends the assistant response exactly once at
   turn end, guarded against empty responses.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import MessageRole
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.load_memory import run_load_memory_node
from agent.state import AgentState


# ─── Test helpers ──────────────────────────────────────────────────────


class _FakeRuntime:
    """Minimal runtime stand-in that mimics langgraph.runtime.Runtime.

    The real Runtime is a pydantic-backed object with a ``context``
    attribute; these tests only need ``runtime.context.get(key)`` and
    ``runtime.context[key]``, which a plain dict-wrapper satisfies.
    """

    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context


def _make_state(
    *,
    message: str = "I feel anxious",
    session_id: str | None = "thread-test",
    user_id: str | None = None,
    transcript: list[dict[str, str]] | None = None,
    memory: dict[str, Any] | None = None,
) -> AgentState:
    """Build a minimal AgentState for unit testing the node helpers.

    AgentState is a TypedDict; the type annotation is an assertion for
    the type checker, not a runtime constructor. A plain dict with the
    keys the node reads is sufficient. Keys the node doesn't touch can
    be omitted.
    """

    state: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
        "transcript": transcript or [],
        "history": transcript or [],
        "working_memory": [],
        "memory": memory
        or {
            "summary": "",
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
        },
    }
    return state  # type: ignore[return-value]


# ─── load_memory_node tests ─────────────────────────────────────────────


class TestLoadMemoryNode:
    """Regression tests for the refactored load_memory_node."""

    @pytest.mark.asyncio
    async def test_returns_only_working_memory_and_memory_summary_keys(
        self,
    ) -> None:
        """The delta must contain exactly ``working_memory`` and ``memory``
        — no transcript, no history, no response, no routing, no progress.
        This is the core regression: the old node wrote to seven keys,
        and the response/routing/transcript writes caused the phantom
        assistant turn bug."""

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state()

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert set(delta.keys()) == {"working_memory", "memory"}
        # Specifically, none of the forbidden keys:
        assert "transcript" not in delta
        assert "history" not in delta
        assert "response" not in delta
        assert "routing" not in delta
        assert "progress" not in delta

    @pytest.mark.asyncio
    async def test_incognito_mode_skips_retrieval_and_returns_empty(self) -> None:
        """Guest mode should return empty working_memory and a matching
        summary without ever touching the store. This is the incognito
        contract: no reads from persistent storage."""

        store = OpenCouchMemoryStore()
        # Seed the store — it should NOT be read in incognito mode.
        await store.aput(
            ("thread-test", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        runtime = _FakeRuntime(
            {"memory_store": store, "memory_mode": MemoryMode.INCOGNITO}
        )
        state = _make_state(message="Sarah")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["working_memory"] == []
        assert delta["memory"]["summary"] == "Guest session without long-term memory."

    @pytest.mark.asyncio
    async def test_retrieval_hit_produces_formatted_snippet(self) -> None:
        """A query that hits a stored fact should return the formatted
        snippet in working_memory and a matching summary."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("thread-test", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="Sarah")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1
        assert (
            delta["working_memory"][0]
            == "Previously noted: I have a sister named Sarah"
        )
        # Structured summary: hits / store size / query meaningful token count.
        # Store has 1 record, query "Sarah" has 1 meaningful token, 1 hit.
        assert (
            delta["memory"]["summary"]
            == "Retrieved 1 of 1 semantic record(s) (query had 1 meaningful token(s))."
        )

    @pytest.mark.asyncio
    async def test_retrieval_miss_returns_empty_with_zero_snippet_summary(
        self,
    ) -> None:
        """A query that hits no stored facts should return an empty
        working_memory and the zero-snippet summary — NOT the incognito
        summary. This pins the distinction between 'no memory because
        incognito' and 'no memory because nothing matched'."""

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="something totally unrelated")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["working_memory"] == []
        # Empty store (0 records), 3 meaningful query tokens, 0 hits.
        assert (
            delta["memory"]["summary"]
            == "Retrieved 0 of 0 semantic record(s) (query had 3 meaningful token(s))."
        )
        # Specifically NOT the guest session string:
        assert "Guest session" not in delta["memory"]["summary"]

    @pytest.mark.asyncio
    async def test_summary_distinguishes_empty_store_from_below_threshold_miss(
        self,
    ) -> None:
        """The summary string must let a dogfood operator tell the difference
        between 'nothing in the store yet' and 'store has records but none
        matched'. This is the core observability win of the v0.3.1 dogfood
        refactor: a zero-hit line used to read identically for both cases."""

        store = OpenCouchMemoryStore()
        # Non-empty store with content that won't match the query.
        await store.aput(
            ("thread-test", "semantic"),
            "fact-work",
            {"evidence_quote": "I worry about work stress"},
        )
        await store.aput(
            ("thread-test", "semantic"),
            "fact-sleep",
            {"evidence_quote": "I cannot sleep at night"},
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # Query has no topical overlap with either stored fact.
        state = _make_state(message="tell me about octopuses")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # 0 hits, BUT store size is 2 — this is the discriminating line.
        assert delta["working_memory"] == []
        assert "0 of 2 semantic record(s)" in delta["memory"]["summary"]

    @pytest.mark.asyncio
    async def test_summary_reports_meaningful_token_count_after_stopword_filter(
        self,
    ) -> None:
        """The meaningful-token count in the summary comes from
        ``tokenize_meaningful`` (stopword filter applied), not from
        ``str.split()``. This matches the scorer's view of the query
        so the CLI panel is consistent with what actually ran."""

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # 11 raw tokens, but after stopword filter: {tired, work, worry}
        # = 3 meaningful tokens. "I", "am", "so", "the", "of", "that",
        # "about" are all stopwords that get dropped; "tired" survives
        # because adjectives are content words, not stopwords.
        state = _make_state(message="I am so tired of the work that I worry about")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # Expect "(query had 3 meaningful token(s))." — the stopword
        # filter dropped the pronouns, copulas, and connectives, but
        # kept the three content words.
        assert "query had 3 meaningful token(s)" in delta["memory"]["summary"]

    @pytest.mark.asyncio
    async def test_summary_reports_zero_meaningful_tokens_for_stopword_only_query(
        self,
    ) -> None:
        """When the user types only stopwords / punctuation, the summary
        should report 0 meaningful tokens — which, combined with 0 hits,
        tells the operator that retrieval couldn't even try."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("thread-test", "semantic"),
            "fact-1",
            {"evidence_quote": "I worry about work"},
        )
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="I am so the")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["working_memory"] == []
        assert "0 of 1 semantic record(s)" in delta["memory"]["summary"]
        assert "query had 0 meaningful token(s)" in delta["memory"]["summary"]

    @pytest.mark.asyncio
    async def test_preserves_other_memory_fields_via_spread(self) -> None:
        """When updating memory.summary, the node must preserve the other
        memory fields (active_concerns, open_loops, current_goal). LangGraph's
        default reducer replaces whole dict values, so the node must spread
        the existing memory dict before overwriting summary."""

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            memory={
                "summary": "old summary",
                "active_concerns": ["work stress"],
                "open_loops": ["unresolved grief"],
                "current_goal": "sleep better",
            }
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["memory"]["active_concerns"] == ["work stress"]
        assert delta["memory"]["open_loops"] == ["unresolved grief"]
        assert delta["memory"]["current_goal"] == "sleep better"
        # summary IS updated to the new structured format:
        assert delta["memory"]["summary"].startswith("Retrieved")

    @pytest.mark.asyncio
    async def test_owner_id_falls_back_to_session_id(self) -> None:
        """When user_id is None, owner_id derives from session_id so
        retrieval reads from the right namespace."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("session-abc", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="Sarah", session_id="session-abc", user_id=None)

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1

    @pytest.mark.asyncio
    async def test_user_id_takes_precedence_over_session_id(self) -> None:
        """When both user_id and session_id are present, user_id wins.
        This is important for cross-session memory: the store is keyed
        to the user, not the ephemeral thread."""

        store = OpenCouchMemoryStore()
        # Fact stored under user_id namespace:
        await store.aput(
            ("user-42", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            message="Sarah",
            session_id="session-xyz",
            user_id="user-42",
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1


# ─── finalize_turn_node tests ───────────────────────────────────────────


class TestFinalizeTurnNode:
    """Tests for the terminal transcript-append node."""

    @pytest.mark.asyncio
    async def test_appends_assistant_response_to_transcript_and_history(
        self,
    ) -> None:
        """The node should append a single assistant turn containing the
        response text to both transcript and history."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [{"role": "user", "content": "Hi"}],
            "response": {"text": "Hello, how can I help?"},
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["transcript"]) == 2
        assert delta["transcript"][0] == {"role": "user", "content": "Hi"}
        assert delta["transcript"][1] == {
            "role": MessageRole.ASSISTANT.value,
            "content": "Hello, how can I help?",
        }
        assert delta["history"] == delta["transcript"]

    @pytest.mark.asyncio
    async def test_empty_response_text_returns_empty_delta(self) -> None:
        """If response.text is empty or missing, the node must return an
        empty delta so the transcript stays clean. This guards against a
        branch short-circuiting without producing a reply, which would
        otherwise inject a blank assistant turn."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [{"role": "user", "content": "Hi"}],
            "response": {"text": ""},
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_empty_delta(self) -> None:
        """Whitespace-only responses should be treated as empty and not
        pollute the transcript."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [],
            "response": {"text": "   \n\t  "},
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}

    @pytest.mark.asyncio
    async def test_missing_response_slot_returns_empty_delta(self) -> None:
        """If the response dict is entirely absent (defensive case), the
        node should return empty rather than crash."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [],
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}

    @pytest.mark.asyncio
    async def test_does_not_touch_other_state_keys(self) -> None:
        """The node's delta must contain only transcript and history.
        This keeps it a focused single-responsibility node."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [],
            "response": {"text": "Hello"},
            "routing": {"mode": "supportive"},
            "memory": {"summary": "x"},
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert set(delta.keys()) == {"transcript", "history"}
