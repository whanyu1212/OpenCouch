"""Unit tests for the refactored load_memory_node and finalize_turn_node.

The pre-refactor version of ``run_load_memory_node`` wrote to seven state
keys (transcript, history, working_memory, memory, session_progress,
exercise_state, routing,
response) including a deterministic bootstrap reply that was appended
to the transcript every turn. That behavior was wrong because the node
runs on the spine, not once per session — see the header comment in
``agent/nodes/load_memory.py`` for the full history.

The refactored node only writes ``working_memory``, ``session_memory``,
and ``procedural_profile``.
These tests pin that shape and verify the specific regressions the
refactor fixed:

1. No phantom assistant turns get appended to the transcript.
2. The node does NOT touch routing/response/session_progress/exercise_state.
3. Guest mode still short-circuits and returns an empty working memory.
4. Real retrieval still returns formatted memory snippets.
5. finalize_turn_node appends the assistant response exactly once at
   turn end, guarded against empty responses.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.load_memory_service import (
    SEMANTIC_SEARCH_LIMIT,
    SEMANTIC_WORKING_MEMORY_LIMIT,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import MessageRole
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.load_memory import run_load_memory_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.working_memory import format_working_memory_entry


# ─── Test helpers ──────────────────────────────────────────────────────


class _FakeRuntime:
    """Minimal runtime stand-in that mimics langgraph.runtime.Runtime.

    The real Runtime is a pydantic-backed object with a ``context``
    attribute. These tests accept context overrides and materialize a
    correctly-shaped :class:`WorkflowContext` instance on
    ``runtime.context``.
    """

    def __init__(self, context: dict[str, Any]) -> None:
        defaults: dict[str, Any] = {
            "llm_client": None,
            "memory_store": OpenCouchMemoryStore(),
            "crisis_log_backend": InMemoryCrisisLogBackend(),
            "memory_mode": MemoryMode.LOCAL,
            "embedding_provider": None,
        }
        defaults.update(context)
        self.context = WorkflowContext(**defaults)


def _assert_semantic_entry(entry: dict[str, Any], *, evidence_quote: str) -> None:
    """Assert that ``entry`` is a semantic working-memory dict."""

    assert entry["type"] == "semantic"
    assert entry["evidence_quote"] == evidence_quote


def _assert_episodic_entry(
    entry: dict[str, Any],
    *,
    summary: str,
    primary_themes: list[str] | None = None,
    is_catch_up: bool,
) -> None:
    """Assert that ``entry`` is an episodic working-memory dict."""

    assert entry["type"] == "episodic"
    assert entry["summary"] == summary
    assert entry["primary_themes"] == (primary_themes or [])
    assert entry["is_catch_up"] is is_catch_up


def _make_state(
    *,
    message: str = "I feel anxious",
    session_id: str | None = "thread-test",
    user_id: str | None = None,
    transcript: list[dict[str, str]] | None = None,
    session_memory: dict[str, Any] | None = None,
    procedural_profile: dict[str, Any] | None = None,
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
        "session_memory": session_memory
        or {
            "summary": "",
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
        },
        "procedural_profile": procedural_profile
        or {
            "procedural_rules": [],
            "proactive_recall_enabled": False,
        },
    }
    return state  # type: ignore[return-value]


# ─── load_memory_node tests ─────────────────────────────────────────────


class TestLoadMemoryNode:
    """Regression tests for the refactored load_memory_node."""

    @pytest.mark.asyncio
    async def test_returns_only_working_memory_and_memory_state_keys(
        self,
    ) -> None:
        """The delta must contain only the expected load-memory channels.

        The node now owns ``working_memory``, ``session_memory``,
        ``procedural_profile``, and ``diagnostics`` — no transcript,
        history, response, routing, session_progress, or exercise_state.
        This is the core regression: the old node wrote to seven keys,
        and the response/routing/transcript writes caused the phantom
        assistant turn bug.

        v0.8 observability added ``diagnostics`` to the allowed-key set
        to carry per-stage timings. The forbidden-keys list is unchanged
        because the regression the original test protects against is
        specifically about transcript/routing/response writes, not
        about strict key-count equality.
        """

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state()

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert set(delta.keys()) == {
            "working_memory",
            "session_memory",
            "procedural_profile",
            "diagnostics",
        }
        # Specifically, none of the forbidden keys:
        assert "transcript" not in delta
        assert "history" not in delta
        assert "response" not in delta
        assert "routing" not in delta
        assert "session_progress" not in delta
        assert "exercise_state" not in delta

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
        assert (
            delta["session_memory"]["summary"]
            == "Guest session without long-term memory."
        )

    @pytest.mark.asyncio
    async def test_retrieval_hit_produces_structured_semantic_entry(self) -> None:
        """A query hit should return a raw semantic entry in working_memory."""

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
        _assert_semantic_entry(
            delta["working_memory"][0],
            evidence_quote="I have a sister named Sarah",
        )
        assert (
            format_working_memory_entry(delta["working_memory"][0])
            == "Previously noted: I have a sister named Sarah"
        )
        # v0.4 structured summary: hits / store size / query token count
        # for BOTH semantic and episodic namespaces. Store has 1 semantic
        # record and 0 episodic, query "Sarah" has 1 meaningful token.
        # v0.7 Stage C: summary extended with procedural rule count and
        # recall toggle state. v0.8.1: summary extended with retrieval_path.
        # Substring assertions are more robust to future extensions than
        # pinning the full string.
        summary = delta["session_memory"]["summary"]
        assert "Retrieved 1 of 1 semantic record(s)" in summary
        assert "0 of 0 episodic record(s)" in summary
        assert "0 procedural rule(s)" in summary
        assert "recall=off" in summary
        assert "1 meaningful token(s)" in summary
        assert "path=token_recall" in summary

    @pytest.mark.asyncio
    async def test_superseded_semantic_fact_is_not_loaded(self) -> None:
        """Superseded semantic facts should not resurface in working memory."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("thread-test", "semantic"),
            "fact-old",
            {
                "evidence_quote": "My sister moved out last month.",
                "dormant_at": "2026-04-19T12:00:00Z",
                "superseded_by": "fact-new",
                "user_visible": True,
            },
        )
        await store.aput(
            ("thread-test", "semantic"),
            "fact-new",
            {
                "evidence_quote": "Actually, my sister moved back in this week.",
                "user_visible": True,
            },
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="sister moved")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1
        _assert_semantic_entry(
            delta["working_memory"][0],
            evidence_quote="Actually, my sister moved back in this week.",
        )
        assert (
            "Retrieved 1 of 2 semantic record(s)" in delta["session_memory"]["summary"]
        )

    @pytest.mark.asyncio
    async def test_inactive_semantic_records_do_not_crowd_out_active_hits(
        self,
    ) -> None:
        """Active semantic hits should survive even when stale records dominate the raw rank window."""

        store = OpenCouchMemoryStore()
        for index in range(SEMANTIC_SEARCH_LIMIT):
            await store.aput(
                ("thread-test", "semantic"),
                f"fact-inactive-{index}",
                {
                    "evidence_quote": f"My sister moved inactive entry {index}",
                    "user_visible": True,
                    "dormant_at": "2026-04-20T12:00:00Z",
                    "superseded_by": f"fact-active-{index}",
                },
            )
        for index in range(SEMANTIC_WORKING_MEMORY_LIMIT):
            await store.aput(
                ("thread-test", "semantic"),
                f"fact-active-{index}",
                {
                    "evidence_quote": f"My sister moved active entry {index}",
                    "user_visible": True,
                },
            )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="sister moved")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert [entry["evidence_quote"] for entry in delta["working_memory"]] == [
            f"My sister moved active entry {index}"
            for index in range(SEMANTIC_WORKING_MEMORY_LIMIT)
        ]
        assert (
            f"Retrieved {SEMANTIC_WORKING_MEMORY_LIMIT} of "
            f"{SEMANTIC_SEARCH_LIMIT + SEMANTIC_WORKING_MEMORY_LIMIT} semantic record(s)"
        ) in delta["session_memory"]["summary"]

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
        # Empty stores (0 semantic, 0 episodic, 0 procedural), 3 meaningful
        # query tokens, 0 hits on any namespace. v0.7 Stage C: procedural
        # rule count and recall toggle included in the summary. v0.8.1:
        # retrieval_path is also part of the summary; we assert the
        # load-bearing substrings rather than pinning the full string so
        # future observability additions don't require updating this test.
        summary = delta["session_memory"]["summary"]
        assert "Retrieved 0 of 0 semantic record(s)" in summary
        assert "0 of 0 episodic record(s)" in summary
        assert "0 procedural rule(s)" in summary
        assert "recall=off" in summary
        assert "3 meaningful token(s)" in summary
        # Empty-store short-circuit: when both semantic and episodic
        # stores are empty, the embedding call is skipped entirely.
        assert "path=skipped_empty_store" in summary
        # Specifically NOT the guest session string:
        assert "Guest session" not in summary

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
        assert "0 of 2 semantic record(s)" in delta["session_memory"]["summary"]

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
        assert "query had 3 meaningful token(s)" in delta["session_memory"]["summary"]

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
        assert "0 of 1 semantic record(s)" in delta["session_memory"]["summary"]
        assert "query had 0 meaningful token(s)" in delta["session_memory"]["summary"]

    @pytest.mark.asyncio
    async def test_preserves_other_session_memory_fields_via_spread(self) -> None:
        """When updating ``session_memory.summary``, the node preserves peers.

        LangGraph's default reducer replaces whole dict values, so the
        node must spread the existing ``session_memory`` dict before
        overwriting ``summary``.
        """

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            session_memory={
                "summary": "old summary",
                "active_concerns": ["work stress"],
                "open_loops": ["unresolved grief"],
                "current_goal": "sleep better",
            }
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["session_memory"]["active_concerns"] == ["work stress"]
        assert delta["session_memory"]["open_loops"] == ["unresolved grief"]
        assert delta["session_memory"]["current_goal"] == "sleep better"
        # summary IS updated to the new structured format:
        assert delta["session_memory"]["summary"].startswith("Retrieved")

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


# ─── v0.7 Stage C procedural retrieval tests ───────────────────────────
#
# load_memory_node now also loads the user's procedural profile (rules
# + recall toggle) and attaches it to ``state["procedural_profile"]``.
# These fields are STRUCTURALLY SEPARATE from working_memory because
# procedural rules
# are directives (silent style shaping) rather than content to be
# referenced — see the "Design call" notes in the Stage C plan.
#
# Coverage:
# 1. Guest mode returns empty procedural state across the board
# 2. Local mode with an empty profile returns empty rules + recall=False
# 3. Local mode with a populated profile returns the full rule list
# 4. Recall toggle propagates from profile to state
# 5. Unrelated memory fields (summary, active_concerns, open_loops)
#    are preserved across the delta merge
# 6. Summary string format includes the procedural count


class TestProceduralRetrieval:
    """Tests for the Stage C procedural read path in load_memory_node."""

    @pytest.mark.asyncio
    async def test_guest_mode_returns_empty_procedural_state(self) -> None:
        """Incognito mode must return empty procedural state.

        Privacy contract: guest sessions never read from persistent
        memory, so the procedural rules list must be empty and the
        recall toggle must be False regardless of what might be
        stored for the same owner_id in another context.
        """

        store = OpenCouchMemoryStore()
        # Plant a rule in the store under the eval-user namespace — the
        # guest-mode read must NOT return it.
        from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule

        rule = build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        )
        await aadd_procedural_rule(store, user_id="thread-test", rule=rule)

        runtime = _FakeRuntime(
            {"memory_store": store, "memory_mode": MemoryMode.INCOGNITO}
        )
        state = _make_state()

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        procedural_profile = delta["procedural_profile"]
        assert procedural_profile["procedural_rules"] == []
        assert procedural_profile["proactive_recall_enabled"] is False
        # Verify the "guest session" summary is still produced
        assert "Guest session" in delta["session_memory"]["summary"]

    @pytest.mark.asyncio
    async def test_empty_profile_returns_empty_rules_and_recall_off(
        self,
    ) -> None:
        """A user with no procedural record yet gets empty defaults.

        The empty-default-on-miss behavior comes from Stage A's
        ``aget_procedural_profile`` helper. This test pins that the
        load_memory node propagates those defaults to state without
        creating a stray record in the store.
        """

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="hello")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        procedural_profile = delta["procedural_profile"]
        assert procedural_profile["procedural_rules"] == []
        assert procedural_profile["proactive_recall_enabled"] is False

        # Verify the empty-default read did NOT persist a profile — the
        # store should still have zero records.
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_populated_profile_returns_all_rules(self) -> None:
        """A user with rules gets the full rule list in state.

        Pin: load is NOT query-based. All rules are returned on every
        turn because rules are directives that apply unconditionally.
        The test plants two rules and verifies both are in the delta
        regardless of the current user message.
        """

        from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule

        store = OpenCouchMemoryStore()
        first = build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        )
        second = build_procedural_rule(
            rule_text=("You've said meditation makes you more anxious."),
            evidence=["Please don't suggest meditation again"],
        )
        await aadd_procedural_rule(store, user_id="thread-test", rule=first)
        await aadd_procedural_rule(store, user_id="thread-test", rule=second)

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # Query text is unrelated to either rule — load should still
        # return both (non-query-based semantics).
        state = _make_state(message="tell me about the weather today")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        procedural_profile = delta["procedural_profile"]
        assert procedural_profile["procedural_rules"] == [
            "You prefer shorter responses.",
            "You've said meditation makes you more anxious.",
        ]

    @pytest.mark.asyncio
    async def test_recall_toggle_propagates_from_profile_to_state(self) -> None:
        """The ``proactive_recall_enabled`` toggle is read from the profile
        and attached to state without modification.

        This is how Stage D's prompt builders will know whether to
        emit the "do not proactively reference past sessions"
        constraint. The load path is the only bridge between the
        stored toggle and the prompt layer.
        """

        from agent.memory.procedural import aset_proactive_recall

        store = OpenCouchMemoryStore()
        await aset_proactive_recall(store, user_id="thread-test", enabled=True)

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state()

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["procedural_profile"]["proactive_recall_enabled"] is True

    @pytest.mark.asyncio
    async def test_delta_preserves_unrelated_session_memory_fields(self) -> None:
        """The procedural delta must not clobber existing session memory.

        ``active_concerns``, ``open_loops``, and ``current_goal`` must
        survive while the procedural fields update in their own channel.
        """

        store = OpenCouchMemoryStore()
        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            session_memory={
                "summary": "prior summary",
                "active_concerns": ["work stress"],
                "open_loops": ["unresolved conflict with partner"],
                "current_goal": "get through this week",
            },
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        session_memory = delta["session_memory"]
        # Preserved:
        assert session_memory["active_concerns"] == ["work stress"]
        assert session_memory["open_loops"] == ["unresolved conflict with partner"]
        assert session_memory["current_goal"] == "get through this week"
        # Overwritten in the session-memory channel:
        assert "prior summary" not in session_memory["summary"]  # new summary
        # Procedural state lives separately and resets to empty defaults.
        assert delta["procedural_profile"]["procedural_rules"] == []
        assert delta["procedural_profile"]["proactive_recall_enabled"] is False

    @pytest.mark.asyncio
    async def test_summary_string_reports_procedural_count_and_recall_state(
        self,
    ) -> None:
        """The retrieval summary string must include procedural count
        and recall toggle state for dogfood observability.

        This is a format test — the exact wording is pinned so future
        changes to the summary shape surface as a visible diff during
        review. The dogfood operator's CLI panel depends on reading
        this string, so drift would be a UX regression.
        """

        from agent.memory.procedural import (
            aadd_procedural_rule,
            build_procedural_rule,
            aset_proactive_recall,
        )

        store = OpenCouchMemoryStore()
        await aadd_procedural_rule(
            store,
            user_id="thread-test",
            rule=build_procedural_rule(
                rule_text="You prefer shorter responses.",
                evidence=["Please keep it short"],
            ),
        )
        await aset_proactive_recall(store, user_id="thread-test", enabled=True)

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="hello")

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        summary = delta["session_memory"]["summary"]
        assert "1 procedural rule(s)" in summary
        assert "recall=on" in summary


# ─── v0.4 episodic retrieval tests ──────────────────────────────────────
#
# Load-memory now queries both the semantic and episodic namespaces.
# These tests cover the four retrieval behaviors:
#
# 1. Catch-up on first turn — the most recent episodic arc is pre-pended
#    to working_memory regardless of query match.
# 2. Query-based episodic retrieval on later turns — an arc must share
#    meaningful tokens with the query to surface, same threshold as
#    semantic retrieval.
# 3. Merged output — semantic and episodic results appear in the same
#    working_memory list with distinct prefixes.
# 4. Summary string — the new format reports both counts separately
#    and the INFO log mirrors the same structure.


def _make_episodic_record_value(
    *,
    session_id: str = "prior-session-1",
    summary: str = "User talked about work anxiety and an upcoming meeting.",
    primary_themes: list[str] | None = None,
    opened: str = "anxious",
    closed: str = "calmer",
) -> dict[str, Any]:
    """Build a dict that matches the serialized StoredSessionArc shape.

    We construct the dict directly rather than going through the pydantic
    model to keep these tests focused on load_memory_node behavior.
    The shape here mirrors what the summarizer actually writes via
    ``stored_arc.model_dump(mode='json')``.
    """

    return {
        "session_id": session_id,
        "started_at": "2026-04-09T12:00:00Z",
        "ended_at": "2026-04-09T12:30:00Z",
        "duration_seconds": 1800,
        "turn_count": 8,
        "primary_themes": primary_themes or ["work stress"],
        "summary": summary,
        "mood_arc": {"opened": opened, "closed": closed},
        "open_loops": [],
        "resolved_threads": [],
        "crisis_level_max": 0,
        "id": "arc-test",
        "owner_id": "thread-test",
        "created_at": "2026-04-09T12:30:00Z",
        "last_referenced_at": "2026-04-09T12:30:00Z",
        "user_visible": True,
    }


def _single_turn_transcript(message: str = "I feel anxious") -> list[dict[str, str]]:
    """Return a single-user-turn transcript — the ``is_first_turn`` trigger."""

    return [{"role": MessageRole.USER.value, "content": message}]


def _multi_turn_transcript() -> list[dict[str, str]]:
    """Return a 4-turn transcript — past the first-turn catch-up window."""

    return [
        {"role": MessageRole.USER.value, "content": "hi"},
        {"role": MessageRole.ASSISTANT.value, "content": "hey, what's up?"},
        {"role": MessageRole.USER.value, "content": "I feel anxious"},
        {"role": MessageRole.ASSISTANT.value, "content": "tell me more"},
    ]


class TestEpisodicRetrieval:
    """Tests for v0.4's episodic branch of load_memory_node."""

    @pytest.mark.asyncio
    async def test_catch_up_fires_on_first_turn_regardless_of_query(
        self,
    ) -> None:
        """On the first turn of a session (transcript has exactly one
        entry — the current user turn), the most recent episodic arc
        should be pre-pended to working_memory even if the query has
        no token overlap with the summary. This is the 'last time we
        talked…' context injection."""

        store = OpenCouchMemoryStore()
        namespace = ("thread-test", "episodic")
        await store.aput(
            namespace,
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about grief and sleep difficulties.",
                primary_themes=["grief", "sleep"],
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # Query has ZERO token overlap with the stored summary
        state = _make_state(
            message="let's talk about something completely different",
            transcript=_single_turn_transcript(
                "let's talk about something completely different"
            ),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1
        entry = delta["working_memory"][0]
        _assert_episodic_entry(
            entry,
            summary="User talked about grief and sleep difficulties.",
            primary_themes=["grief", "sleep"],
            is_catch_up=True,
        )

    @pytest.mark.asyncio
    async def test_catch_up_does_not_fire_on_later_turns(self) -> None:
        """On turn 2+ of a session (transcript has >1 entries because
        finalize_turn_node has appended an assistant reply), the catch-
        up injection should NOT fire. Only query-matched arcs appear."""

        store = OpenCouchMemoryStore()
        namespace = ("thread-test", "episodic")
        await store.aput(
            namespace,
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about grief and sleep.",
                primary_themes=["grief"],
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # Query has ZERO token overlap AND multi-turn transcript → miss
        state = _make_state(
            message="let's talk about something completely different",
            transcript=_multi_turn_transcript(),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["working_memory"] == []
        # And the summary should say 0 of 1 episodic (store has a record
        # but it wasn't retrieved because catch-up didn't fire and the
        # query didn't match)
        assert "0 of 1 episodic" in delta["session_memory"]["summary"]

    @pytest.mark.asyncio
    async def test_query_based_episodic_retrieval_on_later_turns(self) -> None:
        """On turn 2+, a query that shares meaningful tokens with an
        episodic summary should retrieve it via the same token-recall
        scorer used for semantic retrieval."""

        store = OpenCouchMemoryStore()
        namespace = ("thread-test", "episodic")
        await store.aput(
            namespace,
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about work anxiety and an upcoming meeting.",
                primary_themes=["work stress"],
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # Query overlaps the summary on {work, anxiety, meeting} — should hit.
        state = _make_state(
            message="remind me what I said about work anxiety",
            transcript=_multi_turn_transcript(),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["working_memory"]) == 1
        _assert_episodic_entry(
            delta["working_memory"][0],
            summary="User talked about work anxiety and an upcoming meeting.",
            primary_themes=["work stress"],
            is_catch_up=False,
        )

    @pytest.mark.asyncio
    async def test_catch_up_returns_most_recent_arc_when_multiple(self) -> None:
        """When multiple episodic arcs exist, catch-up should return the
        LAST one (insertion order, which matches chronological order
        because the summarizer writes one arc per session at session end)."""

        store = OpenCouchMemoryStore()
        namespace = ("thread-test", "episodic")
        # Write three arcs in chronological order
        await store.aput(
            namespace,
            "arc-1",
            _make_episodic_record_value(
                session_id="session-1",
                summary="Oldest session — talked about work stress.",
                primary_themes=["work"],
            ),
        )
        await store.aput(
            namespace,
            "arc-2",
            _make_episodic_record_value(
                session_id="session-2",
                summary="Middle session — talked about family conflict.",
                primary_themes=["family"],
            ),
        )
        await store.aput(
            namespace,
            "arc-3",
            _make_episodic_record_value(
                session_id="session-3",
                summary="Most recent session — talked about sleep problems.",
                primary_themes=["sleep"],
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            message="hi",
            transcript=_single_turn_transcript("hi"),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # The catch-up entry should be the MOST RECENT arc (arc-3),
        # not the oldest (arc-1). Because "hi" has no token overlap with
        # any summary, catch-up is the only mechanism surfacing a record,
        # so we can check the catch-up target directly.
        assert len(delta["working_memory"]) == 1
        catch_up_entry = delta["working_memory"][0]
        _assert_episodic_entry(
            catch_up_entry,
            summary="Most recent session — talked about sleep problems.",
            primary_themes=["sleep"],
            is_catch_up=True,
        )

    @pytest.mark.asyncio
    async def test_merged_working_memory_puts_episodic_first(self) -> None:
        """When both semantic and episodic retrieval produce results,
        the merged working_memory list should have episodic entries
        first (they frame the context), then semantic entries."""

        store = OpenCouchMemoryStore()
        # One episodic arc
        await store.aput(
            ("thread-test", "episodic"),
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about sister Sarah visiting.",
                primary_themes=["family"],
            ),
        )
        # One semantic fact (same user, same topic — Sarah)
        await store.aput(
            ("thread-test", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            message="tell me about my sister Sarah",
            transcript=_multi_turn_transcript(),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # Both should surface — semantic via token-recall, episodic via
        # query-based retrieval (not catch-up, since this is multi-turn).
        assert len(delta["working_memory"]) == 2
        # Episodic entry first
        _assert_episodic_entry(
            delta["working_memory"][0],
            summary="User talked about sister Sarah visiting.",
            primary_themes=["family"],
            is_catch_up=False,
        )
        # Semantic entry second
        _assert_semantic_entry(
            delta["working_memory"][1],
            evidence_quote="I have a sister named Sarah",
        )

    @pytest.mark.asyncio
    async def test_catch_up_deduplicates_against_query_match(self) -> None:
        """If catch-up returns an arc AND the query-based path would
        return the same arc, it should appear only once in working_memory.
        The dedup compares the rendered string."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("thread-test", "episodic"),
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about work anxiety.",
                primary_themes=["work stress"],
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        # First turn AND query overlaps the summary → both branches
        # would return the same arc.
        state = _make_state(
            message="let me tell you about my work anxiety",
            transcript=_single_turn_transcript("let me tell you about my work anxiety"),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # Only one entry, even though both code paths matched
        assert len(delta["working_memory"]) == 1

    @pytest.mark.asyncio
    async def test_summary_includes_both_namespace_counts(self) -> None:
        """The session-memory summary string must report semantic and episodic
        counts separately. This is the load-bearing dogfood observability
        contract — operators need to tell which layer contributed."""

        store = OpenCouchMemoryStore()
        # 2 semantic facts, 1 episodic arc
        await store.aput(
            ("thread-test", "semantic"),
            "fact-1",
            {"evidence_quote": "I have a sister named Sarah"},
        )
        await store.aput(
            ("thread-test", "semantic"),
            "fact-2",
            {"evidence_quote": "I use meditation for anxiety"},
        )
        await store.aput(
            ("thread-test", "episodic"),
            "arc-1",
            _make_episodic_record_value(
                summary="User talked about work pressure.",
            ),
        )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(message="Sarah", transcript=_multi_turn_transcript())

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        summary_str = delta["session_memory"]["summary"]
        # Semantic count: 2 in store, 1 hit on "Sarah" query
        assert "1 of 2 semantic record(s)" in summary_str
        # Episodic count: 1 in store, 0 hit (query doesn't match "work pressure"
        # and this is not a first turn)
        assert "0 of 1 episodic" in summary_str

    @pytest.mark.asyncio
    async def test_incognito_still_skips_episodic_retrieval(self) -> None:
        """Incognito mode should skip BOTH the semantic and episodic
        paths, matching the 'no memory' contract. Even if episodic
        records exist in the store (e.g., from a prior non-incognito
        session), they must not leak into an incognito session."""

        store = OpenCouchMemoryStore()
        await store.aput(
            ("thread-test", "episodic"),
            "arc-1",
            _make_episodic_record_value(summary="User talked about X"),
        )

        runtime = _FakeRuntime(
            {"memory_store": store, "memory_mode": MemoryMode.INCOGNITO}
        )
        state = _make_state(
            message="hi",
            transcript=_single_turn_transcript("hi"),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        assert delta["working_memory"] == []
        assert (
            delta["session_memory"]["summary"]
            == "Guest session without long-term memory."
        )

    @pytest.mark.asyncio
    async def test_catch_up_returns_newest_arc_beyond_50_sessions(self) -> None:
        """Regression test for the 50-session catch-up bug.

        Before v0.9, the catch-up path used ``asearch(query=None, limit=50)``
        and took ``[-1]``, which returned the 50th-oldest record (not the
        newest) once the user exceeded 50 episodic sessions. The fix uses
        ``store.alatest()`` which does ``ORDER BY insertion_order DESC LIMIT 1``.
        """

        store = OpenCouchMemoryStore()
        namespace = ("thread-test", "episodic")

        # Write 55 episodic arcs — exceeding the old limit=50 ceiling.
        for i in range(55):
            await store.aput(
                namespace,
                f"arc-{i}",
                _make_episodic_record_value(
                    session_id=f"session-{i}",
                    summary=f"Session {i} summary.",
                    primary_themes=[f"topic-{i}"],
                ),
            )

        runtime = _FakeRuntime({"memory_store": store, "memory_mode": MemoryMode.LOCAL})
        state = _make_state(
            message="hi",
            transcript=_single_turn_transcript("hi"),
        )

        delta = await run_load_memory_node(state, runtime)  # type: ignore[arg-type]

        # The catch-up entry MUST be the newest arc (session-54), not the
        # 50th-oldest (session-49) which the old code would have returned.
        assert len(delta["working_memory"]) >= 1
        catch_up = delta["working_memory"][0]
        _assert_episodic_entry(
            catch_up,
            summary="Session 54 summary.",
            primary_themes=["topic-54"],
            is_catch_up=True,
        )


# ─── finalize_turn_node tests ───────────────────────────────────────────


class TestFinalizeTurnNode:
    """Tests for the terminal transcript-append node."""

    @pytest.mark.asyncio
    async def test_appends_assistant_response_to_transcript(
        self,
    ) -> None:
        """The node should return a single-turn transcript delta.

        v0.8 observability: the assistant turn dict also carries a
        ``response_style`` field sourced from top-level state. This state
        has no style set, so the mode resolves to ``None``.

        The transcript is reducer-backed, so finalize_turn_node must emit
        ONLY the assistant turn. Returning the full reconstructed transcript
        would duplicate prior entries when the reducer merges the delta into
        checkpointed state.
        """

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "response_text": "Hello, how can I help?",
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert len(delta["transcript"]) == 1
        assert delta["transcript"][0] == {
            "role": MessageRole.ASSISTANT.value,
            "content": "Hello, how can I help?",
            "response_style": None,
        }

    @pytest.mark.asyncio
    async def test_empty_response_text_returns_empty_delta(self) -> None:
        """If ``response_text`` is empty or missing, the node must return an
        empty delta so the transcript stays clean. This guards against a
        branch short-circuiting without producing a reply, which would
        otherwise inject a blank assistant turn."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [{"role": "user", "content": "Hi"}],
            "response_text": "",
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
            "response_text": "   \n\t  ",
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}

    @pytest.mark.asyncio
    async def test_missing_response_slot_returns_empty_delta(self) -> None:
        """If the response field is entirely absent (defensive case), the
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
        """The node's delta must contain only transcript.
        This keeps it a focused single-responsibility node."""

        state: dict[str, Any] = {
            "transcript": [{"role": "user", "content": "Hi"}],
            "history": [],
            "response_text": "Hello",
            "response_style": "supportive",
            "session_memory": {"summary": "x"},
        }
        runtime = _FakeRuntime({})

        delta = await run_finalize_turn_node(state, runtime)  # type: ignore[arg-type]

        assert set(delta.keys()) == {"transcript"}
