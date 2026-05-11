"""Unit tests for ``agent/nodes/extract_procedural_rules.py``.

Mirrors the structure of ``test_extract_facts.py`` — mock runtime, fake
LLM client that dispatches on structured-output schema name, direct
invocation of the node function with crafted state. Covers:

1. Early-exit contract (no LLM client, incognito mode)
2. Empty-result path (LLM returns ``rules=[]``)
3. Happy-path single rule write
4. Multiple rules in one turn
5. LLM exception falls back to empty delta with no writes
6. Per-draft error isolation (one bad draft doesn't abandon the others)
7. Verification that the ``aadd_procedural_rule`` helper is used
   correctly by re-reading the profile after the node runs

These are shape tests, not LLM-quality tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.models import (
    ProceduralExtractionResult,
    ProceduralRuleDraft,
)
from agent.memory.extraction_service import extract_procedural_rules
from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import (
    aget_procedural_profile,
    aput_procedural_profile,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT


async def run_extract_procedural_rules_node(
    state: AgentState,
    runtime: "_MockRuntime",
) -> dict[str, dict]:
    """Test-only adapter mimicking the old node's delta shape.

    See test_extract_semantic_facts.py for full rationale; same idea —
    extraction is now a runtime-managed service, but the test bodies
    here continue to assert on ``delta["diagnostics"][...]`` shape.
    This adapter delegates to the service while preserving that shape.
    """

    ctx = runtime.context
    outcome = await extract_procedural_rules(
        state,
        llm_client=ctx.llm_client,
        memory_store=ctx.memory_store,
        memory_mode=ctx.memory_mode,
        session_buffer=ctx.session_memory_buffer,
    )
    return {"diagnostics": outcome.as_diagnostics()}


# ─── Test fixtures ────────────────────────────────────────────────────────


def _partial_state(
    *,
    message: str = "Please don't suggest meditation again.",
    user_id: str = "alice",
    session_id: str = "test-session",
    history: list[dict[str, str]] | None = None,
) -> AgentState:
    """Return a minimal ``AgentState`` for the procedural writer tests."""

    state: Any = {
        "message": message,
        "history": history or [],
        "user_id": user_id,
        "session_id": session_id,
        "session_progress": {"turn_count": 1},
    }
    return cast(AgentState, state)


class _FakeProceduralLLM(BaseLLMClient):
    """Fake LLM client that returns a canned ``ProceduralExtractionResult``.

    The fake dispatches extraction, write-policy, and reconciliation
    schemas so the procedural writer can stay LLM-primary in tests.
    """

    def __init__(
        self,
        *,
        result: ProceduralExtractionResult,
        should_raise: bool = False,
        policy_decision: dict[str, Any] | None = None,
        reconciliation_decision: dict[str, Any] | None = None,
        raise_on_schema: set[str] | None = None,
    ) -> None:
        self.result = result
        self.should_raise = should_raise
        self.policy_decision = policy_decision or {
            "action": "commit_now",
            "reason": "test procedural rule is durable",
            "confidence": "high",
        }
        self.reconciliation_decision = reconciliation_decision or {
            "action": "append",
            "replace_indexes": [],
            "reason": "test default appends distinct procedural rules",
            "confidence": "high",
        }
        self.raise_on_schema = raise_on_schema or set()
        self.structured_calls = 0
        self.policy_calls = 0
        self.reconciliation_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.text_calls += 1
        return "fake text"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        if response_schema.__name__ == "ProceduralExtractionResult":
            self.structured_calls += 1
            if self.should_raise:
                raise RuntimeError("simulated procedural writer LLM failure")
            return cast(StructuredResponseT, self.result)

        if response_schema.__name__ == "ProceduralWritePolicyDecision":
            self.policy_calls += 1
            if response_schema.__name__ in self.raise_on_schema:
                raise RuntimeError("simulated procedural policy failure")
            return response_schema(  # type: ignore[call-arg,return-value]
                **self.policy_decision,
            )

        if response_schema.__name__ == "ProceduralReconciliationDecision":
            self.reconciliation_calls += 1
            if response_schema.__name__ in self.raise_on_schema:
                raise RuntimeError("simulated procedural reconciliation failure")
            return response_schema(  # type: ignore[call-arg,return-value]
                **self.reconciliation_decision,
            )

        raise RuntimeError(
            f"_FakeProceduralLLM: unexpected schema {response_schema.__name__}"
        )


class _MockRuntime:
    """Minimal runtime stand-in for procedural writer unit tests."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None,
        memory_store: OpenCouchMemoryStore | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
        session_memory_buffer: SessionMemoryBuffer | None = None,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=memory_store or OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode,
            session_memory_buffer=session_memory_buffer,
        )


# ─── Early-exit contract ───────────────────────────────────────────────────


class TestEarlyExits:
    """The node must no-op on the two privacy/availability early-exit paths."""

    @pytest.mark.asyncio
    async def test_no_llm_client_skips_silently(self) -> None:
        """Without an LLM client, the node returns a diagnostics-only delta."""

        store = OpenCouchMemoryStore()
        runtime = _MockRuntime(llm_client=None, memory_store=store)
        state = _partial_state()

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        # v0.8 observability: the skip paths now return a diagnostics
        # delta so the CLI can distinguish "wasn't run" from "ran and
        # skipped silently" in the stage timings panel. The write
        # count stays at zero and the reason names the early exit.
        assert delta["diagnostics"]["procedural_writes"] == 0
        assert (
            delta["diagnostics"]["extract_procedural_reason"]
            == "skipped: no llm_client"
        )
        # No record was written to the procedural namespace
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_incognito_mode_skips_silently(self) -> None:
        """Incognito mode skips writes even when a rule would otherwise fire.

        Privacy contract: incognito mode must never write to
        long-term memory. The test uses a fake LLM that WOULD return a
        rule if called; the assertion verifies both that no write
        happened AND that the LLM was never called (the early exit
        fires before the structured-output call).
        """

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You've said meditation makes you more anxious.",
                        evidence=["Please don't suggest meditation again"],
                    ),
                ],
                reason="would-be-write",
            ),
        )
        runtime = _MockRuntime(
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.INCOGNITO,
        )
        state = _partial_state()

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["extract_procedural_reason"] == "skipped: incognito"
        assert await store.arecord_count() == 0
        assert fake.structured_calls == 0


# ─── Empty-result path ────────────────────────────────────────────────────


class TestEmptyResult:
    """Most turns produce zero rules; the node must handle that cleanly."""

    @pytest.mark.asyncio
    async def test_empty_rules_no_writes(self) -> None:
        """LLM returns empty rules → node emits diagnostics, nothing persisted."""

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[],
                reason="small talk, no style preference stated",
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="thanks, that helps")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        # v0.8 observability: the empty-rules path flows the LLM's
        # reason through the diagnostics so dashboards can see why a
        # turn produced no rules (prompt drift detection).
        assert delta["diagnostics"]["procedural_writes"] == 0
        assert (
            delta["diagnostics"]["extract_procedural_reason"]
            == "small talk, no style preference stated"
        )
        assert fake.structured_calls == 1
        assert await store.arecord_count() == 0

        # Verify the procedural profile is still "empty by default"
        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.rules == []


# ─── Happy-path single rule ────────────────────────────────────────────────


class TestSingleRuleWrite:
    """The common happy path: one rule per turn is the expected shape."""

    @pytest.mark.asyncio
    async def test_single_rule_write_end_to_end(self) -> None:
        """A single rule draft from the LLM gets persisted to the profile."""

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule=("You've said meditation makes you more anxious."),
                        evidence=[
                            "Please don't suggest meditation again — it "
                            "makes me more anxious."
                        ],
                        confidence="high",
                    ),
                ],
                reason="user asked to stop being offered meditation",
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(
            message=(
                "Please don't suggest meditation again — it makes me more anxious."
            ),
        )

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        # v0.8 observability: the happy path reports write count
        # and reason in the diagnostics delta. The actual rule is a
        # store side effect verified below.
        assert delta["diagnostics"]["procedural_writes"] == 1
        assert (
            delta["diagnostics"]["extract_procedural_reason"]
            == "user asked to stop being offered meditation"
        )
        assert fake.structured_calls == 1

        # Profile contract: one rule persisted with the right shape
        profile = await aget_procedural_profile(store, user_id="alice")
        assert len(profile.rules) == 1

        rule = profile.rules[0]
        assert rule.rule == "You've said meditation makes you more anxious."
        assert rule.evidence == [
            "Please don't suggest meditation again — it makes me more anxious.",
        ]
        assert rule.confidence == "high"
        assert rule.source == "explicit_user"
        assert rule.write_timing == "immediate"
        assert rule.write_reason == "llm_policy: test procedural rule is durable"
        assert rule.added_at.endswith("Z")  # ISO-8601 UTC with Z suffix

    @pytest.mark.asyncio
    async def test_rule_write_uses_user_id_as_owner(self) -> None:
        """The rule is namespaced by user_id, not session_id.

        Regression guard: owner_id resolution prefers user_id over
        session_id. A rule written during one session must be
        retrievable from a different session of the same user.
        """

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Keep it short please"],
                    ),
                ],
                reason="user requested shorter replies",
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(
            message="Keep it short please",
            user_id="bob",
            session_id="session-1",
        )

        await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        # The rule should be in bob's namespace, not session-1's
        bob_profile = await aget_procedural_profile(store, user_id="bob")
        assert len(bob_profile.rules) == 1

        # And NOT in session-1's namespace (which would be the case
        # if the node used session_id as the owner)
        session_profile = await aget_procedural_profile(store, user_id="session-1")
        assert session_profile.rules == []

    @pytest.mark.asyncio
    async def test_conflicting_rule_replaces_stale_existing_rule(self) -> None:
        """A new explicit correction should replace the stale procedural rule."""

        store = OpenCouchMemoryStore()
        profile = await aget_procedural_profile(store, user_id="alice")
        profile.rules.append(
            build_procedural_rule(
                rule_text="Suggest meditation when it seems useful.",
                evidence=["Meditation is okay."],
            )
        )
        await aput_procedural_profile(store, user_id="alice", profile=profile)

        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You've said meditation makes you more anxious.",
                        evidence=["Please don't suggest meditation again."],
                        confidence="high",
                    ),
                ],
                reason="user corrected the meditation preference",
            ),
            reconciliation_decision={
                "action": "replace",
                "replace_indexes": [0],
                "reason": "new rule conflicts with older meditation guidance",
                "confidence": "high",
            },
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="Please don't suggest meditation again.")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 1
        assert fake.reconciliation_calls == 1
        profile = await aget_procedural_profile(store, user_id="alice")
        assert len(profile.rules) == 1
        assert profile.rules[0].rule == "You've said meditation makes you more anxious."
        assert len(profile.archived_rules) == 1
        assert (
            profile.archived_rules[0].rule == "Suggest meditation when it seems useful."
        )
        assert profile.archived_rules[0].superseded_by == profile.rules[0].id

    @pytest.mark.asyncio
    async def test_reconciliation_failure_skips_candidate_without_fallback_write(
        self,
    ) -> None:
        """Reconciliation LLM failures should skip conflicting writes."""

        store = OpenCouchMemoryStore()
        profile = await aget_procedural_profile(store, user_id="alice")
        profile.rules.append(
            build_procedural_rule(
                rule_text="Suggest meditation when it seems useful.",
                evidence=["Meditation is okay."],
            )
        )
        await aput_procedural_profile(store, user_id="alice", profile=profile)

        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You've said meditation makes you more anxious.",
                        evidence=["Please don't suggest meditation again."],
                        confidence="high",
                    ),
                ],
                reason="user corrected the meditation preference",
            ),
            raise_on_schema={"ProceduralReconciliationDecision"},
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="Please don't suggest meditation again.")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["procedural_write_skips"] == 1
        assert fake.reconciliation_calls == 1
        profile = await aget_procedural_profile(store, user_id="alice")
        assert len(profile.rules) == 1
        assert profile.rules[0].rule == "Suggest meditation when it seems useful."
        assert profile.archived_rules == []


# ─── Multiple rules in one turn ───────────────────────────────────────────


class TestMultipleRules:
    """Two rules in one turn is rare but possible; both should land."""

    @pytest.mark.asyncio
    async def test_two_rules_both_persisted(self) -> None:
        """A turn with two rule drafts writes both to the profile.

        The user might combine two style requests in one message:
        'Please keep responses shorter and stop asking so many
        clarifying questions.' The writer can return both.
        """

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Please keep responses shorter"],
                    ),
                    ProceduralRuleDraft(
                        rule="You've asked for fewer clarifying questions.",
                        evidence=["stop asking so many clarifying questions"],
                    ),
                ],
                reason=(
                    "user requested shorter replies and fewer clarifying questions"
                ),
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(
            message=(
                "Please keep responses shorter and stop asking so many "
                "clarifying questions."
            ),
        )

        await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        profile = await aget_procedural_profile(store, user_id="alice")
        assert len(profile.rules) == 2
        assert profile.rules[0].rule == "You prefer shorter responses."
        assert profile.rules[1].rule == ("You've asked for fewer clarifying questions.")


# ─── Failure modes ────────────────────────────────────────────────────────


class TestFailureModes:
    """LLM and store errors must degrade silently without propagating."""

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_empty_delta(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An LLM exception logs a warning and returns {}."""

        import logging

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(rules=[], reason="unused"),
            should_raise=True,
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        with caplog.at_level(
            logging.WARNING,
            logger="agent.memory.extraction_service",
        ):
            delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["extract_procedural_reason"] == "skipped: llm error"
        assert await store.arecord_count() == 0
        assert any(
            "LLM structured-output call failed" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_append_never_raises(self) -> None:
        """Node contract: always returns a dict, never propagates.

        Uses a valid LLM result and a valid store. The point of this
        test is simply to verify the happy path's return contract
        (returns a dict, never None, never a propagated exception).
        The failure-isolation test above already covers the
        exception-swallowing behavior.

        v0.8 observability: the node now returns a diagnostics dict
        with at least the timing + write count + reason keys. The
        test still asserts the "always a dict" contract, just with
        a non-empty expected shape.
        """

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Keep it short"],
                    ),
                ],
                reason="user requested shorter replies",
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert isinstance(delta, dict)
        assert "diagnostics" in delta
        assert delta["diagnostics"]["procedural_writes"] == 1

    @pytest.mark.asyncio
    async def test_policy_failure_skips_candidate_without_fallback_write(self) -> None:
        """Write-policy LLM failures should not become local fallback writes."""

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Keep it short"],
                    ),
                ],
                reason="user requested shorter replies",
            ),
            raise_on_schema={"ProceduralWritePolicyDecision"},
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["procedural_policy_errors"] == 1
        assert (
            delta["diagnostics"]["extract_procedural_reason"]
            == "skipped: procedural policy error"
        )
        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.rules == []

    @pytest.mark.asyncio
    async def test_implicit_preference_is_held_not_written(self) -> None:
        """Implicit procedural preferences should not write immediately."""

        store = OpenCouchMemoryStore()
        session_buffer = SessionMemoryBuffer(session_id="test-session")
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You've said meditation makes you more anxious.",
                        evidence=["Meditation makes me more anxious."],
                    ),
                ],
                reason="implicit dislike of meditation",
            ),
            policy_decision={
                "action": "commit_at_session_end",
                "reason": "implicit preference needs repeated evidence",
                "confidence": "high",
            },
        )
        runtime = _MockRuntime(
            llm_client=fake,
            memory_store=store,
            session_memory_buffer=session_buffer,
        )
        state = _partial_state(message="Meditation makes me more anxious.")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["procedural_candidates"] == 1
        assert delta["diagnostics"]["procedural_session_end_holds"] == 1
        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.rules == []
        assert len(session_buffer.held_procedural_candidates) == 1

    @pytest.mark.asyncio
    async def test_turn_scoped_request_is_dropped(self) -> None:
        """Turn-scoped requests should not become procedural memory."""

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["For this reply, keep it short."],
                    ),
                ],
                reason="turn-scoped shorter reply request",
            ),
            policy_decision={
                "action": "drop",
                "reason": "turn-scoped request should not become durable memory",
                "confidence": "high",
            },
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="For this reply, keep it short.")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["procedural_candidates"] == 1
        assert delta["diagnostics"]["procedural_policy_drops"] == 1
        assert fake.policy_calls == 1
        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.rules == []

    @pytest.mark.asyncio
    async def test_safety_conflicting_request_is_clamped_after_policy(self) -> None:
        """Safety-conflicting requests should be dropped after policy review."""

        store = OpenCouchMemoryStore()
        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="Do not ask the user if they are safe.",
                        evidence=["Don't ask if I'm safe."],
                    ),
                ],
                reason="unsafe durable preference request",
            ),
            policy_decision={
                "action": "commit_now",
                "reason": "incorrectly accepted unsafe preference",
                "confidence": "high",
                "safety_conflict": True,
            },
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="Don't ask if I'm safe.")

        delta = await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["procedural_writes"] == 0
        assert delta["diagnostics"]["procedural_candidates"] == 1
        assert delta["diagnostics"]["procedural_policy_drops"] == 1
        assert fake.policy_calls == 1
        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.rules == []


# ─── Preservation of unrelated state ──────────────────────────────────────


class TestStatePreservation:
    """Writing a rule must not clobber ``proactive_recall_enabled`` or
    existing rules."""

    @pytest.mark.asyncio
    async def test_write_preserves_proactive_recall_setting(self) -> None:
        """Writing a new rule preserves a pre-existing recall toggle.

        Regression guard: the load-mutate-put idiom in the Stage A
        helpers must not reset the recall toggle when appending a
        new rule. This test explicitly sets the toggle BEFORE the
        node runs, then verifies it's still set afterward.
        """

        from agent.memory.procedural_profile import aset_proactive_recall

        store = OpenCouchMemoryStore()
        # Pre-set the recall toggle to True
        await aset_proactive_recall(store, user_id="alice", enabled=True)

        fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Keep it short"],
                    ),
                ],
                reason="user requested shorter replies",
            ),
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

        profile = await aget_procedural_profile(store, user_id="alice")
        assert profile.proactive_recall_enabled is True
        assert len(profile.rules) == 1

    @pytest.mark.asyncio
    async def test_write_appends_to_existing_rules(self) -> None:
        """A second rule write appends, never replaces.

        Two separate turns, each producing one rule. The second
        turn's write must preserve the first turn's rule in the
        profile.
        """

        store = OpenCouchMemoryStore()
        # First turn: one rule
        first_fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule="You prefer shorter responses.",
                        evidence=["Keep it short"],
                    ),
                ],
                reason="first turn",
            ),
        )
        runtime = _MockRuntime(llm_client=first_fake, memory_store=store)
        await run_extract_procedural_rules_node(
            _partial_state(message="Keep it short"),
            runtime,  # type: ignore[arg-type]
        )

        # Second turn: different rule
        second_fake = _FakeProceduralLLM(
            result=ProceduralExtractionResult(
                rules=[
                    ProceduralRuleDraft(
                        rule=("You've said meditation makes you more anxious."),
                        evidence=["Please don't suggest meditation again"],
                    ),
                ],
                reason="second turn",
            ),
        )
        runtime = _MockRuntime(llm_client=second_fake, memory_store=store)
        await run_extract_procedural_rules_node(
            _partial_state(
                message="Please don't suggest meditation again",
            ),
            runtime,  # type: ignore[arg-type]
        )

        # Profile should contain BOTH rules
        profile = await aget_procedural_profile(store, user_id="alice")
        assert len(profile.rules) == 2
        rule_texts = [r.rule for r in profile.rules]
        assert "You prefer shorter responses." in rule_texts
        assert "You've said meditation makes you more anxious." in rule_texts
