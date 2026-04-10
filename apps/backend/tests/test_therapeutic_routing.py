"""Tests for therapeutic dispatch, mode nodes, and subgraph wiring.

Covers four concerns:
    1. ``pick_therapeutic_mode`` as a pure function (no LangGraph runtime)
    2. ``run_therapeutic_dispatch_node`` with mocked runtime + fake LLM
    3. ``build_therapeutic_subgraph`` compiles to the expected shape
    4. End-to-end via ``run_agent`` — each mode reaches its terminal
       node with the right routing metadata and a non-empty response

These are unit/integration tests that run in the default pytest suite.
Dataset-driven evals and live-API tests live in Stage G2.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.graph import run_agent
from agent.memory.models import DispatchDecision
from agent.models import AgentInput, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatcher import (
    CLARIFYING_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    pick_therapeutic_mode,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.graph import build_therapeutic_subgraph
from services.llm.base import BaseLLMClient, StructuredResponseT


# ─── Fake LLM client for dispatcher integration tests ────────────────────


class _FakeDispatchLLM(BaseLLMClient):
    """Fake LLM client that returns a canned :class:`DispatchDecision`.

    Used to exercise the dispatcher's LLM path without hitting a real
    provider. Call counts are tracked so tests can assert whether the
    LLM was actually invoked or whether a fast path bypassed it.
    """

    def __init__(
        self,
        *,
        mode: str = "supportive",
        should_raise: bool = False,
    ) -> None:
        self.mode = mode
        self.should_raise = should_raise
        self.structured_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
        use_search: bool = False,
    ) -> str:
        self.text_calls += 1
        return "fake text"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        yield "fake"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        self.structured_calls += 1
        if self.should_raise:
            raise RuntimeError("simulated LLM failure")
        return cast(
            StructuredResponseT,
            DispatchDecision(
                mode=self.mode,  # type: ignore[arg-type]
                reasoning="fake dispatch decision",
                confidence="high",
            ),
        )


class _MockRuntime:
    """Minimal runtime stand-in exposing only ``.context``.

    LangGraph's real ``Runtime`` has many fields (store, stream_writer,
    execution_info, etc.) but the dispatcher only reads ``context``,
    so a plain object suffices for these unit tests.
    """

    def __init__(self, *, llm_client: BaseLLMClient | None = None) -> None:
        self.context: WorkflowContext = {
            "llm_client": llm_client,
            "memory_store": None,  # type: ignore[typeddict-item]
            "crisis_log_backend": None,  # type: ignore[typeddict-item]
            "memory_mode": None,  # type: ignore[typeddict-item]
        }


def _build_state(
    message: str, history: list[dict[str, str]] | None = None
) -> AgentState:
    """Return a minimal ``AgentState`` for dispatcher unit tests."""

    # Typed as Any so we can build a partial state for the dispatcher
    # tests — the dispatcher only reads ``message`` and ``history``,
    # so missing fields don't matter for these tests.
    state: Any = {"message": message, "history": history or []}
    return cast(AgentState, state)


# ─── 1. pick_therapeutic_mode pure-function tests ────────────────────────


class TestPickTherapeuticMode:
    """Unit tests for the regex-only dispatch helper."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            # Supportive — the default for complete self-reports
            ("I feel overwhelmed today.", "supportive"),
            ("I am so tired and lonely", "supportive"),
            ("I feel sad", "supportive"),  # short but self-report
            ("I do not know what I am feeling right now honestly", "supportive"),
            ("My work is really stressful lately.", "supportive"),
            # Reflective — explicit pattern-recognition language
            ("Why do I keep doing this to myself?", "reflective"),
            ("Why does this keep happening", "reflective"),
            ("Why does this always happen to me", "reflective"),
            ("Why does it always happen to me", "reflective"),
            ("This keeps happening every week.", "reflective"),
            ("Every time I see her I feel this way", "reflective"),
            ("I always end up doing the same thing", "reflective"),
            ("Is there a pattern here you see?", "reflective"),
            # Clarifying — truly sparse OR explicit confusion
            ("huh?", "clarifying"),
            ("ok", "clarifying"),
            ("sad", "clarifying"),
            ("Thanks.", "clarifying"),
            ("What do you mean?", "clarifying"),
            ("I don't understand what you said", "clarifying"),
        ],
    )
    def test_pure_regex_dispatch(self, message: str, expected: str) -> None:
        """Regex-only path returns the expected mode for each case."""

        assert pick_therapeutic_mode(message) == expected

    def test_self_report_overrides_short_message_rule(self) -> None:
        """Short messages that ARE self-reports should stay supportive."""

        assert pick_therapeutic_mode("I am sad") == "supportive"
        assert pick_therapeutic_mode("I feel tired") == "supportive"
        assert pick_therapeutic_mode("I'm anxious") == "supportive"

    def test_reflective_beats_short_message_rule(self) -> None:
        """A short pattern-recognition question routes to reflective, not clarifying."""

        assert pick_therapeutic_mode("Why do I keep?") == "reflective"


# ─── 2. run_therapeutic_dispatch_node integration tests ──────────────────


class TestDispatchNode:
    """Integration tests for the dispatch node with mocked runtimes."""

    @pytest.mark.asyncio
    async def test_reflective_fast_path_bypasses_llm(self) -> None:
        """Reflective regex hit should skip the LLM entirely."""

        fake = _FakeDispatchLLM(mode="clarifying")  # wrong on purpose
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Why do I keep doing this to myself?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == REFLECTIVE_NODE
        assert fake.structured_calls == 0  # LLM was not called

    @pytest.mark.asyncio
    async def test_confusion_marker_fast_path_bypasses_llm(self) -> None:
        """Explicit confusion markers should skip the LLM entirely."""

        fake = _FakeDispatchLLM(mode="supportive")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("huh?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == CLARIFYING_NODE
        assert fake.structured_calls == 0

    @pytest.mark.asyncio
    async def test_llm_path_routes_to_llm_pick(self) -> None:
        """Ambiguous messages go to the LLM and use its decision."""

        fake = _FakeDispatchLLM(mode="reflective")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "I keep finding myself getting frustrated with my sister for no reason."
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == REFLECTIVE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_no_llm_client_uses_regex_fallback(self) -> None:
        """With no LLM client the dispatcher must use the pure regex path."""

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("I had a rough day at work")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == SUPPORTIVE_NODE

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_regex_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """LLM exceptions should be logged loudly and fall back to regex."""

        fake = _FakeDispatchLLM(should_raise=True)
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("I feel really sad today.")

        with caplog.at_level(logging.WARNING, logger="agent.therapeutic.dispatcher"):
            cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        # Regex fallback for "I feel really sad today." → supportive
        assert cmd.goto == SUPPORTIVE_NODE
        assert fake.structured_calls == 1
        assert any(
            "falling back to regex" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_deferred_mode_normalizes_to_supportive(self) -> None:
        """LLM picks for v0.6+ modes (psychoeducation etc.) should fall back to supportive."""

        fake = _FakeDispatchLLM(mode="psychoeducation")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Why do I get so tired in the afternoon every day?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == SUPPORTIVE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_command_update_is_empty(self) -> None:
        """The dispatcher's Command should have an empty update dict.

        Mode nodes own the routing.mode/mode_source/mode_type fields in
        their own deltas; the dispatcher shouldn't write any of those.
        """

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("I had a rough day at work")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.update == {}


# ─── 3. build_therapeutic_subgraph compile tests ─────────────────────────


class TestSubgraphCompile:
    """Sanity checks on the compiled subgraph's shape."""

    def test_subgraph_compiles_with_expected_nodes(self) -> None:
        """The subgraph should compile and expose all four internal nodes."""

        subgraph = build_therapeutic_subgraph()
        node_names = set(subgraph.nodes.keys())

        expected = {
            "__start__",
            "therapeutic_dispatch_node",
            "supportive_response_node",
            "reflective_response_node",
            "clarifying_response_node",
        }
        assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"


# ─── 4. End-to-end routing via run_agent ─────────────────────────────────


class TestEndToEndRouting:
    """Drive the full compiled parent graph and verify the right mode runs."""

    @pytest.mark.asyncio
    async def test_supportive_happy_path(self) -> None:
        """A normal self-report routes through therapeutic → supportive."""

        result = await run_agent(
            AgentInput(message="I had a really rough day at work today.")
        )

        assert result.response_type == ResponseKind.THERAPEUTIC
        assert result.mode == "supportive"
        assert result.response_text  # non-empty
        # Deterministic fallback response should include a warm opener
        assert (
            "Thank you for sharing" in result.response_text
            or "makes sense" in result.response_text
        )

    @pytest.mark.asyncio
    async def test_reflective_happy_path(self) -> None:
        """A pattern question routes through therapeutic → reflective."""

        result = await run_agent(
            AgentInput(
                message="Why do I keep ending up in the same fights with my sister?"
            )
        )

        assert result.response_type == ResponseKind.THERAPEUTIC
        assert result.mode == "reflective"
        assert result.response_text
        assert "pattern" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_clarifying_happy_path(self) -> None:
        """A confusion marker routes through therapeutic → clarifying."""

        result = await run_agent(AgentInput(message="huh?"))

        assert result.response_type == ResponseKind.THERAPEUTIC
        assert result.mode == "clarifying"
        assert result.response_text
        # Clarifying fallback should end with a question (it asks for context)
        assert "?" in result.response_text

    @pytest.mark.asyncio
    async def test_crisis_still_routes_to_crisis_not_therapeutic(self) -> None:
        """Non-therapeutic regression guard: crisis messages bypass the subgraph."""

        result = await run_agent(
            AgentInput(message="I have pills and I am going to kill myself tonight.")
        )

        assert result.response_type == ResponseKind.CRISIS
        assert result.crisis.level == 3
        assert result.crisis.needs_crisis_response is True
        # The therapeutic subgraph should NOT have set the mode
        assert result.mode != "supportive"
        assert result.mode != "reflective"
        assert result.mode != "clarifying"

    @pytest.mark.asyncio
    async def test_ambiguous_concerning_language_routes_to_therapeutic(
        self,
    ) -> None:
        """Level-1 ambiguous messages (not crisis) should reach the therapeutic branch.

        This is the regression case for the Stage E rewiring: under the
        old topology, non-crisis traffic terminated at END with the
        bootstrap reply. After Stage E, it should produce a real mode
        response.
        """

        result = await run_agent(
            AgentInput(message="I just wish I could disappear for a while.")
        )

        assert result.response_type == ResponseKind.THERAPEUTIC
        assert result.mode in {"supportive", "reflective", "clarifying"}
        assert result.response_text
        # Critically: response text is NOT the bootstrap stub
        assert "Persistent mode is active" not in result.response_text
        assert "Guest mode is active" not in result.response_text
