"""Tests for therapeutic dispatch, response-style nodes, and subgraph wiring.

Covers four concerns:
    1. ``pick_therapeutic_response_style`` as a pure function (no LangGraph runtime)
    2. ``run_therapeutic_dispatch_node`` with mocked runtime + fake LLM
    3. ``build_therapeutic_subgraph`` compiles to the expected shape
    4. End-to-end via ``run_agent`` — each response style reaches its terminal
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
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.models import DispatchDecision
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput, CrisisAssessment, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatch import (
    CLARIFYING_NODE,
    CLOSING_NODE,
    EXERCISE_CONSENT_PATTERNS,
    GUIDED_EXERCISE_NODE,
    PSYCHOEDUCATION_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    TECHNIQUE_NODE,
    THERAPEUTIC_RESPONSE_NODE,
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _is_advice_request_without_exercise_consent,
    _matches_any,
    _format_prompt_trigger_phrases,
    _is_bare_ack_to_open_question,
    build_therapeutic_dispatch_system_prompt,
    pick_therapeutic_response_style,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.exercises.types import ExerciseStepDecision
from agent.therapeutic.graph import (
    TherapeuticSubgraphInput,
    TherapeuticSubgraphOutput,
    build_therapeutic_subgraph,
)
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
        response_style: str = "supportive",
        therapeutic_approach: str = "none",
        should_raise: bool = False,
    ) -> None:
        self.response_style = response_style
        self.therapeutic_approach = therapeutic_approach
        self.should_raise = should_raise
        self.structured_calls = 0
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
        self.structured_calls += 1
        if self.should_raise:
            raise RuntimeError("simulated LLM failure")
        return cast(
            StructuredResponseT,
            DispatchDecision(
                response_style=self.response_style,  # type: ignore[arg-type]
                therapeutic_approach=self.therapeutic_approach,  # type: ignore[arg-type]
                reasoning="fake dispatch decision",
                confidence="high",
            ),
        )


class _RecordingTextLLM(BaseLLMClient):
    """Record text-generation prompts while returning canned text."""

    def __init__(self, response_text: str = "recorded") -> None:
        self.prompts: list[str] = []
        self.system_instructions: list[str | None] = []
        self.response_text = response_text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        return self.response_text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        yield self.response_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        raise RuntimeError("structured output not used by text-node prompt tests")


class _FakeStepStateLLM(BaseLLMClient):
    """Fake step classifier that returns a canned exercise-step decision."""

    def __init__(self, step_state: str = "hold") -> None:
        self.step_state = step_state
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake text"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake text"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        self.structured_calls += 1
        return cast(
            StructuredResponseT,
            ExerciseStepDecision(
                step_state=self.step_state,  # type: ignore[arg-type]
                reasoning="fake step-state decision",
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
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


def _build_state(
    message: str, history: list[dict[str, str]] | None = None
) -> AgentState:
    """Return a minimal ``AgentState`` for dispatcher unit tests."""

    # Typed as Any so we can build a partial state for the dispatcher
    # tests — the dispatcher only reads ``message`` and ``history``,
    # so missing fields don't matter for these tests.
    state: Any = {"message": message, "history": history or []}
    return cast(AgentState, state)


# ─── 1. pick_therapeutic_response_style pure-function tests ────────────────────────


class TestPickTherapeuticResponseStyle:
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
            # Reflective — only the narrowest self-referential patterns
            # survive in the regex-only fallback. Broader patterns like
            # "every time I", "is there a pattern" are demoted to LLM.
            ("Why do I keep doing this to myself?", "reflective"),
            ("Why does this keep happening", "reflective"),
            ("Why does this always happen to me", "reflective"),
            ("Why does it always happen to me", "reflective"),
            ("This keeps happening every week.", "reflective"),
            ("I always end up doing the same thing", "reflective"),
            # Demoted to LLM — regex fallback returns supportive:
            ("Every time I see her I feel this way", "supportive"),
            ("Is there a pattern here you see?", "supportive"),
            # Clarifying — only exact-message confusion cues survive
            ("huh?", "clarifying"),
            ("ok", "clarifying"),
            ("sad", "clarifying"),
            ("Thanks.", "clarifying"),
            ("What do you mean?", "clarifying"),
            # Demoted to LLM — regex fallback returns supportive:
            ("I don't understand what you said", "supportive"),
        ],
    )
    def test_pure_regex_dispatch(self, message: str, expected: str) -> None:
        """Regex-only path returns the expected response style for each case."""

        assert pick_therapeutic_response_style(message) == expected

    def test_self_report_overrides_short_message_rule(self) -> None:
        """Short messages that ARE self-reports should stay supportive."""

        assert pick_therapeutic_response_style("I am sad") == "supportive"
        assert pick_therapeutic_response_style("I feel tired") == "supportive"
        assert pick_therapeutic_response_style("I'm anxious") == "supportive"

    def test_reflective_beats_short_message_rule(self) -> None:
        """A short pattern-recognition question routes to reflective, not clarifying."""

        assert pick_therapeutic_response_style("Why do I keep?") == "reflective"


# ─── 2. run_therapeutic_dispatch_node integration tests ──────────────────


class TestDispatchNode:
    """Integration tests for the dispatch node with mocked runtimes."""

    @pytest.mark.asyncio
    async def test_safety_clarification_still_uses_llm_classifier(self) -> None:
        """Level-1 crisis ambiguity should keep the selected response style."""

        fake = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="pfa",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("I don't know how much longer I can do this.")
        state["crisis"] = CrisisAssessment(
            level=1,
            confidence="medium",
            needs_clarification=True,
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == SUPPORTIVE_NODE
        assert cmd.update == {
            "response_style": "supportive",
            "therapeutic_approach": "pfa",
        }
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_primary_classifies_reflective_message(self) -> None:
        """LLM-primary: reflective messages go through the LLM classifier."""

        fake = _FakeDispatchLLM(response_style="reflective")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Why do I keep doing this to myself?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == REFLECTIVE_NODE
        assert fake.structured_calls == 1  # LLM was called

    @pytest.mark.asyncio
    async def test_regex_fallback_without_llm_routes_reflective(self) -> None:
        """Without LLM, regex fallback handles reflective patterns."""

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("Why do I keep doing this to myself?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == REFLECTIVE_NODE

    @pytest.mark.asyncio
    async def test_regex_fallback_without_llm_routes_clarifying(self) -> None:
        """Without LLM, regex fallback handles confusion markers."""

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("huh?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == CLARIFYING_NODE

    @pytest.mark.asyncio
    async def test_llm_path_routes_to_llm_pick(self) -> None:
        """Ambiguous messages go to the LLM and use its decision."""

        fake = _FakeDispatchLLM(response_style="reflective")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "I keep finding myself getting frustrated with my sister for no reason."
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == REFLECTIVE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_bare_ack_to_open_question_bypasses_llm(self) -> None:
        """A bare "yes" after an open question needs clarification."""

        fake = _FakeDispatchLLM(response_style="guided_exercise")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "yes",
            history=[
                {
                    "role": "user",
                    "content": "I don't know if I should bring this up.",
                },
                {"role": "assistant", "content": "What's making you hesitant?"},
            ],
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == CLARIFYING_NODE
        assert fake.structured_calls == 0

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

        with caplog.at_level(
            logging.WARNING, logger="agent.therapeutic.dispatch.router"
        ):
            cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        # Regex fallback for "I feel really sad today." → supportive
        assert cmd.goto == SUPPORTIVE_NODE
        assert fake.structured_calls == 1
        assert any(
            "falling back to regex" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_llm_pick_psychoeducation_routes_to_psychoeducation_node(
        self,
    ) -> None:
        """LLM picks for psychoeducation route to the real psychoeducation node.

        Regression guard for v0.6 Stage A: before Stage A landed, this
        pick normalized to supportive because the psychoeducation node
        didn't exist. After Stage A, the pick must route to the real
        node.
        """

        fake = _FakeDispatchLLM(response_style="psychoeducation")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Why do I get so tired in the afternoon every day?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == PSYCHOEDUCATION_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_pick_closing_routes_to_closing_node(self) -> None:
        """LLM picks for closing route to the real closing node.

        Regression guard for v0.6 Stage B: before Stage B landed, this
        pick normalized to supportive because the closing node didn't
        exist. After Stage B, the pick must route to the real node.
        """

        fake = _FakeDispatchLLM(response_style="closing")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Thanks, I think I need to step away now.")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == CLOSING_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_pick_guided_exercise_routes_to_guided_exercise_node(
        self,
    ) -> None:
        """LLM picks for guided_exercise route to the real guided_exercise node.

        Regression guard for v0.6 Stage C: before Stage C landed, this
        pick normalized to supportive via the deferred-style block. After
        Stage C, the deferred-style block is gone and the pick must
        route to the real node.
        """

        fake = _FakeDispatchLLM(response_style="guided_exercise")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Can you walk me through a grounding exercise?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response_style", "therapeutic_approach", "message", "expected_node"),
        [
            (
                "guided_exercise",
                "none",
                "Can you walk me through a grounding exercise?",
                GUIDED_EXERCISE_NODE,
            ),
            (
                "closing",
                "cbt",
                "I need to go, but what should I remember from this?",
                CLOSING_NODE,
            ),
            (
                "psychoeducation",
                "dbt_skills",
                "Why does my distress spike so fast when conflict starts?",
                PSYCHOEDUCATION_NODE,
            ),
            (
                "technique",
                "dbt_skills",
                "Can you help me work through the exact skill for this moment?",
                TECHNIQUE_NODE,
            ),
            (
                "supportive",
                "pfa",
                "I am really shaken and need a steady response.",
                SUPPORTIVE_NODE,
            ),
        ],
    )
    async def test_response_style_selects_node_and_approach_stays_context(
        self,
        response_style: str,
        therapeutic_approach: str,
        message: str,
        expected_node: str,
    ) -> None:
        """The response style routes; the therapeutic approach only travels as context."""

        fake = _FakeDispatchLLM(
            response_style=response_style,
            therapeutic_approach=therapeutic_approach,
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(message)

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == expected_node
        assert cmd.update == {
            "response_style": response_style,
            "therapeutic_approach": therapeutic_approach,
        }
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_guided_exercise_pick_for_coping_tips_is_guarded(
        self,
    ) -> None:
        """Coping advice requests should not start an exercise automatically."""

        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("What are some tips to cope at different severity levels?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == PSYCHOEDUCATION_NODE
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_explicit_coping_exercise_request_is_not_guarded(
        self,
    ) -> None:
        """Explicit consent to do something structured still starts an exercise."""

        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "Everything is overwhelming right now. "
            "Can we do something to help me cope with this moment?"
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_active_exercise_fast_path_bypasses_llm(self) -> None:
        """Active-exercise fast path short-circuits to guided_exercise.

        This is the CRITICAL multi-turn test for v0.6 Stage C. When a
        prior turn started an exercise (setting exercise_state.exercise_type
        and exercise_state.exercise_step), the dispatcher must route the
        next turn to guided_exercise. With LLM-primary dispatch, the
        LLM sees the active exercise context in the prompt and picks
        guided_exercise to continue. The dispatcher preserves the
        existing therapeutic approach from the entry turn.
        """

        fake = _FakeDispatchLLM(response_style="guided_exercise")
        runtime = _MockRuntime(llm_client=fake)

        state: Any = {
            "message": "I see a lamp, a book, a plant, my coffee, and the window.",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.goto == GUIDED_EXERCISE_NODE
        assert fake.structured_calls == 1  # LLM was called

    @pytest.mark.asyncio
    async def test_active_exercise_stop_request_bypasses_llm(self) -> None:
        """An explicit mid-exercise stop request exits to supportive style."""

        fake = _FakeDispatchLLM(response_style="closing")
        runtime = _MockRuntime(llm_client=fake)

        state: Any = {
            "message": "Actually can we stop? I don't want to do this right now.",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.goto == SUPPORTIVE_NODE
        assert fake.structured_calls == 0
        update = cast(dict[str, Any], cmd.update)
        assert update["exercise_state"]["exercise_type"] is None
        assert update["exercise_state"]["exercise_step"] is None

    @pytest.mark.asyncio
    async def test_no_active_exercise_fast_path_when_fields_none(self) -> None:
        """If exercise_type/step are None, the fast path does NOT fire.

        Regression guard for the fast-path condition. A state with
        exercise_type=None must NOT trigger the active-exercise fast
        path — otherwise an exercise that was cleared would still
        appear "active" if the None was spelled wrong.
        """

        fake = _FakeDispatchLLM(response_style="supportive")
        runtime = _MockRuntime(llm_client=fake)

        state: Any = {
            "message": "I had a rough day at work.",
            "history": [],
            "session_progress": {"turn_count": 1},
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # The fast path should NOT fire; the LLM path should run and
        # pick supportive (what the fake returns).
        assert cmd.goto == SUPPORTIVE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_pending_exercise_selection_choice_routes_to_guided(self) -> None:
        """A numbered reply to offered exercise options returns to the node."""

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "2",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_selection_options": [
                    "grounding_box_breathing",
                    "self_compassion_break",
                ],
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.goto == GUIDED_EXERCISE_NODE

    @pytest.mark.asyncio
    async def test_pending_exercise_selection_alias_routes_to_guided(self) -> None:
        """A named reply to offered exercise options returns to the node."""

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "self compassion",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_selection_options": [
                    "grounding_box_breathing",
                    "self_compassion_break",
                ],
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.goto == GUIDED_EXERCISE_NODE

    @pytest.mark.asyncio
    async def test_command_update_contains_therapeutic_approach(self) -> None:
        """The dispatcher's Command carries top-level ``therapeutic_approach``.

        Response-style nodes own response-style metadata in their own deltas; the
        dispatcher only writes ``therapeutic_approach`` so the response-style node can
        load the correct knowledge overlay.
        """

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("I had a rough day at work")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        # Regex fallback for supportive defaults to MI
        assert "therapeutic_approach" in cmd.update
        assert cmd.update["therapeutic_approach"] == "motivational_interviewing"


# ─── 3. build_therapeutic_subgraph compile tests ─────────────────────────


class TestSubgraphCompile:
    """Sanity checks on the compiled subgraph's shape."""

    def test_subgraph_compiles_with_expected_nodes(self) -> None:
        """The subgraph should compile and expose its simplified internal nodes."""

        subgraph = build_therapeutic_subgraph()
        node_names = set(subgraph.nodes.keys())

        expected = {
            "__start__",
            "therapeutic_dispatch_node",
            THERAPEUTIC_RESPONSE_NODE,
            "guided_exercise_response_node",
        }
        assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"

    def test_subgraph_declares_explicit_input_and_output_schemas(self) -> None:
        """The therapeutic subgraph should narrow its LangGraph boundary."""

        subgraph = build_therapeutic_subgraph()
        assert subgraph.builder.input_schema is TherapeuticSubgraphInput
        assert subgraph.builder.output_schema is TherapeuticSubgraphOutput


# ─── 4. End-to-end routing via run_agent ─────────────────────────────────


class TestEndToEndRouting:
    """Drive the full compiled parent graph and verify the right response style runs."""

    @pytest.mark.asyncio
    async def test_supportive_happy_path(self) -> None:
        """A normal self-report routes through therapeutic → supportive."""

        result = await run_agent(
            AgentInput(message="I had a really rough day at work today.")
        )

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.response_style == "supportive"
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

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.response_style == "reflective"
        assert result.response_text
        assert "pattern" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_reflective_node_preserves_clean_reflection_when_llm_omits_question(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reflective node should allow a clean reflection to stand on its own."""

        from agent.therapeutic import response
        from agent.therapeutic.response import run_therapeutic_response_node

        monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

        runtime = _MockRuntime(llm_client=_RecordingTextLLM())
        state: Any = {
            "message": "Why do I keep ending up in the same fight with my partner?",
            "history": [],
            "response_style": "reflective",
        }

        delta = await run_therapeutic_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert delta["response_style"] == "reflective"
        assert delta["response_text"].startswith("recorded")
        assert "?" not in delta["response_text"]

    @pytest.mark.asyncio
    async def test_technique_node_adds_attuned_opening_when_llm_omits_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Technique node should add an attuned opening before structure."""

        from agent.therapeutic import response
        from agent.therapeutic.response import run_therapeutic_response_node

        monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

        runtime = _MockRuntime(
            llm_client=_RecordingTextLLM(
                "Absolutely. Let’s take it one piece at a time.\n\n"
                "What’s the exact thought you want to examine, in your own words?"
            )
        )
        state: Any = {
            "message": (
                "Can you help me examine this thought step by step? "
                "I want to look at evidence for and against it."
            ),
            "history": [],
            "response_style": "technique",
            "therapeutic_approach": "cbt",
        }

        delta = await run_therapeutic_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert delta["response_style"] == "technique"
        assert delta["response_text"].startswith("Let's slow this down for a second.")

    @pytest.mark.asyncio
    async def test_clarifying_happy_path(self) -> None:
        """A confusion marker routes through therapeutic → clarifying."""

        result = await run_agent(AgentInput(message="huh?"))

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.response_style == "clarifying"
        assert result.response_text
        # Clarifying fallback should end with a question (it asks for context)
        assert "?" in result.response_text

    @pytest.mark.asyncio
    async def test_psychoeducation_deterministic_fallback_path(self) -> None:
        """Verify the psychoeducation node's deterministic fallback is reachable.

        The dispatcher's regex-only path can only route to supportive,
        reflective, or clarifying — never to psychoeducation, because no
        regex pattern identifies "confusion about a reaction" with
        acceptable precision. Psychoeducation is only reached via the
        LLM classifier path.

        Since ``run_agent`` with no LLM client will always take the
        regex path, we can't exercise the full end-to-end psychoeducation
        path in a pure unit test without a fake LLM injected via
        ``create_configured_llm_client``. That's test-infrastructure
        work out of Stage A's scope — the dispatcher-level integration
        tests above already verify the routing, and the subgraph
        compile test verifies the node is reachable in principle.

        This test instead exercises the node's fallback contract by
        invoking the node function directly with a no-LLM runtime,
        asserting the delta dict has the right shape.
        """

        from agent.therapeutic.response import run_therapeutic_response_node

        runtime = _MockRuntime(llm_client=None)
        state = _build_state(
            "I don't understand why I always cry when she calls.",
        )
        state["response_style"] = "psychoeducation"

        delta = await run_therapeutic_response_node(state, runtime)  # type: ignore[arg-type]

        assert delta["response_kind"] == ResponseCategory.THERAPEUTIC
        assert delta["response_text"]  # non-empty
        assert delta["response_style"] == "psychoeducation"
        assert delta["response_style_source"] == "therapeutic_dispatch"
        # Deterministic fallback is permission-first by design
        assert "?" in delta["response_text"]

    @pytest.mark.asyncio
    async def test_psychoeducation_node_allows_clean_frame_when_llm_omits_question(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Psychoeducation replies may land on a clean frame without a forced question."""

        from agent.therapeutic import response
        from agent.therapeutic.response import run_therapeutic_response_node

        monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

        runtime = _MockRuntime(
            llm_client=_RecordingTextLLM(
                "That can happen when your body alarm turns on at night. "
                "The quiet can make sensations feel louder."
            )
        )
        state = _build_state(
            "My heart races and my chest gets tight at night. Why does this happen?",
        )
        state["response_style"] = "psychoeducation"

        delta = await run_therapeutic_response_node(state, runtime)  # type: ignore[arg-type]

        assert delta["response_style"] == "psychoeducation"
        assert delta["response_text"].startswith("That can happen")
        assert "?" not in delta["response_text"]

    @pytest.mark.asyncio
    async def test_closing_deterministic_fallback_path(self) -> None:
        """Verify the closing node's deterministic fallback is reachable.

        Same rationale as ``test_psychoeducation_deterministic_fallback_path``:
        the regex dispatcher can't route to closing (no regex for
        wind-down signals with acceptable precision), so end-to-end
        reachability of the closing fallback has to be tested by
        invoking the node function directly.

        Two extra assertions specific to closing:
        - The fallback MUST NOT contain "it was nice talking" or
          "nice to meet you" (the most-common closing-style failure
          response style pinned in the knowledge file).
        - The fallback MUST include an open-door phrasing so the
          user doesn't feel dismissed.
        """

        from agent.therapeutic.response import run_therapeutic_response_node

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("I should probably go, thanks for this.")
        state["response_style"] = "closing"

        delta = await run_therapeutic_response_node(state, runtime)  # type: ignore[arg-type]

        assert delta["response_kind"] == ResponseCategory.THERAPEUTIC
        assert delta["response_text"]  # non-empty
        assert delta["response_style"] == "closing"
        assert delta["response_style_source"] == "therapeutic_dispatch"

        text_lower = delta["response_text"].lower()
        # The single most common closing-style failure mode
        assert "nice talking" not in text_lower
        assert "nice to meet" not in text_lower
        # Open-door signal — some form of "here when you want"
        assert (
            "whenever" in text_lower
            or "here" in text_lower
            or "come back" in text_lower
        )

    @pytest.mark.asyncio
    async def test_guided_exercise_start_writes_exercise_state(self) -> None:
        """Starting a new exercise sets exercise_type and exercise_step.

        Entry condition 1 for the guided_exercise node: no active
        exercise, so the node should start 5-4-3-2-1 grounding at
        step 0. The delta must include exercise_state.exercise_type and
        exercise_state.exercise_step.
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        # State with no active exercise
        state: Any = {
            "message": "Can you walk me through a grounding exercise?",
            "history": [],
            "session_progress": {"turn_count": 1},
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # Response delta is populated with the start-step text
        assert delta["response_kind"] == ResponseCategory.THERAPEUTIC
        assert delta["response_text"]
        assert "five things" in delta["response_text"].lower()
        assert delta["response_style"] == "guided_exercise"
        # Progress delta starts the exercise at step 0
        assert delta["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
        assert delta["exercise_state"]["exercise_step"] == 0

    @pytest.mark.asyncio
    async def test_guided_exercise_advances_on_complete_response(self) -> None:
        """A completion-triggering response advances exercise_step by 1.

        Entry condition 2 for the guided_exercise node: exercise is
        active. The user names enough items to count as complete for
        the current step; the node should advance to step+1.
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        # State with an active exercise on step 0
        state: Any = {
            "message": "I see a lamp, a plant, my coffee, the window, and a book.",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # Exercise advanced to step 1 (4 things you can hear)
        assert delta["exercise_state"]["exercise_step"] == 1
        assert "exercise_type" not in delta["exercise_state"]
        assert "four things" in delta["response_text"].lower()
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_holds_on_tentative_response(self) -> None:
        """A tentative response keeps the step unchanged (hold, not advance).

        The user names fewer items than the step's min_count threshold,
        so the classifier returns HOLD. The exercise_step stays where
        it is and the response offers space.
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "um, a plant?",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # HOLD: exercise_state is NOT in the delta (no state change), OR
        # exercise_state.exercise_step is still 0 if it IS in the delta.
        # The current implementation omits exercise_state on HOLD to signal
        # "no state change," which is the idiomatic LangGraph pattern
        # — but we tolerate either for robustness.
        if "exercise_state" in delta:
            assert delta["exercise_state"]["exercise_step"] == 0
        # Fallback response gives space without advancing
        assert delta["response_text"]
        assert "time" in delta["response_text"].lower()
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_hold_prompt_reanchors_same_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hold-path prompt should restate the active concrete step."""

        from agent.therapeutic import guided_exercise
        from agent.therapeutic.guided_exercise import run_guided_exercise_response_node

        monkeypatch.setattr(
            guided_exercise, "get_stream_writer", lambda: lambda _: None
        )

        runtime = _MockRuntime(llm_client=_RecordingTextLLM())
        state: Any = {
            "message": "That makes sense. I can keep going.",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        llm = cast(_RecordingTextLLM, runtime.context.llm_client)
        assert delta["response_text"] == "recorded"
        assert llm.prompts
        prompt = llm.prompts[0].lower()
        assert "restate this same step" in prompt
        assert "things you can see around you right now" in prompt

    @pytest.mark.asyncio
    async def test_guided_exercise_resume_request_holds_current_step(self) -> None:
        """A request to resume the exercise should not be treated as exit."""

        from agent.therapeutic.guided_exercise import run_guided_exercise_response_node

        fake = _FakeStepStateLLM(step_state="exit")
        runtime = _MockRuntime(llm_client=fake)
        state: Any = {
            "message": "Okay, let's go back to the grounding step.",
            "history": [],
            "session_progress": {"turn_count": 4},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        assert fake.structured_calls == 0
        if "exercise_state" in delta:
            assert delta["exercise_state"]["exercise_step"] == 0
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_stuck_offers_rephrase(self) -> None:
        """An explicit 'I can't' response triggers the stuck/rephrase path.

        The user can't engage with the step. The node should hold the
        step (not advance) AND offer to simplify. The stuck path is
        the escalation step above hold — same state behavior (no
        advancement), different response tone (offer to make it
        smaller).
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "I can't focus on this right now.",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # STUCK: exercise_state is NOT updated (step stays at 0)
        if "exercise_state" in delta:
            assert delta["exercise_state"]["exercise_step"] == 0
        # Response offers a smaller version of the step
        text_lower = delta["response_text"].lower()
        assert "smaller" in text_lower or "one thing" in text_lower
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_exit_clears_state(self) -> None:
        """An exit signal clears exercise_type and exercise_step.

        The user wants to stop the exercise. The node must null out
        both exercise-state fields so the next dispatcher turn does NOT
        take the active-exercise fast path.
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "This isn't helping, can we just talk?",
            "history": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # EXIT: exercise-state fields must be cleared (None)
        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        # Response acknowledges the exit without defending the exercise
        text_lower = delta["response_text"].lower()
        assert "stop" in text_lower or "of course" in text_lower
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_last_step_completion_clears_state(self) -> None:
        """Completing the LAST step of the exercise clears exercise state.

        The 5-4-3-2-1 grounding exercise has 5 steps (indexes 0-4).
        A completion-triggering response on step 4 should finish the
        exercise, not advance to step 5 (which doesn't exist).
        """

        from agent.therapeutic.guided_exercise import (
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        # State on the LAST step (index 4 = "one thing you can taste")
        state: Any = {
            "message": "Coffee. That's what I can taste right now.",
            "session_id": "test-routing",
            "history": [],
            "session_progress": {"turn_count": 6},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 4,
            },
        }

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
        )

        # Exercise completes naturally — state is cleared
        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        # Response names what the user just did
        text_lower = delta["response_text"].lower()
        assert "grounding" in text_lower or "walked" in text_lower
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_crisis_still_routes_to_crisis_not_therapeutic(self) -> None:
        """Non-therapeutic regression guard: crisis messages bypass the subgraph."""

        result = await run_agent(
            AgentInput(message="I have pills and I am going to kill myself tonight.")
        )

        assert result.response_type == ResponseCategory.CRISIS
        assert result.crisis.level == 3
        assert result.crisis.needs_crisis_response is True
        # The therapeutic subgraph should NOT have set the response style
        assert result.response_style != "supportive"
        assert result.response_style != "reflective"
        assert result.response_style != "clarifying"
        assert result.response_style != "psychoeducation"
        assert result.response_style != "closing"
        assert result.response_style != "guided_exercise"

    @pytest.mark.asyncio
    async def test_ambiguous_concerning_language_routes_to_therapeutic(
        self,
    ) -> None:
        """Level-1 ambiguous messages (not crisis) should reach the therapeutic branch.

        This is the regression case for the Stage E rewiring: under the
        old topology, non-crisis traffic terminated at END with the
        bootstrap reply. After Stage E, it should produce a real response style
        response.
        """

        result = await run_agent(
            AgentInput(message="I just wish I could disappear for a while.")
        )

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.response_style in {"supportive", "reflective", "clarifying"}
        assert result.response_text
        # Critically: response text is NOT the bootstrap stub
        assert "Persistent mode is active" not in result.response_text
        assert "Guest mode is active" not in result.response_text

    @pytest.mark.asyncio
    async def test_level_one_crisis_does_not_escalate_to_crisis(self) -> None:
        """Level-1 crisis ambiguity should stay therapeutic, not escalate."""

        result = await run_agent(
            AgentInput(
                message=(
                    "I don't know how much longer I can do this. "
                    "Everything feels hopeless."
                )
            )
        )

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.crisis.level == 1
        assert result.crisis.needs_clarification is True
        assert result.response_style in {"supportive", "clarifying"}
        assert "emergency services" not in result.response_text.lower()
        assert "hotline" not in result.response_text.lower()


# ── Mid-exercise therapeutic approach preservation tests ─────────────────────────


class TestMidExerciseTherapeuticApproachPreservation:
    """Verify therapeutic approach routing behavior on mid-exercise side-turns."""

    @pytest.mark.asyncio
    async def test_clarifying_preserves_exercise_therapeutic_approach(self) -> None:
        """A clarifying side-turn preserves the exercise's therapeutic approach in routing."""

        llm = _FakeDispatchLLM(
            response_style="clarifying",
            therapeutic_approach="grief_support",
        )
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": "what do you mean by notice?",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
            },
            "therapeutic_approach": "cbt",
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        # Clarifying should preserve the exercise's existing therapeutic approach (cbt),
        # NOT use the LLM's fresh pick (grief_support).
        assert cmd.update["therapeutic_approach"] == "cbt"
        assert cmd.goto == CLARIFYING_NODE
        # Exercise state must still be intact
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_clarifying_uses_pinned_exercise_therapeutic_approach_when_routing_missing(
        self,
    ) -> None:
        """A clarifying side-turn falls back to exercise_state.exercise_therapeutic_approach."""

        llm = _FakeDispatchLLM(
            response_style="clarifying",
            therapeutic_approach="grief_support",
        )
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": "what do you mean by notice?",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
                "exercise_therapeutic_approach": "cbt",
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.update["therapeutic_approach"] == "cbt"
        assert cmd.goto == CLARIFYING_NODE
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_active_exercise_instruction_question_bypasses_llm(
        self,
    ) -> None:
        """Instruction clarification should not let LLM routing clear exercise."""

        llm = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="none",
        )
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": (
                "Do you mean things I can see right now, or just around me in general?"
            ),
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 0,
                "exercise_therapeutic_approach": "dbt_skills",
            },
            "therapeutic_approach": "dbt_skills",
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 0
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert cmd.goto == CLARIFYING_NODE
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_psychoeducation_uses_fresh_therapeutic_approach(self) -> None:
        """A psychoeducation side-turn uses the LLM's fresh therapeutic approach pick."""

        llm = _FakeDispatchLLM(
            response_style="psychoeducation",
            therapeutic_approach="grief_support",
        )
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": "how does grief work?",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
            },
            "therapeutic_approach": "cbt",
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        # Psychoeducation should use the LLM's fresh pick (grief_support),
        # NOT the exercise's therapeutic approach (cbt).
        assert cmd.update["therapeutic_approach"] == "grief_support"
        assert cmd.goto == PSYCHOEDUCATION_NODE
        # Exercise state must still be intact
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_regex_active_exercise_uses_pinned_therapeutic_approach_when_routing_missing(
        self,
    ) -> None:
        """Regex continuation falls back to exercise_state.exercise_therapeutic_approach."""

        runtime = _MockRuntime(llm_client=None)
        state: Any = {
            "message": "okay",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
                "exercise_therapeutic_approach": "act",
            },
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.update["therapeutic_approach"] == "act"
        assert cmd.goto == GUIDED_EXERCISE_NODE

    @pytest.mark.asyncio
    async def test_deterministic_exit_clears_exercise_therapeutic_approach(
        self,
    ) -> None:
        """Deterministic exit override clears exercise_therapeutic_approach in exercise_state."""

        llm = _FakeDispatchLLM(response_style="supportive")
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": "never mind",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
                "exercise_therapeutic_approach": "cbt",
            },
            "therapeutic_approach": "cbt",
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        # Exit must clear all exercise state including therapeutic approach
        assert cmd.update["exercise_state"]["exercise_type"] is None
        assert cmd.update["exercise_state"]["exercise_step"] is None
        assert cmd.update["exercise_state"]["exercise_therapeutic_approach"] is None
        assert cmd.goto == SUPPORTIVE_NODE

    @pytest.mark.asyncio
    async def test_llm_exit_clears_exercise_therapeutic_approach(self) -> None:
        """LLM-driven exit from active exercise clears exercise_therapeutic_approach."""

        llm = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="none",
        )
        runtime = _MockRuntime(llm_client=llm)
        state: Any = {
            "message": "actually let's talk about something else",
            "history": [],
            "session_progress": {"turn_count": 3},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
                "exercise_therapeutic_approach": "cbt",
            },
            "therapeutic_approach": "cbt",
        }

        cmd = await run_therapeutic_dispatch_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert cmd.update["exercise_state"]["exercise_type"] is None
        assert cmd.update["exercise_state"]["exercise_step"] is None
        assert cmd.update["exercise_state"]["exercise_therapeutic_approach"] is None


# ─── Anaphoric/walkthrough guidance guard ────────────────────────────────────


_OFFER_GROUNDING = "Would you like to try a grounding exercise?"


def _state_with_offer(message: str) -> AgentState:
    """Build a state where the prior assistant turn offered grounding."""

    return _build_state(
        message,
        history=[
            {"role": "user", "content": "I'm anxious."},
            {"role": "assistant", "content": _OFFER_GROUNDING},
        ],
    )


def _state_with_active_exercise(message: str) -> AgentState:
    """Build a state with an active grounding exercise in progress."""

    state: Any = {
        "message": message,
        "history": [],
        "exercise_state": {
            "exercise_type": "grounding",
            "exercise_step": 2,
            "exercise_therapeutic_approach": "dbt_skills",
        },
    }
    return cast(AgentState, state)


class TestAdviceRequestWithoutExerciseConsent:
    """Pure-function tests for ``_is_advice_request_without_exercise_consent``.

    Test matrix follows ``UNCONSENTED_EXERCISE_FIX_PLAN.md`` Step 5a. The cases
    encode the regex-shape decisions made across 14 adversarial-review
    iterations.
    """

    @pytest.mark.parametrize(
        "message",
        [
            # Anaphoric — phrasal terminal pronouns (Branch 1).
            "how to break out of it",
            "how do I break out of this",
            "how do I break out of this.",
            "how do I break out of this?",
            # Anaphoric — phrasal verb + pattern noun (Branch 2).
            "how do I get out of this loop",
            "how do I break out of this cycle",
            "how do I snap out of this spiral",
            "how do I get out of this pattern",
            # Anaphoric — bare verb + pattern noun (Branch 3).
            "how do I break this cycle",
            "how do I break the pattern",
            "how do I break this habit",
            "how do I stop this pattern",
            # Anaphoric — "stop doing this" (Branch 4) + softeners.
            "how do I stop doing this",
            "how can I stop doing that",
            "how do I just stop doing this in general",
            "how do I stop doing this for good",
            # Anaphoric — Branches 5/6.
            "what do I do about this",
            "what now?",
            # Informational walkthrough — wh form.
            "walk me through why this happens",
            "guide me through what just happened",
            "walk me through how this works",
            "walk me through what grounding actually does",
            "walk me through how STOP works",
            # Informational walkthrough — tool noun + non-completer trailing.
            "walk me through grounding theory",
            "walk me through breathing problems",
            "walk me through breathing technique problems",
            # Note: "walk me through grounding exercise theory" is documented
            # as out of scope — the inherited bare-noun pattern in
            # EXPLICIT_EXERCISE_REQUEST_PATTERNS matches "grounding exercise"
            # as consent. See UNCONSENTED_EXERCISE_FIX_PLAN.md Risk section
            # ("negated exercise mentions / pre-existing condition").
        ],
    )
    def test_should_fire(self, message: str) -> None:
        state = _build_state(message)
        assert _is_advice_request_without_exercise_consent(state, message) is True

    @pytest.mark.parametrize(
        "message",
        [
            # Content references with bare pronouns — must not fire.
            "how do I break it to my mom",
            "how do I break it down for them",
            "how do I fix it with my partner",
            "how do I get out of this lease",
            "how do I get out of this relationship",
            "how do I tell my partner about this",
            "how do I tell my partner that I love this",
            # Bare "how do I stop this" — intentional gap.
            "how do I stop this",
            # Statements, not questions.
            "I need to stop doing this",
            "I want to stop",
            # Explicit exercise request via canonical trigger.
            "let's do a thought record",
            "ground me",
            # Walkthrough WITH exercise/tool noun as completed direct object.
            "walk me through grounding",
            "guide me through breathing",
            "walk me through a thought record",
            "walk me through grounding exercise",
            "walk me through a grounding exercise",
            "walk me through a short grounding practice",
            "walk me through your favorite breathing technique",
            "walk me through some grounding",
            "walk me through how to do grounding",
            "walk me through how to work through a thought record",
            "walk me through how to fill out a thought record",
            # Prompt-only consent triggers.
            "can we figure out a way to test it",
            "can we figure out a way to test the thought",
            "can we look at what actually matters to me",
            # Combined anaphoric + consent — consent wins.
            "how do I break this cycle, can we figure out a way to test it?",
            "how do I get out of this loop, can we look at what actually matters to me?",
        ],
    )
    def test_should_not_fire(self, message: str) -> None:
        state = _build_state(message)
        assert _is_advice_request_without_exercise_consent(state, message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "yes",
            "yeah",
            "sure",
            "ok",
            "okay",
            "yes please",
            "sure, let's try it.",
            "let's try it",
            "let's do it",
            "go ahead",
            "sounds good",
            "yes, please",
            "okay we can try",
            "yeah let's do that",
            "okay, walk me through it",
        ],
    )
    def test_clean_acceptance_after_offer_does_not_fire(self, message: str) -> None:
        state = _state_with_offer(message)
        assert _is_advice_request_without_exercise_consent(state, message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "how do I break this cycle",
            "how do I stop doing this",
            # Acknowledgment + new question is NOT an acceptance — guard fires.
            "yes, that makes sense, how do I break this cycle?",
            "yes, that makes sense, but how do I stop doing this?",
        ],
    )
    def test_non_acceptance_after_offer_still_fires(self, message: str) -> None:
        state = _state_with_offer(message)
        assert _is_advice_request_without_exercise_consent(state, message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "how to break out of it",
            "how do I break this cycle",
            "walk me through why this happens",
        ],
    )
    def test_active_exercise_suppresses_guard(self, message: str) -> None:
        state = _state_with_active_exercise(message)
        assert _is_advice_request_without_exercise_consent(state, message) is False


class TestBareAcknowledgmentClarifyingOverride:
    """Short-turn acknowledgment routing should distinguish bare acks from continuers."""

    @pytest.mark.parametrize("message", ["yeah", "ok", "okay", "sure"])
    def test_bare_ack_after_open_question_triggers_clarifying(
        self, message: str
    ) -> None:
        state = _build_state(
            message,
            history=[
                {"role": "user", "content": "Work has been rough."},
                {
                    "role": "assistant",
                    "content": "What part of it has been hitting the hardest?",
                },
            ],
        )

        assert _is_bare_ack_to_open_question(state, message) is True

    @pytest.mark.parametrize(
        "message",
        ["yeah that makes sense", "got it", "fair", "right", "okay yeah"],
    )
    def test_soft_continuer_after_open_question_does_not_trigger_clarifying(
        self, message: str
    ) -> None:
        state = _build_state(
            message,
            history=[
                {"role": "user", "content": "Work has been rough."},
                {
                    "role": "assistant",
                    "content": "What part of it has been hitting the hardest?",
                },
            ],
        )

        assert _is_bare_ack_to_open_question(state, message) is False


class TestAnaphoricGuardIntegration:
    """Integration tests via ``run_therapeutic_dispatch_node`` (Step 5b).

    Each test mocks the LLM to return ``guided_exercise`` + a therapeutic approach, then
    asserts the guard rewrites (or correctly declines to rewrite) routing.
    """

    PANIC_HISTORY: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "I keep overworking and panicking when I try to slow down.",
        },
        {
            "role": "assistant",
            "content": "Yeah — control becomes the way you buy safety.",
        },
        {"role": "user", "content": "Yes you are right."},
        {
            "role": "assistant",
            "content": "That fits with what you've been describing.",
        },
    ]

    @pytest.mark.parametrize(
        "message",
        [
            "How to break out of it",
            "How do I stop doing this",
            "How do I break this cycle",
            "How do I just stop doing this in general",
            "walk me through why this happens",
            "walk me through what grounding actually does",
            "walk me through how STOP works",
            "walk me through grounding theory",
            "walk me through breathing problems",
        ],
    )
    @pytest.mark.asyncio
    async def test_guard_rewrites_to_psychoeducation_with_therapeutic_approach_preserved(
        self, message: str
    ) -> None:
        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(message, history=self.PANIC_HISTORY)

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == PSYCHOEDUCATION_NODE
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert fake.structured_calls == 1

    @pytest.mark.parametrize(
        "message",
        [
            "walk me through how to do grounding",
            "walk me through a short grounding practice",
            "walk me through your favorite breathing technique",
            "walk me through how to fill out a thought record",
            "can you walk me through a grounding exercise",
            "ground me",
            "can we figure out a way to test it",
            "can we look at what actually matters to me",
            "how do I break this cycle, can we figure out a way to test it?",
        ],
    )
    @pytest.mark.asyncio
    async def test_explicit_consent_routes_to_guided_exercise(
        self, message: str
    ) -> None:
        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(message, history=self.PANIC_HISTORY)

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE

    @pytest.mark.parametrize(
        "message",
        [
            "yes, please",
            "yes",
            "go ahead",
            "okay we can try",
            "yeah let's do that",
            "okay, walk me through it",
        ],
    )
    @pytest.mark.asyncio
    async def test_clean_acceptance_after_offer_routes_to_guided_exercise(
        self, message: str
    ) -> None:
        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            message,
            history=[
                {"role": "user", "content": "I'm anxious."},
                {"role": "assistant", "content": _OFFER_GROUNDING},
            ],
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE

    @pytest.mark.parametrize(
        "message",
        [
            "how do I stop doing this",
            "yes, that makes sense, but how do I stop doing this?",
        ],
    )
    @pytest.mark.asyncio
    async def test_non_acceptance_after_offer_still_rewrites(
        self, message: str
    ) -> None:
        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            message,
            history=[
                {"role": "user", "content": "I'm anxious."},
                {"role": "assistant", "content": _OFFER_GROUNDING},
            ],
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == PSYCHOEDUCATION_NODE


class TestCopingAdviceConsentNowBroader:
    """Step 5c — the existing coping-advice guard now uses the broader
    ``EXERCISE_CONSENT_PATTERNS`` superset, so combined utterances with a
    consent clause keep the LLM's guided_exercise pick.
    """

    @pytest.mark.asyncio
    async def test_combined_advice_plus_cbt_consent_keeps_guided_exercise(
        self,
    ) -> None:
        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="cbt",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "what are some ways to cope, can we figure out a way to test it?"
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE


class TestPromptTriggerContract:
    """Step 5d — bidirectional prompt-trigger contract.

    Two tests pin the canonical-list-as-source-of-truth invariant:

    1. The delimited trigger sentence is mechanically rendered AND no other
       trigger-list anchor or canonical multi-word-imperative trigger appears
       outside the delimited span.
    2. Every canonical trigger matches ``EXERCISE_CONSENT_PATTERNS``.
    """

    _TRIGGER_LIST_ANCHORS: tuple[str, ...] = (
        "Trigger phrases include:",
        "Triggers include:",
        "Examples of triggers:",
        "Trigger examples:",
        "Explicit starts include:",
        "trigger phrases such as",
    )

    # Triggers that are also generic therapeutic vocabulary and may legitimately
    # appear in non-trigger prose elsewhere in the prompt.
    _GENERIC_VOCAB_EXEMPTIONS: frozenset[str] = frozenset(
        {
            "behavioral experiment",
            "breathing exercise",
            "gratitude exercise",
        }
    )

    def test_dispatcher_prompt_trigger_sentence_is_mechanically_rendered(self) -> None:
        prompt = build_therapeutic_dispatch_system_prompt()

        expected_span = (
            "<!-- triggers:start -->Trigger phrases include: "
            + _format_prompt_trigger_phrases()
            + ".<!-- triggers:end -->"
        )

        assert prompt.count("<!-- triggers:start -->") == 1, (
            "Expected exactly one start delimiter"
        )
        assert prompt.count("<!-- triggers:end -->") == 1, (
            "Expected exactly one end delimiter"
        )

        import re as _re

        span_match = _re.search(
            r"<!-- triggers:start -->(.*?)<!-- triggers:end -->",
            prompt,
            _re.DOTALL,
        )
        assert span_match is not None
        assert span_match.group(0) == expected_span, (
            "Trigger span content drifted from canonical list"
        )

        prompt_outside_span = prompt.replace(span_match.group(0), "")

        for anchor in self._TRIGGER_LIST_ANCHORS:
            assert anchor not in prompt_outside_span, (
                f"Trigger-list anchor {anchor!r} appears outside the delimited "
                "span. Add new triggers to _PROMPT_GUIDED_EXERCISE_TRIGGERS."
            )

        # Strong guarantee for unambiguous consent triggers.
        strong_guarded = tuple(
            t
            for t in _PROMPT_GUIDED_EXERCISE_TRIGGERS
            if t.lower() not in self._GENERIC_VOCAB_EXEMPTIONS
        )
        for trigger in strong_guarded:
            assert trigger.lower() not in prompt_outside_span.lower(), (
                f"Canonical trigger {trigger!r} appears outside the delimited "
                "span. Add it to _PROMPT_GUIDED_EXERCISE_TRIGGERS only."
            )

    def test_canonical_triggers_match_consent_patterns(self) -> None:
        for trigger in _PROMPT_GUIDED_EXERCISE_TRIGGERS:
            assert _matches_any(trigger, EXERCISE_CONSENT_PATTERNS), (
                f"Canonical trigger {trigger!r} does not match "
                "EXERCISE_CONSENT_PATTERNS"
            )
