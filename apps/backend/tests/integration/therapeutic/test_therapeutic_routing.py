"""Tests for therapeutic dispatch, response-style nodes, and subgraph wiring.

Covers three concerns:
    1. ``run_therapeutic_dispatch_node`` with mocked runtime + fake LLM
    2. ``build_therapeutic_subgraph`` compiles to the expected shape
    3. End-to-end via ``run_agent`` — each response style reaches its terminal
       node with the right routing metadata and a non-empty response

These are unit/integration tests that run in the default pytest suite.
Dataset-driven evals and live-API tests live in Stage G2.
"""

from __future__ import annotations

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
    GUIDED_EXERCISE_NODE,
    THERAPEUTIC_RESPONSE_NODE,
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _format_prompt_trigger_phrases,
    build_therapeutic_dispatch_system_prompt,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.exercises.node import (
    run_guided_exercise_response_node as _run_guided_exercise_response_node,
)
from agent.therapeutic.exercises.types import ExerciseStepDecision
from agent.therapeutic.graph import (
    TherapeuticSubgraphInput,
    TherapeuticSubgraphOutput,
    build_therapeutic_subgraph,
)
from llm.base import BaseLLMClient, StructuredResponseT


# ─── Fake LLM client for dispatcher integration tests ────────────────────


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the guided exercise node with a no-op test stream writer.

    Args:
        state (AgentState): Test graph state.
        runtime (Any): Runtime stub.
        **kwargs (Any): Optional node keyword overrides.

    Returns:
        dict[str, Any]: Node delta.
    """

    kwargs.setdefault("stream_writer_factory", lambda: lambda _: None)
    return await _run_guided_exercise_response_node(state, runtime, **kwargs)


class _FakeDispatchLLM(BaseLLMClient):
    """Fake LLM client that returns a canned :class:`DispatchDecision`.

    Used to exercise the dispatcher's LLM path without hitting a real
    provider. Call counts are tracked so tests can assert whether the
    LLM was actually invoked.
    """

    def __init__(
        self,
        *,
        response_style: str = "supportive",
        therapeutic_approach: str = "none",
        should_raise: bool = False,
        text_should_raise: bool = False,
    ) -> None:
        self.response_style = response_style
        self.therapeutic_approach = therapeutic_approach
        self.should_raise = should_raise
        self.text_should_raise = text_should_raise
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
        if self.text_should_raise:
            raise RuntimeError("simulated response LLM failure")
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
        if response_schema.__name__ == "CrisisAssessmentSchema":
            prompt_lower = prompt.lower()
            level = (
                1
                if (
                    "disappear" in prompt_lower
                    or "hopeless" in prompt_lower
                    or "how much longer" in prompt_lower
                )
                else 0
            )
            return cast(
                StructuredResponseT,
                response_schema(
                    level=level,
                    confidence="high",
                    reason="fake crisis assessment",
                    needs_crisis_response=False,
                    needs_clarification=level == 1,
                ),
            )
        if response_schema.__name__ == "ExerciseSelectionDecision":
            return cast(
                StructuredResponseT,
                response_schema(
                    exercise_type="grounding_5_4_3_2_1",
                    reasoning="fake exercise selection",
                    confidence="high",
                ),
            )
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

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        response_llm: BaseLLMClient | None = None,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm,
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

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
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

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1  # LLM was called

    @pytest.mark.asyncio
    async def test_llm_primary_adds_dispatch_trace_reason(self) -> None:
        """Dispatch diagnostics should preserve the LLM routing reason."""

        fake = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="cbt",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("I need help slowing down.")
        state["diagnostics"] = {}

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]
        trace = cmd.update["diagnostics"]["routing_trace"]

        assert trace[-1]["stage"] == "dispatch"
        assert trace[-1]["decision"] == "supportive/cbt"
        assert trace[-1]["source"] == "llm_primary"
        assert trace[-1]["reason"] == "fake dispatch decision"

    @pytest.mark.asyncio
    async def test_llm_path_routes_to_llm_pick(self) -> None:
        """Ambiguous messages go to the LLM and use its decision."""

        fake = _FakeDispatchLLM(response_style="reflective")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state(
            "I keep finding myself getting frustrated with my sister for no reason."
        )

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_bare_ack_to_open_question_routed_by_llm(self) -> None:
        """A bare "yes" after an open question — LLM decides routing."""

        fake = _FakeDispatchLLM(response_style="clarifying")
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

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_failure_propagates_exception(self) -> None:
        """LLM exceptions propagate — the retry policy handles transients."""

        fake = _FakeDispatchLLM(should_raise=True)
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("I feel really sad today.")

        with pytest.raises(RuntimeError, match="simulated LLM failure"):
            await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_pick_psychoeducation_uses_shared_response_node(
        self,
    ) -> None:
        """Psychoeducation is preserved as style context on the shared node."""

        fake = _FakeDispatchLLM(response_style="psychoeducation")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Why do I get so tired in the afternoon every day?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_pick_closing_uses_shared_response_node(self) -> None:
        """Closing is preserved as style context on the shared node."""

        fake = _FakeDispatchLLM(response_style="closing")
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("Thanks, I think I need to step away now.")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_llm_pick_guided_exercise_routes_to_guided_exercise_node(
        self,
    ) -> None:
        """LLM picks for guided_exercise route to the guided_exercise node.

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
                THERAPEUTIC_RESPONSE_NODE,
            ),
            (
                "psychoeducation",
                "dbt_skills",
                "Why does my distress spike so fast when conflict starts?",
                THERAPEUTIC_RESPONSE_NODE,
            ),
            (
                "technique",
                "dbt_skills",
                "Can you help me work through the exact skill for this moment?",
                THERAPEUTIC_RESPONSE_NODE,
            ),
            (
                "supportive",
                "pfa",
                "I am really shaken and need a steady response.",
                THERAPEUTIC_RESPONSE_NODE,
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
    async def test_llm_guided_exercise_pick_routes_directly(self) -> None:
        """LLM guided_exercise picks now route directly without guards."""

        fake = _FakeDispatchLLM(
            response_style="guided_exercise",
            therapeutic_approach="dbt_skills",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("What are some tips to cope at different severity levels?")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

        assert cmd.goto == GUIDED_EXERCISE_NODE
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_active_exercise_llm_pick_routes_guided_exercise(self) -> None:
        """An active exercise continues when the LLM keeps guided_exercise."""

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
    async def test_active_exercise_stop_request_clears_state(self) -> None:
        """An explicit mid-exercise stop request — LLM routes away, state cleared."""

        fake = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="none",
        )
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

        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1
        update = cast(dict[str, Any], cmd.update)
        assert update["exercise_state"]["exercise_type"] is None
        assert update["exercise_state"]["exercise_step"] is None

    @pytest.mark.asyncio
    async def test_inactive_exercise_state_does_not_force_guided(self) -> None:
        """Cleared exercise fields do not force a guided-exercise route."""

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

        # The LLM path should run and pick supportive (what the fake returns).
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert fake.structured_calls == 1

    @pytest.mark.asyncio
    async def test_command_update_contains_therapeutic_approach(self) -> None:
        """The dispatcher's Command carries top-level ``therapeutic_approach``."""

        fake = _FakeDispatchLLM(
            response_style="supportive",
            therapeutic_approach="motivational_interviewing",
        )
        runtime = _MockRuntime(llm_client=fake)
        state = _build_state("I had a rough day at work")

        cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

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
            AgentInput(message="I had a really rough day at work today."),
            llm_client=_FakeDispatchLLM(),
        )

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert result.response_style == "supportive"
        assert result.response_text == "fake"

    @pytest.mark.asyncio
    async def test_end_to_end_diagnostics_include_dispatch_trace(self) -> None:
        """Dispatch trace should cross the therapeutic subgraph boundary."""

        result = await run_agent(
            AgentInput(message="I had a really rough day."),
            llm_client=_FakeDispatchLLM(),
        )
        trace = result.diagnostics["routing_trace"]

        assert any(entry["stage"] == "dispatch" for entry in trace)
        assert trace[-1]["stage"] == "dispatch"
        assert trace[-1]["decision"].startswith("supportive")

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
    async def test_technique_node_passes_llm_response_directly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Technique node should pass through LLM response without post-processing."""

        from agent.therapeutic import response
        from agent.therapeutic.response import run_therapeutic_response_node

        monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

        runtime = _MockRuntime(
            llm_client=_RecordingTextLLM(
                "Absolutely. Let's take it one piece at a time.\n\n"
                "What's the exact thought you want to examine, in your own words?"
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
        assert delta["response_text"].startswith("Absolutely.")

    @pytest.mark.asyncio
    async def test_response_node_requires_llm_client(self) -> None:
        """Non-exercise response generation should fail without a response LLM."""

        from agent.therapeutic.response import run_therapeutic_response_node

        runtime = _MockRuntime(llm_client=None)
        state = _build_state("I had a rough day.")
        state["response_style"] = "supportive"

        with pytest.raises(RuntimeError, match="No LLM client available"):
            await run_therapeutic_response_node(state, runtime)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_response_llm_failure_propagates_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Response LLM failures should propagate to the graph retry policy."""

        from agent.therapeutic import response
        from agent.therapeutic.response import run_therapeutic_response_node

        monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

        runtime = _MockRuntime(llm_client=_FakeDispatchLLM(text_should_raise=True))
        state = _build_state("I had a rough day.")
        state["response_style"] = "supportive"

        with pytest.raises(RuntimeError, match="simulated response LLM failure"):
            await run_therapeutic_response_node(state, runtime)  # type: ignore[arg-type]

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
    async def test_guided_exercise_start_writes_exercise_state(self) -> None:
        """Starting a new exercise sets exercise_type and exercise_step.

        Entry condition 1 for the guided_exercise node: no active
        exercise, so the node should start 5-4-3-2-1 grounding at
        step 0. The delta must include exercise_state.exercise_type and
        exercise_state.exercise_step.
        """

        runtime = _MockRuntime(llm_client=_FakeDispatchLLM())
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

        assert delta["response_text"]
        assert delta["response_style"] == "guided_exercise"
        assert delta["exercise_state"]["exercise_type"] == "grounding_5_4_3_2_1"
        assert delta["exercise_state"]["exercise_step"] == 0

    @pytest.mark.asyncio
    async def test_guided_exercise_advances_on_complete_response(self) -> None:
        """A completion-triggering response advances exercise_step by 1.

        Entry condition 2 for the guided_exercise node: exercise is
        active. The user names enough items to count as complete for
        the current step; the node should advance to step+1.
        """

        runtime = _MockRuntime(llm_client=_FakeStepStateLLM(step_state="complete"))
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
        assert delta["response_text"]
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_holds_on_tentative_response(self) -> None:
        """A tentative response keeps the step unchanged (hold, not advance).

        The user names fewer items than the step's min_count threshold,
        so the classifier returns HOLD. The exercise_step stays where
        it is and the response offers space.
        """

        runtime = _MockRuntime(llm_client=_FakeStepStateLLM(step_state="hold"))
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
        assert delta["response_text"]
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_hold_prompt_reanchors_same_step(
        self,
    ) -> None:
        """Hold-path prompt should restate the active concrete step."""

        response_llm = _RecordingTextLLM()
        runtime = _MockRuntime(
            llm_client=_FakeStepStateLLM(step_state="hold"),
            response_llm=response_llm,
        )
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
            stream_writer_factory=lambda: lambda _: None,
        )

        assert delta["response_text"] == "recorded"
        assert response_llm.prompts
        prompt = response_llm.prompts[0].lower()
        assert "restate this same step" in prompt
        assert "things you can see around you right now" in prompt

    @pytest.mark.asyncio
    async def test_guided_exercise_resume_request_uses_llm_decision(self) -> None:
        """A request to resume the exercise is classified by the LLM."""

        fake = _FakeStepStateLLM(step_state="hold")
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

        assert fake.structured_calls == 1
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

        runtime = _MockRuntime(llm_client=_FakeStepStateLLM(step_state="stuck"))
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
        assert delta["response_text"]
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_exit_clears_state(self) -> None:
        """An exit signal clears exercise_type and exercise_step.

        The user wants to stop the exercise. The node must null out
        both exercise-state fields so the next dispatcher turn does NOT
        see an active exercise.
        """

        runtime = _MockRuntime(llm_client=_FakeStepStateLLM(step_state="exit"))
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
        assert delta["response_text"]
        assert delta["response_style"] == "guided_exercise"

    @pytest.mark.asyncio
    async def test_guided_exercise_last_step_completion_clears_state(self) -> None:
        """Completing the LAST step of the exercise clears exercise state.

        The 5-4-3-2-1 grounding exercise has 5 steps (indexes 0-4).
        A completion-triggering response on step 4 should finish the
        exercise, not advance to step 5 (which doesn't exist).
        """

        runtime = _MockRuntime(llm_client=_FakeStepStateLLM(step_state="complete"))
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
        assert delta["response_text"]
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
            AgentInput(message="I just wish I could disappear for a while."),
            llm_client=_FakeDispatchLLM(),
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
            ),
            llm_client=_FakeDispatchLLM(),
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
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
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
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_active_exercise_clarification_preserves_approach(
        self,
    ) -> None:
        """Clarification mid-exercise preserves exercise approach and doesn't clear state."""

        llm = _FakeDispatchLLM(
            response_style="clarifying",
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

        assert llm.structured_calls == 1
        assert cmd.update["therapeutic_approach"] == "dbt_skills"
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE
        assert "exercise_type" not in cmd.update.get("exercise_state", {})

    @pytest.mark.asyncio
    async def test_psychoeducation_clears_active_exercise(self) -> None:
        """A psychoeducation turn exits active exercise continuity."""

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

        assert cmd.update["therapeutic_approach"] == "grief_support"
        assert cmd.update["exercise_state"]["exercise_type"] is None
        assert cmd.update["exercise_state"]["exercise_step"] is None
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE

    @pytest.mark.asyncio
    async def test_no_llm_active_exercise_raises(
        self,
    ) -> None:
        """Therapeutic dispatch requires the classifier LLM."""

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

        with pytest.raises(RuntimeError, match="classifier LLM"):
            await run_therapeutic_dispatch_node(
                cast(AgentState, state),
                runtime,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_exit_route_clears_exercise_therapeutic_approach(
        self,
    ) -> None:
        """Routing away from an active exercise clears its pinned approach."""

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
        assert cmd.goto == THERAPEUTIC_RESPONSE_NODE

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


class TestPromptTriggerContract:
    """Prompt-trigger contract: the trigger sentence is mechanically rendered."""

    _TRIGGER_LIST_ANCHORS: tuple[str, ...] = (
        "Trigger phrases include:",
        "Triggers include:",
        "Examples of triggers:",
        "Trigger examples:",
        "Explicit starts include:",
        "trigger phrases such as",
    )

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
