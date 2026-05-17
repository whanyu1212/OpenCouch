"""Tests for the exercise expansion — completion modes and flows.

Tests cover:
1. ExerciseStep completion_mode field
2. LLM-driven guided exercise step flow
3. End-to-end flows for box breathing and thought record
4. Exit mid-exercise for confirmation-based exercises
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BEHAVIORAL_EXPERIMENT,
    EXERCISE_BOX_BREATHING,
    EXERCISE_CONTINUUM,
    EXERCISE_GRATITUDE,
    EXERCISE_IMPROVE,
    EXERCISE_LEAVES_ON_STREAM,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_SELF_COMPASSION,
    EXERCISE_STOP_TECHNIQUE,
    EXERCISE_THOUGHT_RECORD,
    EXERCISE_TINY_ACTION,
    EXERCISE_VALUES_COMPASS,
)
from agent.therapeutic.exercises.types import ExerciseDefinition, ExerciseStep
from agent.therapeutic.exercises.node import (
    run_guided_exercise_response_node as _run_guided_exercise_response_node,
)
from agent.therapeutic.exercises.memory import (
    ExerciseCompletionMemoryRequest,
    write_exercise_completion_fact,
)

# ── Helper ────────────────────────────────────────────────────────────


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


class _RecordingMemoryStore:
    """In-memory store that records aput calls for assertions."""

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    async def aput(
        self,
        namespace: Any,
        key: str,
        value: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.writes.append({"namespace": namespace, "key": key, "value": value})


class _MockRuntime:
    """Minimal runtime mock for exercise node tests."""

    def __init__(
        self,
        llm_client: Any = None,
        memory_store: Any = None,
        memory_mode: MemoryMode | str = MemoryMode.INCOGNITO,
    ) -> None:
        resolved_memory_mode = (
            memory_mode
            if isinstance(memory_mode, MemoryMode)
            else MemoryMode(memory_mode)
        )
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=memory_store or OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=resolved_memory_mode,
        )


class _StepClassifierLLM:
    """Fake LLM for guided-exercise step classification tests."""

    def __init__(
        self,
        *,
        step_state: str = "complete",
        exercise_type: str | None = None,
        response_text: str = "next step",
        fail_selection: bool = False,
        selection_confidence: str = "high",
    ) -> None:
        self.step_state = step_state
        self.exercise_type = exercise_type
        self.response_text = response_text
        self.fail_selection = fail_selection
        self.selection_confidence = selection_confidence
        self.structured_calls = 0

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: Any,
        system_instruction: str | None = None,
    ) -> Any:
        self.structured_calls += 1
        if response_schema.__name__ == "ExerciseSelectionDecision":
            if self.fail_selection:
                raise RuntimeError("selection failure")
            return response_schema(
                exercise_type=self.exercise_type or EXERCISE_BOX_BREATHING,
                reasoning="fake exercise selection",
                confidence=self.selection_confidence,
            )
        return response_schema(
            step_state=self.step_state,
            reasoning="fake step classification",
            confidence="high",
        )

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> Any:
        yield self.response_text


def _make_state(
    message: str,
    exercise_type: str | None = None,
    exercise_step: int | None = None,
) -> Any:
    """Build a minimal state dict for exercise tests."""

    exercise_state: dict[str, Any] = {}
    if exercise_type is not None:
        exercise_state["exercise_type"] = exercise_type
    if exercise_step is not None:
        exercise_state["exercise_step"] = exercise_step

    return {
        "message": message,
        "session_id": "test-exercise",
        "history": [],
        "session_progress": {"turn_count": 1},
        "exercise_state": exercise_state,
        "turn_lifecycle": {"active_flow": "none", "action": "none"},
    }


# ── End-to-end node tests ────────────────────────────────────────────


class TestBoxBreathingFlow:
    """End-to-end flow for box breathing exercise."""

    @pytest.mark.asyncio
    async def test_start_without_llm_raises(
        self,
    ) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I need an exercise")

        with pytest.raises(RuntimeError, match="classifier LLM"):
            await run_guided_exercise_response_node(
                cast(AgentState, state),
                runtime,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_confirmation_advances_box_breathing(self) -> None:
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="Good. Now hold the breath for four counts.",
            )
        )
        state = _make_state("done", EXERCISE_BOX_BREATHING, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_step"] == 1
        assert delta["exercise_state"]["exercise_step_id"] == "hold_full"
        assert "hold" in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_box_breathing_completion(self) -> None:
        """Completing the last step clears exercise state."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed box breathing.",
            )
        )
        # Step 3 is the last step (0-indexed, 4 steps total)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        assert delta["exercise_state"]["exercise_step_id"] is None
        assert delta["exercise_state"]["exercise_version"] is None
        assert "box breathing" in delta["response_text"].lower()


class TestThoughtRecordFlow:
    """End-to-end flow for simple thought record."""

    @pytest.mark.asyncio
    async def test_thought_record_advances_on_description(self) -> None:
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="Now identify the automatic thought.",
            )
        )
        state = _make_state(
            "I was at work and my boss asked to talk to me after the meeting",
            EXERCISE_THOUGHT_RECORD,
            0,
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_step"] == 1


class TestMuscleRelaxationFlow:
    """End-to-end flow for progressive muscle relaxation."""

    @pytest.mark.asyncio
    async def test_pmr_advances_on_confirmation(self) -> None:
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="Now move to your shoulders.",
            )
        )
        state = _make_state("done", EXERCISE_MUSCLE_RELAXATION, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_step"] == 1
        assert "shoulder" in delta["response_text"].lower()


class TestSelfCompassionFlow:
    """End-to-end flow for self-compassion break."""

    @pytest.mark.asyncio
    async def test_llm_selection_starts_self_compassion_without_keyword(self) -> None:
        llm = _StepClassifierLLM(
            exercise_type=EXERCISE_SELF_COMPASSION,
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("I'm being really harsh with myself today")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_SELF_COMPASSION
        assert delta["exercise_state"]["exercise_step"] == 0
        assert delta["exercise_state"]["exercise_step_id"] == "acknowledge_suffering"
        assert delta["exercise_state"]["exercise_version"] == 1

    @pytest.mark.asyncio
    async def test_low_confidence_selection_raises(self) -> None:
        llm = _StepClassifierLLM(selection_confidence="low")
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("Can we do something for this?")

        with pytest.raises(ValueError, match="low confidence"):
            await run_guided_exercise_response_node(
                cast(AgentState, state),
                runtime,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_llm_selection_failure_propagates(self) -> None:
        llm = _StepClassifierLLM(fail_selection=True)
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("Can we do an exercise?")

        with pytest.raises(RuntimeError, match="selection failure"):
            await run_guided_exercise_response_node(
                cast(AgentState, state),
                runtime,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_self_compassion_completes_in_3_steps(self) -> None:
        """Self-compassion break has 3 steps; completing step 2 should clear state."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed the self-compassion break.",
            )
        )
        # Step 2 is the last step (0-indexed, 3 steps total)
        state = _make_state(
            "May I give myself what I need right now",
            EXERCISE_SELF_COMPASSION,
            2,
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None

    @pytest.mark.asyncio
    async def test_llm_classifier_advances_natural_confirmation(self) -> None:
        """LLM step classification handles confirmations beyond regex coverage."""

        llm = _StepClassifierLLM(
            step_state="complete",
            response_text="Good. Now remind yourself that you're not alone.",
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state(
            "I sat with that for a moment",
            EXERCISE_SELF_COMPASSION,
            0,
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 1
        assert delta["exercise_state"]["exercise_step"] == 1
        assert "not alone" in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_items_step_uses_llm_decision(self) -> None:
        """Item-list steps are classified by the LLM, not local counting."""

        llm = _StepClassifierLLM(
            step_state="complete",
            response_text="Good. Now four things you can hear.",
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state(
            "Right now. I can see my desk, lamp, and window.",
            EXERCISE_5_4_3_2_1,
            0,
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 1
        assert delta["exercise_state"]["exercise_step"] == 1
        assert "hear" in delta["response_text"].lower()


class TestExitMidExercise:
    """Test exit from confirmation-based exercises."""

    @pytest.mark.asyncio
    async def test_exit_box_breathing_clears_state(self) -> None:
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="exit",
                response_text="We can stop. What would help now?",
            )
        )
        state = _make_state("I don't want to do this", EXERCISE_BOX_BREATHING, 1)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        assert (
            "stop" in delta["response_text"].lower()
            or "helpful" in delta["response_text"].lower()
        )

    @pytest.mark.asyncio
    async def test_explicit_exit_uses_llm_classifier(self) -> None:
        llm = _StepClassifierLLM(
            step_state="exit",
            response_text="We can stop. What would help now?",
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("stop, I want to just talk", EXERCISE_SELF_COMPASSION, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 1
        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None

    @pytest.mark.asyncio
    async def test_exit_leaves_on_stream_clears_state(self) -> None:
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="exit",
                response_text="We can stop. What would help now?",
            )
        )
        state = _make_state(
            "never mind, can we just talk", EXERCISE_LEAVES_ON_STREAM, 2
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None


class TestRegistryCompleteness:
    """Verify the registry and display names cover all exercises."""

    def test_catalog_contains_current_core_exercises(self) -> None:
        from agent.therapeutic.exercises.registry import (
            EXERCISE_5_4_3_2_1,
            EXERCISE_BOX_BREATHING,
            EXERCISE_LEAVES_ON_STREAM,
            EXERCISE_MUSCLE_RELAXATION,
            EXERCISE_SELF_COMPASSION,
            EXERCISE_THOUGHT_RECORD,
            iter_exercise_definitions,
        )

        expected = {
            EXERCISE_5_4_3_2_1,
            EXERCISE_BOX_BREATHING,
            EXERCISE_STOP_TECHNIQUE,
            EXERCISE_THOUGHT_RECORD,
            EXERCISE_TINY_ACTION,
            EXERCISE_LEAVES_ON_STREAM,
            EXERCISE_MUSCLE_RELAXATION,
            EXERCISE_BEHAVIORAL_EXPERIMENT,
            EXERCISE_SELF_COMPASSION,
            EXERCISE_IMPROVE,
            EXERCISE_VALUES_COMPASS,
            EXERCISE_GRATITUDE,
            EXERCISE_CONTINUUM,
        }
        registered = {definition.id for definition in iter_exercise_definitions()}
        assert expected.issubset(registered)

    def test_all_exercises_have_display_names(self) -> None:
        from agent.therapeutic.exercises.registry import (
            get_exercise_display_name,
            iter_exercise_definitions,
        )

        for definition in iter_exercise_definitions():
            assert get_exercise_display_name(definition.id) == definition.display_name

    def test_all_steps_have_instructions(self) -> None:
        from agent.therapeutic.exercises.registry import iter_exercise_definitions

        for definition in iter_exercise_definitions():
            for i, step in enumerate(definition.steps):
                assert step.instruction, (
                    f"Empty instruction for {definition.id} step {i}"
                )

    def test_catalog_public_helpers_match_definitions(self) -> None:
        from agent.therapeutic.exercises.registry import (
            get_exercise_definition,
            get_exercise_display_name,
            get_exercise_steps,
            iter_exercise_definitions,
            iter_exercise_selection_aliases,
            voice_exercise_ids,
        )

        definitions = iter_exercise_definitions()
        ids = [definition.id for definition in definitions]
        assert len(ids) == len(set(ids))
        registered = set(ids)

        for definition in definitions:
            assert get_exercise_definition(definition.id) == definition
            assert get_exercise_steps(definition.id) == definition.steps
            assert get_exercise_display_name(definition.id) == definition.display_name
            assert definition.version >= 1
            assert definition.category
            assert definition.tags
            assert definition.selection_use_case
            assert definition.selection_aliases
            assert all(alias.strip() for alias in definition.selection_aliases)
            assert definition.steps
            step_ids = [step.id for step in definition.steps]
            assert len(step_ids) == len(set(step_ids))
            assert all(step_id.strip() for step_id in step_ids)

        assert get_exercise_definition("not_registered") is None
        assert get_exercise_steps("not_registered") is None
        assert get_exercise_display_name("not_registered") == "not_registered"
        assert (
            get_exercise_display_name(
                "not_registered",
                default="fallback",
            )
            == "fallback"
        )

        alias_targets = {
            exercise_type for _, exercise_type in iter_exercise_selection_aliases()
        }
        assert alias_targets == registered

        voice_ids = set(voice_exercise_ids())
        expected_voice_ids = {
            definition.id for definition in definitions if definition.voice_supported
        }
        assert voice_ids == expected_voice_ids

    def test_availability_helpers_filter_by_capability_metadata(self) -> None:
        from agent.therapeutic.exercises.registry import (
            available_exercise_definitions,
        )

        def ids(definitions: tuple[ExerciseDefinition, ...]) -> tuple[str, ...]:
            return tuple(definition.id for definition in definitions)

        step = ExerciseStep(
            instruction="Try one small step.",
            completion_mode="confirmation",
        )
        basic = ExerciseDefinition(
            id="basic",
            display_name="Basic",
            selection_use_case="general support",
            steps=(step,),
            selection_aliases=("basic",),
        )
        gated = ExerciseDefinition(
            id="gated",
            display_name="Gated",
            selection_use_case="skill-gated support",
            steps=(step,),
            selection_aliases=("gated",),
            required_skill="advanced_exercises",
        )
        voice_only = ExerciseDefinition(
            id="voice_only",
            display_name="Voice Only",
            selection_use_case="voice support",
            steps=(step,),
            selection_aliases=("voice",),
            channels=("voice",),
        )
        cbt_only = ExerciseDefinition(
            id="cbt_only",
            display_name="CBT Only",
            selection_use_case="CBT support",
            steps=(step,),
            selection_aliases=("cbt",),
            approaches=("cbt",),
        )
        definitions = (basic, gated, voice_only, cbt_only)

        assert ids(available_exercise_definitions(definitions=definitions)) == (
            "basic",
        )
        assert ids(
            available_exercise_definitions(
                installed_skills=["advanced_exercises"],
                definitions=definitions,
            )
        ) == ("basic", "gated")
        assert ids(
            available_exercise_definitions(
                channel="voice",
                definitions=definitions,
            )
        ) == ("voice_only",)
        assert ids(
            available_exercise_definitions(
                therapeutic_approach="cbt",
                definitions=definitions,
            )
        ) == ("basic", "cbt_only")


# ── Memory write tests ───────────────────────────────────────────────


class TestExerciseCompletionMemory:
    """Tests that exercise completions write semantic facts to memory."""

    @pytest.mark.asyncio
    async def test_completion_writes_coping_strategy_fact(self) -> None:
        """Completing an exercise in persistent mode writes a semantic fact."""
        store = _RecordingMemoryStore()
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed box breathing.",
            ),
            memory_store=store,
            memory_mode="local",
        )
        # Last step of box breathing
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)
        state["user_id"] = "test-user"
        state["session_id"] = "test-session"

        await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert len(store.writes) == 1
        fact = store.writes[0]["value"]
        assert fact["category"] == "coping_strategy"
        assert fact["predicate"] == "USES"
        assert "box_breathing" in fact["object"]["identifier"]
        assert fact["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_completion_write_accepts_neutral_request(self) -> None:
        store = _RecordingMemoryStore()

        await write_exercise_completion_fact(
            request=ExerciseCompletionMemoryRequest(
                owner_id="test-user",
                session_id="test-session",
                turn_count=3,
                exercise_type=EXERCISE_BOX_BREATHING,
                display_name="box breathing",
            ),
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
        )

        assert len(store.writes) == 1
        fact = store.writes[0]["value"]
        assert fact["subject"]["identifier"] == "test-user"
        assert fact["source_session_id"] == "test-session"
        assert fact["source_turn_index"] == 3

    @pytest.mark.asyncio
    async def test_exit_does_not_write_fact(self) -> None:
        """Exiting an exercise does NOT write a memory fact."""
        store = _RecordingMemoryStore()
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="exit",
                response_text="We can stop.",
            ),
            memory_store=store,
            memory_mode="local",
        )
        state = _make_state("stop, I don't want to do this", EXERCISE_BOX_BREATHING, 1)

        await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert len(store.writes) == 0

    @pytest.mark.asyncio
    async def test_incognito_does_not_write_fact(self) -> None:
        """Completing an exercise in incognito mode does NOT write a fact."""
        store = _RecordingMemoryStore()
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed box breathing.",
            ),
            memory_store=store,
            memory_mode="incognito",
        )
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert len(store.writes) == 0

    @pytest.mark.asyncio
    async def test_no_store_does_not_error(self) -> None:
        """Completing with no memory store configured doesn't crash."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed box breathing.",
            ),
            memory_store=None,
            memory_mode="local",
        )
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        # Should complete normally without error
        assert delta["exercise_state"]["exercise_type"] is None


# ── Exercise therapeutic approach lifecycle tests ────────────────────────────────


class TestExerciseTherapeuticApproach:
    """Verify exercise_therapeutic_approach is captured at start, cleared on exit/completion,
    and used by the prompt builder."""

    @pytest.mark.asyncio
    async def test_start_captures_routing_approach(self) -> None:
        """Starting an exercise via LLM stores routing.therapeutic_approach in exercise_state."""
        llm = _StepClassifierLLM(
            exercise_type=EXERCISE_THOUGHT_RECORD,
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("let's do a thought record")
        state["therapeutic_approach"] = "cbt"

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_THOUGHT_RECORD
        assert delta["exercise_state"]["exercise_therapeutic_approach"] == "cbt"

    @pytest.mark.asyncio
    async def test_start_without_approach_stores_none(self) -> None:
        """Starting with no therapeutic approach stores None (approach-agnostic)."""
        llm = _StepClassifierLLM(
            exercise_type=EXERCISE_BOX_BREATHING,
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("can we do a breathing exercise")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_therapeutic_approach"] is None

    @pytest.mark.asyncio
    async def test_completion_clears_approach(self) -> None:
        """Completing the last step clears exercise_therapeutic_approach."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="You completed box breathing.",
            )
        )
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)
        state["exercise_state"]["exercise_therapeutic_approach"] = "dbt_skills"

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_therapeutic_approach"] is None

    @pytest.mark.asyncio
    async def test_exit_clears_approach(self) -> None:
        """Exiting mid-exercise clears exercise_therapeutic_approach."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="exit",
                response_text="We can stop.",
            )
        )
        state = _make_state("stop, I don't want to do this", EXERCISE_BOX_BREATHING, 1)
        state["exercise_state"]["exercise_therapeutic_approach"] = "dbt_skills"

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_therapeutic_approach"] is None

    def test_prompt_builder_prefers_exercise_therapeutic_approach(self) -> None:
        """build_guided_exercise_system_prompt reads exercise_state.exercise_therapeutic_approach
        over routing.therapeutic_approach when an exercise is active."""
        from agent.therapeutic.prompts import build_guided_exercise_system_prompt

        state: dict[str, Any] = {
            "exercise_state": {
                "exercise_type": "defusion_leaves_on_stream",
                "exercise_step": 1,
                "exercise_therapeutic_approach": "act",
            },
            "therapeutic_approach": "cbt",
            "working_memory": [],
            "procedural_profile": {},
            "turn_lifecycle": {"active_flow": "none", "action": "none"},
        }

        prompt = build_guided_exercise_system_prompt(cast(AgentState, state))
        # ACT overlay should be present (acceptance/defusion content)
        assert "acceptance" in prompt.lower()
        # CBT-specific content should NOT dominate — verify ACT was chosen
        # by checking for ACT-specific language that wouldn't appear in CBT
        assert "defusion" in prompt.lower() or "willingness" in prompt.lower()

    def test_prompt_builder_falls_back_to_routing(self) -> None:
        """When exercise_therapeutic_approach is None, falls back to routing approach."""
        from agent.therapeutic.prompts import build_guided_exercise_system_prompt

        state: dict[str, Any] = {
            "exercise_state": {"exercise_therapeutic_approach": None},
            "therapeutic_approach": "cbt",
            "working_memory": [],
            "procedural_profile": {},
            "turn_lifecycle": {"active_flow": "none", "action": "none"},
        }

        prompt = build_guided_exercise_system_prompt(cast(AgentState, state))
        # CBT overlay should be present
        assert "cbt" in prompt.lower() or "cognitive" in prompt.lower()

    def test_prompt_builder_ignores_stale_approach_without_exercise(self) -> None:
        """When exercise_type is None but exercise_therapeutic_approach is stale,
        the prompt builder ignores the stale therapeutic approach and falls back to routing."""
        from agent.therapeutic.prompts import build_guided_exercise_system_prompt

        state: dict[str, Any] = {
            "exercise_state": {
                "exercise_type": None,
                "exercise_therapeutic_approach": "act",
            },
            "therapeutic_approach": "cbt",
            "working_memory": [],
            "procedural_profile": {},
            "turn_lifecycle": {"active_flow": "none", "action": "none"},
        }

        prompt = build_guided_exercise_system_prompt(cast(AgentState, state))
        # Should use CBT from routing, not stale ACT from exercise_state
        assert "cbt" in prompt.lower() or "cognitive" in prompt.lower()


class TestCompletionResponse:
    """Verify completion uses generated response text."""

    @pytest.mark.asyncio
    async def test_completion_uses_response_llm_text(self) -> None:
        """The completion path returns generated response text."""
        runtime = _MockRuntime(
            llm_client=_StepClassifierLLM(
                step_state="complete",
                response_text="Generated completion response.",
            )
        )
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        text = delta["response_text"]
        assert text == "Generated completion response."
