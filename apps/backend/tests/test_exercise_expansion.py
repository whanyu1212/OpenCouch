"""Tests for the exercise expansion — new exercises, confirmation mode, selection logic.

Tests cover:
1. ExerciseStep completion_mode field
2. _classify_step_state with user_confirmation mode
3. _select_exercise keyword-based selection
4. End-to-end flows for box breathing and thought record
5. Exit mid-exercise for confirmation-based exercises
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
from agent.therapeutic.exercises.selection import _select_exercise
from agent.therapeutic.exercises.step_classifier import _classify_step_state
from agent.therapeutic.exercises.types import ExerciseStep
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node

# ── Helper ────────────────────────────────────────────────────────────


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
        selection_kind: str = "selected",
        exercise_type: str | None = None,
        option_types: list[str] | None = None,
        response_text: str = "next step",
        fail_selection: bool = False,
    ) -> None:
        self.step_state = step_state
        self.selection_kind = selection_kind
        self.exercise_type = exercise_type
        self.option_types = option_types or []
        self.response_text = response_text
        self.fail_selection = fail_selection
        self.structured_calls = 0

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: Any,
        system_instruction: str | None = None,
    ) -> Any:
        self.structured_calls += 1
        if response_schema.__name__ == "ExerciseOptionChoiceDecision":
            return response_schema(
                choice_kind="selected" if self.exercise_type else "unclear",
                exercise_type=self.exercise_type,
                reasoning="fake pending option choice",
                confidence="high",
            )
        if response_schema.__name__ == "ExerciseSelectionDecision":
            if self.fail_selection:
                raise RuntimeError("selection failure")
            return response_schema(
                selection_kind=self.selection_kind,
                exercise_type=self.exercise_type,
                option_types=self.option_types,
                reasoning="fake exercise selection",
                confidence="high",
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
    }


# ── Classifier tests ────────────────────────────────────────────────


class TestConfirmationMode:
    """Tests for _classify_step_state with completion_mode='user_confirmation'."""

    def test_bare_ok_completes(self) -> None:
        step = ExerciseStep(
            prompt_fallback="Breathe in. Let me know when done.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("ok", step) == "complete"
        assert _classify_step_state("done", step) == "complete"
        assert _classify_step_state("yes", step) == "complete"
        assert _classify_step_state("yep", step) == "complete"
        assert _classify_step_state("ready", step) == "complete"
        assert _classify_step_state("Done.", step) == "complete"

    def test_action_phrases_complete(self) -> None:
        step = ExerciseStep(
            prompt_fallback="Take a breath.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("I took a breath", step) == "complete"
        assert _classify_step_state("I've done that", step) == "complete"
        assert _classify_step_state("done that", step) == "complete"
        assert _classify_step_state("did that", step) == "complete"
        assert _classify_step_state("I'm ready", step) == "complete"
        assert _classify_step_state("I exhaled", step) == "complete"

    def test_tentative_holds(self) -> None:
        step = ExerciseStep(
            prompt_fallback="Breathe in.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("hmm", step) == "hold"
        assert _classify_step_state("I'm trying", step) == "hold"
        assert _classify_step_state("not sure", step) == "hold"

    def test_stuck_overrides_confirmation(self) -> None:
        """STUCK patterns take priority over confirmation."""
        step = ExerciseStep(
            prompt_fallback="Breathe in.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("I can't do this", step) == "stuck"
        assert _classify_step_state("this is stupid", step) == "stuck"

    def test_exit_overrides_confirmation(self) -> None:
        """EXIT patterns take priority over everything."""
        step = ExerciseStep(
            prompt_fallback="Breathe in.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("stop", step) == "exit"
        assert _classify_step_state("can we just talk", step) == "exit"
        assert _classify_step_state("this isn't helping", step) == "exit"


class TestItemCountModeBackwardCompat:
    """Verify existing item_count mode still works after the extension."""

    def test_item_count_still_works(self) -> None:
        step = ExerciseStep(
            prompt_fallback="Name 5 things you can see.",
            expected_count=5,
            min_count_for_completion=3,
        )
        # 3 items → complete (meets min_count)
        assert _classify_step_state("a lamp, a book, and my coffee", step) == "complete"
        # 1 item → hold
        assert _classify_step_state("a lamp", step) == "hold"

    def test_confirmation_words_dont_complete_item_count_step(self) -> None:
        """Saying 'ok' on an item_count step should hold, not complete."""
        step = ExerciseStep(
            prompt_fallback="Name 5 things.",
            expected_count=5,
            min_count_for_completion=3,
        )
        assert _classify_step_state("ok", step) == "hold"
        assert _classify_step_state("done", step) == "hold"


# ── Exercise selection tests ─────────────────────────────────────────


class TestExerciseSelection:
    """Tests for _select_exercise keyword matching."""

    def test_breathing_keywords(self) -> None:
        assert (
            _select_exercise("can we do a breathing exercise") == EXERCISE_BOX_BREATHING
        )
        assert _select_exercise("I need to breathe") == EXERCISE_BOX_BREATHING
        assert _select_exercise("box breathing please") == EXERCISE_BOX_BREATHING

    def test_thought_record_keywords(self) -> None:
        assert _select_exercise("let's do a thought record") == EXERCISE_THOUGHT_RECORD
        assert _select_exercise("can we examine this belief") == EXERCISE_THOUGHT_RECORD

    def test_tiny_action_keywords(self) -> None:
        assert (
            _select_exercise("I'm stuck, can't start anything") == EXERCISE_TINY_ACTION
        )
        assert _select_exercise("I feel depleted") == EXERCISE_TINY_ACTION

    def test_defusion_keywords(self) -> None:
        assert (
            _select_exercise("I need to let go of this thought")
            == EXERCISE_LEAVES_ON_STREAM
        )
        assert (
            _select_exercise("can we try the leaves exercise")
            == EXERCISE_LEAVES_ON_STREAM
        )
        assert (
            _select_exercise("I want to stop fighting this feeling")
            == EXERCISE_LEAVES_ON_STREAM
        )

    def test_stop_technique_keywords(self) -> None:
        assert (
            _select_exercise("let's try the stop technique") == EXERCISE_STOP_TECHNIQUE
        )

    def test_muscle_relaxation_keywords(self) -> None:
        assert (
            _select_exercise("I need to release some tension")
            == EXERCISE_MUSCLE_RELAXATION
        )
        assert (
            _select_exercise("can we do progressive muscle relaxation")
            == EXERCISE_MUSCLE_RELAXATION
        )
        assert _select_exercise("relax my body") == EXERCISE_MUSCLE_RELAXATION

    def test_behavioral_experiment_keywords(self) -> None:
        assert (
            _select_exercise("can we test this belief")
            == EXERCISE_BEHAVIORAL_EXPERIMENT
        )
        assert (
            _select_exercise("let's do a behavioral experiment")
            == EXERCISE_BEHAVIORAL_EXPERIMENT
        )

    def test_self_compassion_keywords(self) -> None:
        assert (
            _select_exercise("I need some self-compassion") == EXERCISE_SELF_COMPASSION
        )
        assert _select_exercise("I'm so hard on myself") == EXERCISE_SELF_COMPASSION

    def test_improve_keywords(self) -> None:
        assert _select_exercise("help me get through this") == EXERCISE_IMPROVE
        assert (
            _select_exercise("I'm overwhelmed, everything is too much")
            == EXERCISE_IMPROVE
        )

    def test_values_compass_keywords(self) -> None:
        assert _select_exercise("what matters to me") == EXERCISE_VALUES_COMPASS
        assert (
            _select_exercise("I feel like I've lost my purpose")
            == EXERCISE_VALUES_COMPASS
        )

    def test_gratitude_keywords(self) -> None:
        assert _select_exercise("can we do a gratitude exercise") == EXERCISE_GRATITUDE
        assert (
            _select_exercise("I want to think about something positive")
            == EXERCISE_GRATITUDE
        )

    def test_grounding_keywords_and_no_match(self) -> None:
        assert _select_exercise("ground me") == EXERCISE_5_4_3_2_1
        assert _select_exercise("5-4-3-2-1 please") == EXERCISE_5_4_3_2_1
        assert _select_exercise("help me calm down") is None
        assert _select_exercise("I need an exercise") is None

    def test_work_through_that_uses_recent_cognitive_context(self) -> None:
        history = [
            {
                "role": "user",
                "content": "I always assume one mistake means everyone will see I'm incompetent.",
            }
        ]

        assert (
            _select_exercise("Yeah, can we work through that?", history=history)
            == EXERCISE_THOUGHT_RECORD
        )


# ── End-to-end node tests ────────────────────────────────────────────


class TestBoxBreathingFlow:
    """End-to-end flow for box breathing exercise."""

    @pytest.mark.asyncio
    async def test_no_llm_start_offers_fallback_suggestions_without_match(
        self,
    ) -> None:
        from agent.therapeutic.exercises.registry import fallback_suggestion_options

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I need an exercise")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        assert delta["exercise_state"]["exercise_selection_options"] == list(
            fallback_suggestion_options()
        )
        assert "which would you like" in delta["response_text"].lower()
        assert "five things" not in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_start_selects_box_breathing(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("can we do a breathing exercise")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_BOX_BREATHING
        assert delta["exercise_state"]["exercise_step"] == 0
        assert "breathe in" in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_confirmation_advances_box_breathing(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_step"] == 1
        assert "hold" in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_box_breathing_completion(self) -> None:
        """Completing the last step clears exercise state."""
        runtime = _MockRuntime(llm_client=None)
        # Step 3 is the last step (0-indexed, 4 steps total)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        assert "box breathing" in delta["response_text"].lower()


class TestThoughtRecordFlow:
    """End-to-end flow for simple thought record."""

    @pytest.mark.asyncio
    async def test_start_selects_thought_record(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("let's do a thought record")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_THOUGHT_RECORD
        assert delta["exercise_state"]["exercise_step"] == 0
        assert "situation" in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_thought_record_advances_on_description(self) -> None:
        runtime = _MockRuntime(llm_client=None)
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
    async def test_start_selects_pmr(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I need to release some tension in my body")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_MUSCLE_RELAXATION
        assert delta["exercise_state"]["exercise_step"] == 0
        assert (
            "hands" in delta["response_text"].lower()
            or "fist" in delta["response_text"].lower()
        )

    @pytest.mark.asyncio
    async def test_pmr_advances_on_confirmation(self) -> None:
        runtime = _MockRuntime(llm_client=None)
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
    async def test_start_selects_self_compassion(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I'm so hard on myself all the time")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_SELF_COMPASSION
        assert delta["exercise_state"]["exercise_step"] == 0

    @pytest.mark.asyncio
    async def test_llm_selection_starts_self_compassion_without_keyword(self) -> None:
        llm = _StepClassifierLLM(
            selection_kind="selected",
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
        assert delta["exercise_state"]["exercise_selection_options"] is None

    @pytest.mark.asyncio
    async def test_ambiguous_selection_offers_options(self) -> None:
        llm = _StepClassifierLLM(
            selection_kind="ambiguous",
            option_types=[EXERCISE_BOX_BREATHING, EXERCISE_SELF_COMPASSION],
        )
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("Can we do something for this?")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None
        assert delta["exercise_state"]["exercise_selection_options"] == [
            EXERCISE_BOX_BREATHING,
            EXERCISE_SELF_COMPASSION,
        ]
        assert "which would you like" in delta["response_text"].lower()
        assert "five things" not in delta["response_text"].lower()

    @pytest.mark.asyncio
    async def test_llm_selection_failure_offers_fallback_suggestions(self) -> None:
        from agent.therapeutic.exercises.registry import fallback_suggestion_options

        llm = _StepClassifierLLM(fail_selection=True)
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("Can we do an exercise?")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_selection_options"] == list(
            fallback_suggestion_options()
        )

    @pytest.mark.asyncio
    async def test_pending_selection_number_starts_option(self) -> None:
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("2")
        state["exercise_state"]["exercise_selection_options"] = [
            EXERCISE_BOX_BREATHING,
            EXERCISE_SELF_COMPASSION,
        ]

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_type"] == EXERCISE_SELF_COMPASSION
        assert delta["exercise_state"]["exercise_step"] == 0
        assert delta["exercise_state"]["exercise_selection_options"] is None

    @pytest.mark.asyncio
    async def test_pending_selection_llm_resolves_natural_choice(self) -> None:
        llm = _StepClassifierLLM(exercise_type=EXERCISE_SELF_COMPASSION)
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("self-kindness option")
        state["exercise_state"]["exercise_selection_options"] = [
            EXERCISE_BOX_BREATHING,
            EXERCISE_SELF_COMPASSION,
        ]

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 1
        assert delta["exercise_state"]["exercise_type"] == EXERCISE_SELF_COMPASSION
        assert delta["exercise_state"]["exercise_step"] == 0
        assert delta["exercise_state"]["exercise_selection_options"] is None

    @pytest.mark.asyncio
    async def test_self_compassion_completes_in_3_steps(self) -> None:
        """Self-compassion break has 3 steps; completing step 2 should clear state."""
        runtime = _MockRuntime(llm_client=None)
        # Step 2 is the last step (0-indexed, 3 steps total)
        # Step 2 uses item_count mode — user says their kind wish
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
    async def test_clear_item_count_completion_bypasses_llm_hold(self) -> None:
        """Clear sensory item lists should advance even if the LLM would hold."""

        llm = _StepClassifierLLM(
            step_state="hold",
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

        assert llm.structured_calls == 0
        assert delta["exercise_state"]["exercise_step"] == 1
        assert "hear" in delta["response_text"].lower()


class TestExitMidExercise:
    """Test exit from confirmation-based exercises."""

    @pytest.mark.asyncio
    async def test_exit_box_breathing_clears_state(self) -> None:
        runtime = _MockRuntime(llm_client=None)
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
    async def test_explicit_exit_bypasses_llm_classifier(self) -> None:
        llm = _StepClassifierLLM(step_state="complete")
        runtime = _MockRuntime(llm_client=llm)
        state = _make_state("stop, I want to just talk", EXERCISE_SELF_COMPASSION, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert llm.structured_calls == 0
        assert delta["exercise_state"]["exercise_type"] is None
        assert delta["exercise_state"]["exercise_step"] is None

    @pytest.mark.asyncio
    async def test_exit_leaves_on_stream_clears_state(self) -> None:
        runtime = _MockRuntime(llm_client=None)
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
            EXERCISE_BEHAVIORAL_EXPERIMENT,
            EXERCISE_BOX_BREATHING,
            EXERCISE_GRATITUDE,
            EXERCISE_IMPROVE,
            EXERCISE_LEAVES_ON_STREAM,
            EXERCISE_MUSCLE_RELAXATION,
            EXERCISE_SELF_COMPASSION,
            EXERCISE_STOP_TECHNIQUE,
            EXERCISE_THOUGHT_RECORD,
            EXERCISE_TINY_ACTION,
            EXERCISE_VALUES_COMPASS,
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

    def test_all_steps_have_fallback_text(self) -> None:
        from agent.therapeutic.exercises.registry import iter_exercise_definitions

        for definition in iter_exercise_definitions():
            for i, step in enumerate(definition.steps):
                assert step.prompt_fallback, (
                    f"Empty fallback for {definition.id} step {i}"
                )

    def test_catalog_public_helpers_match_definitions(self) -> None:
        from agent.therapeutic.exercises.registry import (
            fallback_suggestion_options,
            get_exercise_definition,
            get_exercise_display_name,
            get_exercise_steps,
            is_valid_exercise_type,
            iter_exercise_definitions,
            iter_exercise_selection_aliases,
            iter_exercise_selectors,
            voice_exercise_ids,
        )

        definitions = iter_exercise_definitions()
        ids = [definition.id for definition in definitions]
        assert len(ids) == len(set(ids))
        registered = set(ids)

        for definition in definitions:
            assert is_valid_exercise_type(definition.id)
            assert get_exercise_definition(definition.id) == definition
            assert get_exercise_steps(definition.id) == definition.steps
            assert get_exercise_display_name(definition.id) == definition.display_name
            assert definition.selection_use_case
            assert definition.selection_aliases
            assert all(alias.strip() for alias in definition.selection_aliases)
            assert definition.steps

        assert not is_valid_exercise_type("not_registered")
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

        fallback_options = fallback_suggestion_options()
        assert len(fallback_options) >= 2
        assert len(fallback_options) == len(set(fallback_options))
        assert set(fallback_options).issubset(registered)

        ranked_fallbacks = [
            definition.id
            for definition in sorted(
                (
                    definition
                    for definition in definitions
                    if definition.fallback_suggestion_rank is not None
                ),
                key=lambda definition: definition.fallback_suggestion_rank or 0,
            )
        ]
        assert fallback_suggestion_options(limit=len(ranked_fallbacks)) == tuple(
            ranked_fallbacks
        )
        assert fallback_suggestion_options(limit=0) == ()

        selector_targets = {
            exercise_type for _, exercise_type in iter_exercise_selectors()
        }
        assert selector_targets
        assert selector_targets.issubset(registered)

        alias_targets = {
            exercise_type for _, exercise_type in iter_exercise_selection_aliases()
        }
        assert alias_targets == registered

        voice_ids = set(voice_exercise_ids())
        expected_voice_ids = {
            definition.id for definition in definitions if definition.voice_supported
        }
        assert voice_ids == expected_voice_ids


# ── Memory write tests ───────────────────────────────────────────────


class TestExerciseCompletionMemory:
    """Tests that exercise completions write semantic facts to memory."""

    @pytest.mark.asyncio
    async def test_completion_writes_coping_strategy_fact(self) -> None:
        """Completing an exercise in persistent mode writes a semantic fact."""
        store = _RecordingMemoryStore()
        runtime = _MockRuntime(
            llm_client=None,
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
    async def test_exit_does_not_write_fact(self) -> None:
        """Exiting an exercise does NOT write a memory fact."""
        store = _RecordingMemoryStore()
        runtime = _MockRuntime(
            llm_client=None,
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
            llm_client=None,
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
        runtime = _MockRuntime(llm_client=None, memory_store=None, memory_mode="local")
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
        """Starting an exercise stores routing.therapeutic_approach in exercise_state."""
        runtime = _MockRuntime(llm_client=None)
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
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("can we do a breathing exercise")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["exercise_state"]["exercise_therapeutic_approach"] is None

    @pytest.mark.asyncio
    async def test_completion_clears_approach(self) -> None:
        """Completing the last step clears exercise_therapeutic_approach."""
        runtime = _MockRuntime(llm_client=None)
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
        runtime = _MockRuntime(llm_client=None)
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
        }

        prompt = build_guided_exercise_system_prompt(cast(AgentState, state))
        # Should use CBT from routing, not stale ACT from exercise_state
        assert "cbt" in prompt.lower() or "cognitive" in prompt.lower()


class TestCompletionCheckIn:
    """Verify the completion fallback includes the check-in question."""

    @pytest.mark.asyncio
    async def test_completion_fallback_asks_how_it_felt(self) -> None:
        """The deterministic completion text includes a check-in question."""
        runtime = _MockRuntime(llm_client=None)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        text = delta["response_text"]
        assert "how was that for you" in text.lower()
