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

from agent.memory.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

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
        memory_mode: MemoryMode = MemoryMode.INCOGNITO,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=memory_store or OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode,
        )


def _make_state(
    message: str,
    exercise_type: str | None = None,
    exercise_step: int | None = None,
) -> Any:
    """Build a minimal state dict for exercise tests."""

    progress: dict[str, Any] = {"turn_count": 1}
    if exercise_type is not None:
        progress["exercise_type"] = exercise_type
    if exercise_step is not None:
        progress["exercise_step"] = exercise_step

    return {
        "message": message,
        "history": [],
        "progress": progress,
        "response": {},
        "routing": {},
    }


# ── Classifier tests ────────────────────────────────────────────────


class TestConfirmationMode:
    """Tests for _classify_step_state with completion_mode='user_confirmation'."""

    def test_bare_ok_completes(self) -> None:
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

        step = ExerciseStep(
            prompt_fallback="Take a breath.",
            expected_count=1,
            min_count_for_completion=1,
            completion_mode="user_confirmation",
        )
        assert _classify_step_state("I took a breath", step) == "complete"
        assert _classify_step_state("I've done that", step) == "complete"
        assert _classify_step_state("I'm ready", step) == "complete"
        assert _classify_step_state("I exhaled", step) == "complete"

    def test_tentative_holds(self) -> None:
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import ExerciseStep, _classify_step_state

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            _select_exercise,
        )

        assert (
            _select_exercise("can we do a breathing exercise") == EXERCISE_BOX_BREATHING
        )
        assert _select_exercise("I need to breathe") == EXERCISE_BOX_BREATHING
        assert _select_exercise("box breathing please") == EXERCISE_BOX_BREATHING

    def test_thought_record_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_THOUGHT_RECORD,
            _select_exercise,
        )

        assert _select_exercise("let's do a thought record") == EXERCISE_THOUGHT_RECORD
        assert _select_exercise("can we examine this belief") == EXERCISE_THOUGHT_RECORD

    def test_tiny_action_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_TINY_ACTION,
            _select_exercise,
        )

        assert (
            _select_exercise("I'm stuck, can't start anything") == EXERCISE_TINY_ACTION
        )
        assert _select_exercise("I feel depleted") == EXERCISE_TINY_ACTION

    def test_defusion_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_LEAVES_ON_STREAM,
            _select_exercise,
        )

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_STOP_TECHNIQUE,
            _select_exercise,
        )

        assert (
            _select_exercise("let's try the stop technique") == EXERCISE_STOP_TECHNIQUE
        )

    def test_muscle_relaxation_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_MUSCLE_RELAXATION,
            _select_exercise,
        )

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BEHAVIORAL_EXPERIMENT,
            _select_exercise,
        )

        assert (
            _select_exercise("can we test this belief")
            == EXERCISE_BEHAVIORAL_EXPERIMENT
        )
        assert (
            _select_exercise("let's do a behavioral experiment")
            == EXERCISE_BEHAVIORAL_EXPERIMENT
        )

    def test_self_compassion_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_SELF_COMPASSION,
            _select_exercise,
        )

        assert (
            _select_exercise("I need some self-compassion") == EXERCISE_SELF_COMPASSION
        )
        assert _select_exercise("I'm so hard on myself") == EXERCISE_SELF_COMPASSION

    def test_improve_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_IMPROVE,
            _select_exercise,
        )

        assert _select_exercise("help me get through this") == EXERCISE_IMPROVE
        assert (
            _select_exercise("I'm overwhelmed, everything is too much")
            == EXERCISE_IMPROVE
        )

    def test_values_compass_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_VALUES_COMPASS,
            _select_exercise,
        )

        assert _select_exercise("what matters to me") == EXERCISE_VALUES_COMPASS
        assert (
            _select_exercise("I feel like I've lost my purpose")
            == EXERCISE_VALUES_COMPASS
        )

    def test_gratitude_keywords(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_GRATITUDE,
            _select_exercise,
        )

        assert _select_exercise("can we do a gratitude exercise") == EXERCISE_GRATITUDE
        assert (
            _select_exercise("I want to think about something positive")
            == EXERCISE_GRATITUDE
        )

    def test_default_is_grounding(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_5_4_3_2_1,
            _select_exercise,
        )

        assert _select_exercise("help me calm down") == EXERCISE_5_4_3_2_1
        assert _select_exercise("ground me") == EXERCISE_5_4_3_2_1
        assert _select_exercise("I need an exercise") == EXERCISE_5_4_3_2_1

    def test_work_through_that_uses_recent_cognitive_context(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_THOUGHT_RECORD,
            _select_exercise,
        )

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
    async def test_start_selects_box_breathing(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("can we do a breathing exercise")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] == EXERCISE_BOX_BREATHING
        assert delta["progress"]["exercise_step"] == 0
        assert "breathe in" in delta["response"]["text"].lower()

    @pytest.mark.asyncio
    async def test_confirmation_advances_box_breathing(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_step"] == 1
        assert "hold" in delta["response"]["text"].lower()

    @pytest.mark.asyncio
    async def test_box_breathing_completion(self) -> None:
        """Completing the last step clears exercise state."""
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        # Step 3 is the last step (0-indexed, 4 steps total)
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] is None
        assert delta["progress"]["exercise_step"] is None
        assert "box breathing" in delta["response"]["text"].lower()


class TestThoughtRecordFlow:
    """End-to-end flow for simple thought record."""

    @pytest.mark.asyncio
    async def test_start_selects_thought_record(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_THOUGHT_RECORD,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("let's do a thought record")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] == EXERCISE_THOUGHT_RECORD
        assert delta["progress"]["exercise_step"] == 0
        assert "situation" in delta["response"]["text"].lower()

    @pytest.mark.asyncio
    async def test_thought_record_advances_on_description(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_THOUGHT_RECORD,
            run_guided_exercise_response_node,
        )

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

        assert delta["progress"]["exercise_step"] == 1


class TestMuscleRelaxationFlow:
    """End-to-end flow for progressive muscle relaxation."""

    @pytest.mark.asyncio
    async def test_start_selects_pmr(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_MUSCLE_RELAXATION,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I need to release some tension in my body")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] == EXERCISE_MUSCLE_RELAXATION
        assert delta["progress"]["exercise_step"] == 0
        assert (
            "hands" in delta["response"]["text"].lower()
            or "fist" in delta["response"]["text"].lower()
        )

    @pytest.mark.asyncio
    async def test_pmr_advances_on_confirmation(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_MUSCLE_RELAXATION,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("done", EXERCISE_MUSCLE_RELAXATION, 0)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_step"] == 1
        assert "shoulder" in delta["response"]["text"].lower()


class TestSelfCompassionFlow:
    """End-to-end flow for self-compassion break."""

    @pytest.mark.asyncio
    async def test_start_selects_self_compassion(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_SELF_COMPASSION,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I'm so hard on myself all the time")

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] == EXERCISE_SELF_COMPASSION
        assert delta["progress"]["exercise_step"] == 0

    @pytest.mark.asyncio
    async def test_self_compassion_completes_in_3_steps(self) -> None:
        """Self-compassion break has 3 steps; completing step 2 should clear state."""
        from agent.therapeutic.guided_exercise import (
            EXERCISE_SELF_COMPASSION,
            run_guided_exercise_response_node,
        )

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

        assert delta["progress"]["exercise_type"] is None
        assert delta["progress"]["exercise_step"] is None


class TestExitMidExercise:
    """Test exit from confirmation-based exercises."""

    @pytest.mark.asyncio
    async def test_exit_box_breathing_clears_state(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state("I don't want to do this", EXERCISE_BOX_BREATHING, 1)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] is None
        assert delta["progress"]["exercise_step"] is None
        assert (
            "stop" in delta["response"]["text"].lower()
            or "helpful" in delta["response"]["text"].lower()
        )

    @pytest.mark.asyncio
    async def test_exit_leaves_on_stream_clears_state(self) -> None:
        from agent.therapeutic.guided_exercise import (
            EXERCISE_LEAVES_ON_STREAM,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None)
        state = _make_state(
            "never mind, can we just talk", EXERCISE_LEAVES_ON_STREAM, 2
        )

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        assert delta["progress"]["exercise_type"] is None
        assert delta["progress"]["exercise_step"] is None


class TestRegistryCompleteness:
    """Verify the registry and display names cover all exercises."""

    def test_all_exercises_registered(self) -> None:
        from agent.therapeutic.guided_exercise import (
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
            _EXERCISE_REGISTRY,
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
        }
        assert set(_EXERCISE_REGISTRY.keys()) == expected
        assert len(_EXERCISE_REGISTRY) == 12

    def test_all_exercises_have_display_names(self) -> None:
        from agent.therapeutic.guided_exercise import (
            _EXERCISE_DISPLAY_NAMES,
            _EXERCISE_REGISTRY,
        )

        for key in _EXERCISE_REGISTRY:
            assert key in _EXERCISE_DISPLAY_NAMES, f"Missing display name for {key}"

    def test_all_steps_have_fallback_text(self) -> None:
        from agent.therapeutic.guided_exercise import _EXERCISE_REGISTRY

        for exercise_type, steps in _EXERCISE_REGISTRY.items():
            for i, step in enumerate(steps):
                assert step.prompt_fallback, (
                    f"Empty fallback for {exercise_type} step {i}"
                )


# ── Memory write tests ───────────────────────────────────────────────


class TestExerciseCompletionMemory:
    """Tests that exercise completions write semantic facts to memory."""

    @pytest.mark.asyncio
    async def test_completion_writes_coping_strategy_fact(self) -> None:
        """Completing an exercise in persistent mode writes a semantic fact."""
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

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
        from agent.therapeutic.guided_exercise import (
            EXERCISE_BOX_BREATHING,
            run_guided_exercise_response_node,
        )

        runtime = _MockRuntime(llm_client=None, memory_store=None, memory_mode="local")
        state = _make_state("done", EXERCISE_BOX_BREATHING, 3)

        delta = await run_guided_exercise_response_node(
            cast(AgentState, state),
            runtime,  # type: ignore[arg-type]
        )

        # Should complete normally without error
        assert delta["progress"]["exercise_type"] is None
