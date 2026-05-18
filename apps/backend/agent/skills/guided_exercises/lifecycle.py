"""Guided-exercise skill lifecycle service."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.runtime.session.state import current_turn_lifecycle
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.state import AgentState
from agent.skills.guided_exercises.responses import (
    StreamWriterFactory,
    _build_advance_delta,
    _build_complete_delta,
    _build_exit_delta,
    _build_hold_delta,
    _build_resume_delta,
    _build_start_delta,
    _build_stuck_delta,
)
from agent.skills.guided_exercises.selection import _select_exercise_llm_primary
from agent.skills.guided_exercises.state import (
    _get_current_step,
    _is_last_step,
    clear_exercise_delta,
)
from agent.skills.guided_exercises.step_classifier import classify_step_state
from agent.skills.guided_exercises.types import ExerciseStep
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


def _noop_stream_writer_factory() -> Callable[[dict[str, str]], None]:
    """Return a no-op stream writer for direct service tests."""

    return lambda _payload: None


@dataclass(frozen=True)
class GuidedExerciseSkillService:
    """Run one guided-exercise skill turn without depending on agent orchestration."""

    classifier_llm: BaseLLMClient | None
    response_llm: BaseLLMClient | None
    memory_store: MemoryStore | None
    memory_mode: MemoryMode
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory

    async def run_turn(self, state: AgentState) -> dict[str, Any]:
        """Run one guided-exercise skill turn.

        Args:
            state: Current runtime state.

        Returns:
            Response and state delta for the exercise turn.
        """

        exercise_state = state.get("exercise_state", {})
        exercise_type = exercise_state.get("exercise_type")
        step_index = exercise_state.get("exercise_step")

        if exercise_type is None or step_index is None:
            return await self._handle_start(state)

        current_step = _get_current_step(exercise_type, step_index)
        if current_step is None:
            logger.warning(
                "GuidedExerciseSkillService: invalid exercise state exercise_type=%r "
                "step_index=%r; clearing and restarting",
                exercise_type,
                step_index,
            )
            cleared = clear_exercise_delta(state)
            start_delta = await self._handle_start(state)
            return {**cleared, **start_delta}

        return await self._handle_continue(
            state=state,
            exercise_type=exercise_type,
            step_index=step_index,
            current_step=current_step,
        )

    async def _handle_start(self, state: AgentState) -> dict[str, Any]:
        """Start a new exercise at step 0."""

        selected = await _select_exercise_llm_primary(
            state,
            classifier_llm=self.classifier_llm,
        )
        return await _build_start_delta(
            state,
            llm_client=self.response_llm,
            exercise_type=selected,
            stream_writer_factory=self.stream_writer_factory,
        )

    async def _handle_continue(
        self,
        *,
        state: AgentState,
        exercise_type: str,
        step_index: int,
        current_step: ExerciseStep,
    ) -> dict[str, Any]:
        """Continue an exercise based on the user's current message."""

        active_flow = current_turn_lifecycle(state)
        if (
            active_flow.active_flow == "guided_exercise"
            and active_flow.action == "resume"
        ):
            return await _build_resume_delta(
                state,
                llm_client=self.response_llm,
                stream_writer_factory=self.stream_writer_factory,
            )

        step_state = await classify_step_state(
            state=state,
            classifier_llm=self.classifier_llm,
            exercise_type=exercise_type,
            step_index=step_index,
            current_step=current_step,
        )

        logger.debug(
            "guided_exercise continue: exercise_type=%s step_index=%d step_state=%s",
            exercise_type,
            step_index,
            step_state,
        )

        if step_state == "exit":
            return await _build_exit_delta(
                state,
                llm_client=self.response_llm,
                stream_writer_factory=self.stream_writer_factory,
            )

        if step_state == "stuck":
            return await _build_stuck_delta(
                state,
                llm_client=self.response_llm,
                stream_writer_factory=self.stream_writer_factory,
            )

        if step_state == "hold":
            return await _build_hold_delta(
                state,
                llm_client=self.response_llm,
                stream_writer_factory=self.stream_writer_factory,
            )

        if _is_last_step(exercise_type, step_index):
            return await _build_complete_delta(
                state,
                llm_client=self.response_llm,
                memory_store=self.memory_store,
                memory_mode=self.memory_mode,
                stream_writer_factory=self.stream_writer_factory,
            )

        return await _build_advance_delta(
            state=state,
            llm_client=self.response_llm,
            exercise_type=exercise_type,
            next_step_index=step_index + 1,
            stream_writer_factory=self.stream_writer_factory,
        )


__all__ = ["GuidedExerciseSkillService"]
