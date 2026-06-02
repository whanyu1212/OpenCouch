"""Guided-exercise loadout projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.state import AgentState
from agent.skills.guided_exercises.registry import available_exercise_definitions


@dataclass(frozen=True)
class GuidedExerciseLoadout:
    """Exercise availability projected from the current runtime state."""

    available_exercise_ids: tuple[str, ...]
    selected_exercise_id: str | None
    channel: str
    therapeutic_approach: str | None
    installed_skills: tuple[str, ...]


def build_guided_exercise_loadout(
    state: AgentState,
    *,
    selected_exercise_id: str | None = None,
) -> GuidedExerciseLoadout:
    """Build a guided-exercise loadout without changing selection behavior."""

    installed_skills = tuple(
        str(skill) for skill in state.get("installed_skills") or ()
    )
    channel = _channel_value(state.get("channel", "text"))
    therapeutic_approach = _therapeutic_approach_value(
        state.get("therapeutic_approach")
    )
    available_ids = tuple(
        definition.id
        for definition in available_exercise_definitions(
            installed_skills=installed_skills,
            channel=channel,
            therapeutic_approach=therapeutic_approach,
        )
    )

    return GuidedExerciseLoadout(
        available_exercise_ids=available_ids,
        selected_exercise_id=selected_exercise_id,
        channel=channel,
        therapeutic_approach=therapeutic_approach,
        installed_skills=installed_skills,
    )


def _channel_value(value: Any) -> str:
    raw_value = getattr(value, "value", value)
    channel = str(raw_value or "text").strip()
    return channel or "text"


def _therapeutic_approach_value(value: Any) -> str | None:
    if value is None:
        return None
    approach = str(value).strip()
    if not approach or approach == "none":
        return None
    return approach


__all__ = ["GuidedExerciseLoadout", "build_guided_exercise_loadout"]
