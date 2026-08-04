"""App-owned lifecycle service for guided exercises."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.skills.guided_exercises.lifecycle.service import (
        GuidedExerciseSkillService,
    )


def __getattr__(name: str) -> object:
    """Lazily expose lifecycle services without importing presentation code."""

    if name == "GuidedExerciseSkillService":
        from agent.skills.guided_exercises.lifecycle.service import (
            GuidedExerciseSkillService,
        )

        return GuidedExerciseSkillService
    raise AttributeError(name)


__all__ = ["GuidedExerciseSkillService"]
