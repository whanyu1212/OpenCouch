"""Drift tests for shared therapeutic policy labels."""

from __future__ import annotations

from typing import get_args

from agent.memory.types import (
    GuidancePermission as TextGuidancePermission,
    SessionIntent as TextSessionIntent,
    TherapeuticApproach as TextTherapeuticApproach,
)
from agent.therapeutic_policy import (
    GuidancePermission,
    SessionIntent,
    TherapeuticApproach,
)


def test_text_runtime_uses_core_policy_labels() -> None:
    """Text memory models should not drift from core policy labels."""

    assert TextSessionIntent == SessionIntent
    assert TextGuidancePermission == GuidancePermission
    assert TextTherapeuticApproach == TherapeuticApproach
    assert "repair" in get_args(SessionIntent)
