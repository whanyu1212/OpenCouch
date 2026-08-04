"""Drift tests for shared therapeutic policy labels."""

from __future__ import annotations

from typing import get_args

from agent.models import (
    GuidancePermission,
    SessionIntent,
    SessionStage,
    TherapeuticApproach,
)
from agent.runtime.dispatch_models import DispatchDecision


def test_runtime_dispatch_uses_core_policy_labels() -> None:
    """Runtime dispatch models must use the canonical policy labels."""

    fields = DispatchDecision.model_fields
    assert fields["therapeutic_approach"].annotation == TherapeuticApproach
    assert fields["session_intent"].annotation == SessionIntent
    assert fields["session_stage"].annotation == SessionStage
    assert fields["guidance_permission"].annotation == GuidancePermission
    assert "repair" in get_args(SessionIntent)
