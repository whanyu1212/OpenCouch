"""Drift tests for shared therapeutic policy labels."""

from __future__ import annotations

from typing import get_args

from agent.memory.models import (
    GuidancePermission as TextGuidancePermission,
    SessionIntent as TextSessionIntent,
    TherapeuticApproach as TextTherapeuticApproach,
)
from agent.therapeutic_policy import (
    GuidancePermission,
    SessionIntent,
    TherapeuticApproach,
)
from agent.voice.session_data import (
    GuidancePermission as VoiceGuidancePermission,
    SessionIntent as VoiceSessionIntent,
    TherapeuticApproach as VoiceTherapeuticApproach,
)


def test_text_and_voice_share_core_policy_labels() -> None:
    """Text and voice should not drift on shared therapeutic policy labels."""

    assert TextSessionIntent == SessionIntent
    assert VoiceSessionIntent == SessionIntent
    assert TextGuidancePermission == GuidancePermission
    assert VoiceGuidancePermission == GuidancePermission
    assert TextTherapeuticApproach == TherapeuticApproach
    assert VoiceTherapeuticApproach == TherapeuticApproach
    assert "repair" in get_args(SessionIntent)
