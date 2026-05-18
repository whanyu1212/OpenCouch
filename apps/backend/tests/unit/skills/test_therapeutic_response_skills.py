"""Tests for therapeutic response style skills."""

from __future__ import annotations

import pytest

from agent.skills.therapeutic_response import (
    THERAPEUTIC_RESPONSE_SKILL_STYLES,
    render_therapeutic_response_skill_context,
)


def _state() -> dict:
    return {
        "message": "I feel tense today",
        "history": [],
        "transcript": [],
        "working_memory": [],
        "session_memory": {},
        "procedural_profile": {},
        "turn_lifecycle": {"active_flow": "none", "action": "none"},
        "memory_reference": {"mode": "none"},
        "session_progress": {"turn_count": 1},
        "response_guidance": "",
    }


@pytest.mark.parametrize("style", THERAPEUTIC_RESPONSE_SKILL_STYLES)
def test_response_style_skill_renders_bounded_context(style: str) -> None:
    rendered = render_therapeutic_response_skill_context(
        _state(),
        response_style=style,
        therapeutic_approach="cbt",
    )

    assert rendered.startswith("Therapeutic response skill:")
    assert f"- skill_id: therapeutic_response/{style}" in rendered
    assert f"- response_style: {style}" in rendered
    assert "- therapeutic_approach: cbt" in rendered
    assert "- side_effect: none" in rendered
    assert "- retry_safe: true" in rendered
    assert "Operating boundaries:" in rendered
    assert "Skill guidance:" in rendered


def test_unknown_response_style_falls_back_to_supportive() -> None:
    rendered = render_therapeutic_response_skill_context(
        _state(),
        response_style="unknown",
        therapeutic_approach=None,
    )

    assert "- skill_id: therapeutic_response/supportive" in rendered
    assert "- response_style: supportive" in rendered
    assert "- therapeutic_approach: none" in rendered
