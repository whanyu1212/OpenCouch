"""Tests for therapeutic response style guidance rendering."""

from __future__ import annotations

import pytest

from agent.specialists.therapeutic_response.style_guidance import (
    THERAPEUTIC_RESPONSE_STYLE_GUIDANCE_STYLES,
    render_therapeutic_response_skill_context,
    render_therapeutic_response_style_guidance,
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


@pytest.mark.parametrize("style", THERAPEUTIC_RESPONSE_STYLE_GUIDANCE_STYLES)
def test_response_style_skill_context_renders_bounded_context(style: str) -> None:
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


@pytest.mark.parametrize("style", THERAPEUTIC_RESPONSE_STYLE_GUIDANCE_STYLES)
def test_response_style_guidance_renders_without_tool_metadata(style: str) -> None:
    rendered = render_therapeutic_response_style_guidance(
        _state(),
        response_style=style,
        therapeutic_approach="act",
    )

    assert rendered.startswith("Therapeutic response guidance:")
    assert f"- response_style: {style}" in rendered
    assert "- therapeutic_approach: act" in rendered
    assert "Style guidance:" in rendered
    assert "skill_id" not in rendered
    assert "side_effect" not in rendered
    assert "retry_safe" not in rendered


def test_response_style_guidance_includes_optional_appendix_once() -> None:
    appendix = "TUI-only command guidance"
    rendered = render_therapeutic_response_style_guidance(
        _state(),
        response_style="supportive",
        therapeutic_approach="cbt",
        prompt_appendix=appendix,
    )

    assert rendered.count(appendix) == 1


def test_unknown_response_style_falls_back_to_supportive() -> None:
    rendered = render_therapeutic_response_skill_context(
        _state(),
        response_style="unknown",
        therapeutic_approach=None,
    )

    assert "- skill_id: therapeutic_response/supportive" in rendered
    assert "- response_style: supportive" in rendered
    assert "- therapeutic_approach: none" in rendered
