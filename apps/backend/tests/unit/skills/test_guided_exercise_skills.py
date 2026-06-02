"""Tests for guided exercise skill catalog and rendering."""

from __future__ import annotations

import pytest

from agent.skills.guided_exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_STOP_TECHNIQUE,
    available_exercise_definitions,
    get_exercise_definition,
    iter_exercise_definitions,
)
from agent.skills.guided_exercises.rendering.skill_context import (
    build_exercise_skill,
    render_exercise_skill_context,
)


def test_all_registered_exercises_build_prompt_ready_skills() -> None:
    for definition in iter_exercise_definitions():
        skill = build_exercise_skill(definition.id)

        assert skill.skill_id == definition.id
        assert skill.version == definition.version
        assert skill.name == definition.display_name
        assert skill.when_to_use == definition.selection_use_case
        assert len(skill.steps) == len(definition.steps)
        assert skill.steps[0].step_id == definition.steps[0].id


def test_exercise_skill_context_lazy_loads_current_step_detail() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_5_4_3_2_1,
        current_step_index=1,
        runtime_action="advance",
    )

    assert rendered.startswith("Exercise skill:")
    assert f"- skill_id: {EXERCISE_5_4_3_2_1}" in rendered
    assert "- runtime_action: advance" in rendered
    assert "Current runtime step:" in rendered
    assert "- step_id: hear" in rendered
    assert "Step map:" in rendered


def test_exercise_skill_context_can_render_l1_l2_without_current_step() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_BOX_BREATHING,
        current_step_index=None,
        runtime_action="start",
    )

    assert "Exercise skill:" in rendered
    assert "- name: a box breathing cycle" in rendered
    assert "Current runtime step:" not in rendered
    assert "Step map:" in rendered


def test_exercise_skill_context_includes_selected_skill_doc_guidance() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_5_4_3_2_1,
        current_step_index=1,
        runtime_action="advance",
    )

    assert "Skill document guidance:" in rendered
    assert "# 5-4-3-2-1 Grounding" in rendered
    assert "## Operating boundaries" in rendered
    assert "Operating boundaries:" in rendered
    assert "Current runtime step:" in rendered
    assert "- step_id: hear" in rendered
    assert "Step map:" in rendered


def test_exercise_skill_context_omits_doc_guidance_for_undocumented_exercise() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_STOP_TECHNIQUE,
        current_step_index=0,
        runtime_action="start",
    )

    assert "Skill document guidance:" not in rendered
    assert f"- skill_id: {EXERCISE_STOP_TECHNIQUE}" in rendered
    assert "Operating boundaries:" in rendered
    assert "Current runtime step:" in rendered
    assert "Step map:" in rendered


def test_available_exercises_filter_by_channel_and_capability() -> None:
    text_ids = {definition.id for definition in available_exercise_definitions()}
    voice_ids = {
        definition.id for definition in available_exercise_definitions(channel="voice")
    }

    assert EXERCISE_5_4_3_2_1 in text_ids
    assert voice_ids
    assert voice_ids.issubset(
        {definition.id for definition in iter_exercise_definitions()}
    )


def test_legacy_skill_context_import_path_reexports_renderer() -> None:
    from agent.skills.guided_exercises.skills import (
        render_exercise_skill_context as legacy_render,
    )

    rendered = legacy_render(
        EXERCISE_5_4_3_2_1,
        current_step_index=0,
        runtime_action="start",
    )

    assert "Exercise skill:" in rendered
    assert f"- skill_id: {EXERCISE_5_4_3_2_1}" in rendered


def test_build_exercise_skill_rejects_unknown_exercise() -> None:
    assert get_exercise_definition("unknown_exercise") is None
    with pytest.raises(KeyError):
        build_exercise_skill("unknown_exercise")
