"""Tests for guided exercise skill catalog and rendering."""

from __future__ import annotations

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.workflow_context import WorkflowContext

from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    available_exercise_definitions,
    get_exercise_definition,
    iter_exercise_definitions,
)
from agent.skills.guided_exercises.rendering.skill_context import (
    build_exercise_skill,
    render_exercise_skill_context,
)
from agent.tools.guided_exercise import execute_guided_exercise_discovery_tool


def test_all_registered_exercises_build_prompt_ready_skills() -> None:
    for definition in iter_exercise_definitions():
        skill = build_exercise_skill(definition.id)

        assert skill.skill_id == definition.id
        assert skill.version == definition.version
        assert skill.name == definition.display_name
        assert skill.when_to_use == definition.selection_use_case
        assert len(skill.steps) == len(definition.steps)
        assert skill.steps[0].step_id == definition.steps[0].id


def _run_context() -> OpenAITextRunContext:
    return OpenAITextRunContext(
        thread_id="thread-1",
        user_id="user-1",
        session_id="session-1",
        current_user_message="start a breathing exercise",
        workflow_context=WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        ),
    )


def test_all_registered_exercises_have_delivery_suitability_metadata() -> None:
    for definition in iter_exercise_definitions():
        assert definition.text_fit in {"good", "okay", "poor"}
        assert definition.voice_fit in {"good", "okay", "poor"}
        assert definition.interaction_pattern in {
            "paced_confirmation",
            "item_collection",
            "reflection",
            "planning",
            "cognitive_reframe",
            "imagery",
        }
        assert definition.cognitive_load in {"low", "medium", "high"}


@pytest.mark.asyncio
async def test_discovery_tool_exposes_delivery_suitability_metadata() -> None:
    result = await execute_guided_exercise_discovery_tool(
        _run_context(),
        therapeutic_approach="none",
        channel="text",
    )

    skill = next(
        skill for skill in result.skills if skill.skill_id == EXERCISE_BOX_BREATHING
    )
    assert skill.text_fit == "good"
    assert skill.voice_fit == "good"
    assert skill.interaction_pattern == "paced_confirmation"
    assert skill.cognitive_load == "low"
    assert "voice" in skill.supported_channels


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
    assert "Good. Now four things you can hear" in rendered
    assert "Operating boundaries:" in rendered
    assert "Step map:" in rendered


def test_exercise_skill_context_can_render_l1_l2_without_current_step() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_BOX_BREATHING,
        current_step_index=None,
        runtime_action="start",
    )

    assert "Exercise skill:" in rendered
    assert "- name: a box breathing cycle" in rendered
    assert "- supported_channels: text, voice" in rendered
    assert "Current runtime step:" not in rendered
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


def test_build_exercise_skill_rejects_unknown_exercise() -> None:
    assert get_exercise_definition("unknown_exercise") is None
    with pytest.raises(KeyError):
        build_exercise_skill("unknown_exercise")
