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
from agent.skills.guided_exercises.catalog.types import ExerciseDefinition, ExerciseStep
from agent.skills.guided_exercises.rendering.skill_context import (
    build_exercise_skill,
    render_exercise_skill_context,
)
from agent.tools import guided_exercise as guided_exercise_tools
from agent.tools.guided_exercise import (
    execute_guided_exercise_discovery_tool,
    execute_guided_exercise_progress_tool,
    execute_guided_exercise_start_tool,
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


def _run_context(
    *,
    installed_skills: list[str] | None = None,
) -> OpenAITextRunContext:
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
        installed_skills=list(installed_skills or []),
    )


def _guided_exercise_context(
    *,
    step_index: int = 0,
) -> OpenAITextRunContext:
    context = _run_context()
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None
    step = definition.steps[step_index]
    context.agent_state = {
        "exercise_state": {
            "exercise_type": EXERCISE_BOX_BREATHING,
            "exercise_step": step_index,
            "exercise_step_id": step.id,
            "exercise_version": definition.version,
            "exercise_therapeutic_approach": "dbt_skills",
        }
    }
    return context


@pytest.mark.asyncio
async def test_start_tool_returns_first_step_state_and_skill_context() -> None:
    result = await execute_guided_exercise_start_tool(
        _run_context(),
        exercise_type=EXERCISE_BOX_BREATHING,
        therapeutic_approach="dbt_skills",
    )

    assert result.status == "active"
    assert result.runtime_action == "start"
    assert result.current_step_index == 0
    assert result.current_step_id == "inhale"
    assert result.exercise_state_delta["exercise_state"] == {
        "exercise_type": EXERCISE_BOX_BREATHING,
        "exercise_step": 0,
        "exercise_step_id": "inhale",
        "exercise_version": 1,
        "exercise_therapeutic_approach": "dbt_skills",
    }
    assert f"- skill_id: {EXERCISE_BOX_BREATHING}" in result.skill_context


@pytest.mark.asyncio
async def test_start_tool_rejects_unknown_exercise() -> None:
    with pytest.raises(ValueError, match="Unknown guided exercise"):
        await execute_guided_exercise_start_tool(
            _run_context(),
            exercise_type="not_registered",
        )


@pytest.mark.asyncio
async def test_start_tool_rejects_replacing_an_active_exercise() -> None:
    result = await execute_guided_exercise_start_tool(
        _guided_exercise_context(),
        exercise_type=EXERCISE_5_4_3_2_1,
    )

    assert result.status == "conflict"
    assert result.runtime_action == "conflict"
    assert result.skill_id == EXERCISE_BOX_BREATHING
    assert result.exercise_state_delta == {}


def _definition(
    exercise_id: str,
    *,
    required_capability: str | None = None,
) -> ExerciseDefinition:
    return ExerciseDefinition(
        id=exercise_id,
        display_name=exercise_id.replace("_", " ").title(),
        selection_use_case=f"{exercise_id} support",
        steps=(
            ExerciseStep(
                instruction="Try one small step.",
                id="step",
                completion_mode="confirmation",
            ),
        ),
        selection_aliases=(exercise_id.replace("_", " "),),
        required_capability=required_capability,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "status", "runtime_action", "exercise_state_delta"),
    [
        ("partial", "active", "hold", {}),
        ("hold", "active", "hold", {}),
        ("stuck", "active", "simplify", {}),
        ("unsafe", "unsafe", "crisis", None),
        ("exit", "cancelled", "cancel", None),
    ],
)
async def test_progress_tool_adapts_shared_transition_outcomes(
    outcome: str,
    status: str,
    runtime_action: str,
    exercise_state_delta: dict[str, object] | None,
) -> None:
    result = await execute_guided_exercise_progress_tool(
        _guided_exercise_context(),
        expected_skill_id=EXERCISE_BOX_BREATHING,
        expected_step_id="inhale",
        outcome=outcome,  # type: ignore[arg-type]
        user_response_summary="The user responded to the current step.",
    )

    assert result.status == status
    assert result.runtime_action == runtime_action
    if outcome == "exit":
        assert result.exercise_state_delta["exercise_state"]["exercise_type"] is None
    elif exercise_state_delta is None:
        assert result.exercise_state_delta == {}
    else:
        assert result.exercise_state_delta == {"exercise_state": exercise_state_delta}


@pytest.mark.asyncio
async def test_progress_tool_adapts_terminal_completion() -> None:
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None
    final_index = len(definition.steps) - 1
    final_step = definition.steps[final_index]

    result = await execute_guided_exercise_progress_tool(
        _guided_exercise_context(step_index=final_index),
        expected_skill_id=EXERCISE_BOX_BREATHING,
        expected_step_id=final_step.id,
        outcome="complete",
        user_response_summary="The user completed the final step.",
    )

    assert result.status == "completed"
    assert result.runtime_action == "complete"
    assert result.exercise_state_delta["exercise_state"]["exercise_type"] is None


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


def test_no_production_exercises_require_installed_capabilities() -> None:
    assert all(
        definition.required_capability is None
        for definition in iter_exercise_definitions()
    )


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


@pytest.mark.asyncio
async def test_discovery_tool_filters_gated_exercises_by_installed_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basic = _definition("basic")
    gated = _definition(
        "gated",
        required_capability="advanced_exercises",
    )
    definitions = (basic, gated)

    def fake_available_exercise_definitions(
        **kwargs: object,
    ) -> tuple[ExerciseDefinition, ...]:
        return available_exercise_definitions(
            definitions=definitions,
            installed_skills=kwargs.get("installed_skills", ()),
            channel=kwargs.get("channel", "text"),
            therapeutic_approach=kwargs.get("therapeutic_approach"),
        )

    monkeypatch.setattr(
        guided_exercise_tools,
        "available_exercise_definitions",
        fake_available_exercise_definitions,
    )

    without_capability = await execute_guided_exercise_discovery_tool(_run_context())
    assert [skill.skill_id for skill in without_capability.skills] == ["basic"]

    with_capability = await execute_guided_exercise_discovery_tool(
        _run_context(installed_skills=["advanced_exercises"])
    )
    assert [skill.skill_id for skill in with_capability.skills] == [
        "basic",
        "gated",
    ]
    assert with_capability.skills[1].required_capability == "advanced_exercises"


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
