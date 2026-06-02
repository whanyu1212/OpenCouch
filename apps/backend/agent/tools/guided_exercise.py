"""OpenAI Agents SDK guided-exercise tools."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.runtime.context import (
    GuidedExerciseProgressOutcome,
    GuidedExerciseProgressStatus,
    GuidedExerciseRuntimeAction,
    OpenAITextRunContext,
)
from agent.skills.guided_exercises.registry import (
    available_exercise_definitions,
    get_exercise_definition,
)
from agent.skills.guided_exercises.rendering.skill_context import (
    render_exercise_skill_context,
)


class GuidedExerciseSkillSummary(BaseModel):
    """Metadata-only view of a guided exercise skill."""

    skill_id: str = Field(description="Registered guided exercise skill identifier.")
    name: str = Field(description="Human-readable skill name.")
    description: str = Field(
        description="Compact description of when to use the skill."
    )
    category: str = Field(description="Broad guided exercise category.")
    tags: list[str] = Field(description="Selection and filtering tags.")
    estimated_seconds: int | None = Field(
        default=None,
        description="Approximate duration in seconds, when known.",
    )
    intensity: str = Field(description="Expected user effort or emotional load.")
    supported_channels: list[str] = Field(description="Supported delivery channels.")
    required_capability: str | None = Field(
        default=None,
        description="Capability required before this skill can be offered.",
    )


class GuidedExerciseSkillDiscoveryToolResult(BaseModel):
    """Structured result returned by guided-exercise discovery tools."""

    skills: list[GuidedExerciseSkillSummary] = Field(
        description="Available guided exercise skill metadata."
    )
    therapeutic_approach: str = Field(
        default="none",
        description="Therapeutic approach used for filtering.",
    )
    channel: str = Field(description="Delivery channel used for filtering.")
    side_effect: str = Field(
        default="none",
        description="Skill discovery does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying skill discovery can duplicate side effects.",
    )


class GuidedExerciseProgressToolResult(BaseModel):
    """Structured result returned by guided-exercise progress tools."""

    status: GuidedExerciseProgressStatus = Field(
        description="Validated active exercise status after applying the outcome."
    )
    runtime_action: GuidedExerciseRuntimeAction = Field(
        description="Runtime-approved next action for the guided exercise."
    )
    skill_id: str | None = Field(
        default=None,
        description="Active guided exercise skill id, when available.",
    )
    previous_step_id: str | None = Field(
        default=None,
        description="Step id the agent expected to update.",
    )
    current_step_id: str | None = Field(
        default=None,
        description="Current step id after the runtime-approved transition.",
    )
    next_step_id: str | None = Field(
        default=None,
        description="Next step id to guide, when the exercise advanced.",
    )
    exercise_state_delta: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime state delta for active exercise progress.",
    )
    response_instruction: str = Field(
        description="Instruction for the agent's next user-facing message."
    )
    side_effect: Literal["active_skill_state_update", "none"] = Field(
        default="none",
        description="Whether the tool updated active guided-exercise state.",
    )
    retry_safe: bool = Field(
        default=False,
        description="Progress updates are not retry-safe without idempotency.",
    )


class GuidedExerciseSkillToolResult(BaseModel):
    """Structured result returned by guided-exercise skill tools."""

    skill_context: str = Field(
        description="Prompt-ready exercise skill context selected by the runtime."
    )
    exercise_type: str = Field(description="Registered exercise identifier.")
    current_step_index: int | None = Field(
        default=None,
        description="Current runtime step index, when one applies.",
    )
    runtime_action: str = Field(
        description="Runtime-owned action such as start, hold, advance, or exit."
    )
    side_effect: str = Field(
        default="none",
        description="Skill loading does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the skill load can duplicate side effects.",
    )


async def execute_guided_exercise_discovery_tool(
    context: OpenAITextRunContext,
    *,
    therapeutic_approach: str | None = None,
    channel: str | None = None,
) -> GuidedExerciseSkillDiscoveryToolResult:
    """List guided exercise skill metadata available to the current run."""

    state = context.agent_state or {}
    approach = (
        str(therapeutic_approach).strip()
        if therapeutic_approach is not None
        else str(state.get("therapeutic_approach") or "none")
    )
    if not approach:
        approach = "none"
    filter_approach = None if approach == "none" else approach
    delivery_channel = str(channel or context.channel.value).strip() or "text"
    definitions = available_exercise_definitions(
        installed_skills=tuple(context.installed_skills),
        channel=delivery_channel,
        therapeutic_approach=filter_approach,
    )
    summaries = [
        GuidedExerciseSkillSummary(
            skill_id=definition.id,
            name=definition.display_name,
            description=definition.selection_use_case,
            category=definition.category or "general",
            tags=list(definition.tags),
            estimated_seconds=definition.duration_seconds,
            intensity=definition.intensity,
            supported_channels=list(definition.channels),
            required_capability=definition.required_skill,
        )
        for definition in definitions
    ]
    return GuidedExerciseSkillDiscoveryToolResult(
        skills=summaries,
        therapeutic_approach=approach,
        channel=delivery_channel,
    )


async def execute_guided_exercise_progress_tool(
    context: OpenAITextRunContext,
    *,
    expected_skill_id: str,
    expected_step_id: str,
    outcome: GuidedExerciseProgressOutcome,
    user_response_summary: str,
) -> GuidedExerciseProgressToolResult:
    """Record and validate progress for the active guided exercise step."""

    skill_id = expected_skill_id.strip()
    step_id = expected_step_id.strip()
    summary = user_response_summary.strip()
    if not skill_id:
        raise ValueError("record_guided_exercise_progress requires expected_skill_id.")
    if not step_id:
        raise ValueError("record_guided_exercise_progress requires expected_step_id.")
    if not summary:
        raise ValueError(
            "record_guided_exercise_progress requires user_response_summary."
        )

    state = context.agent_state or {}
    exercise_state = state.get("exercise_state", {}) or {}
    active_skill_id = exercise_state.get("exercise_type")
    active_step_id = exercise_state.get("exercise_step_id")
    active_step_index = exercise_state.get("exercise_step")
    definition = get_exercise_definition(skill_id)

    if (
        definition is None
        or active_skill_id != skill_id
        or active_step_id != step_id
        or not isinstance(active_step_index, int)
    ):
        result = GuidedExerciseProgressToolResult(
            status="conflict",
            runtime_action="conflict",
            skill_id=active_skill_id if isinstance(active_skill_id, str) else None,
            previous_step_id=step_id,
            current_step_id=active_step_id if isinstance(active_step_id, str) else None,
            response_instruction=(
                "Do not advance the exercise. Re-orient to the runtime-provided "
                "active step or ask the user whether they want to continue."
            ),
            side_effect="none",
            retry_safe=True,
        )
        _record_progress_result(
            context,
            expected_skill_id=skill_id,
            expected_step_id=step_id,
            outcome=outcome,
            result=result,
        )
        return result

    result = _progress_result_for_outcome(
        definition_steps=tuple(definition.steps),
        skill_id=skill_id,
        step_id=step_id,
        step_index=active_step_index,
        outcome=outcome,
        exercise_state=exercise_state,
    )
    _record_progress_result(
        context,
        expected_skill_id=skill_id,
        expected_step_id=step_id,
        outcome=outcome,
        result=result,
    )
    return result


async def execute_guided_exercise_skill_tool(
    context: OpenAITextRunContext,
    *,
    exercise_type: str,
    current_step_index: int | None,
    runtime_action: str,
) -> GuidedExerciseSkillToolResult:
    """Render one runtime-selected exercise skill through the catalog."""

    exercise_id = exercise_type.strip()
    if not exercise_id:
        raise ValueError("load_guided_exercise_skill requires exercise_type.")
    action = runtime_action.strip()
    if not action:
        raise ValueError("load_guided_exercise_skill requires runtime_action.")

    try:
        skill_context = render_exercise_skill_context(
            exercise_id,
            current_step_index=current_step_index,
            runtime_action=action,
        )
    except KeyError:
        skill_context = (
            "Exercise skill:\n"
            f"- skill_id: {exercise_id}\n"
            f"- runtime_action: {action}\n"
            "- registry_status: unavailable\n"
            "Operating boundaries:\n"
            "- Follow the runtime task exactly and do not invent extra steps."
        )
    result = GuidedExerciseSkillToolResult(
        skill_context=skill_context,
        exercise_type=exercise_id,
        current_step_index=current_step_index,
        runtime_action=action,
    )
    context.record_guided_exercise_skill_tool_result(
        exercise_type=result.exercise_type,
        current_step_index=result.current_step_index,
        runtime_action=result.runtime_action,
        skill_context=result.skill_context,
    )
    return result


@function_tool(
    name_override="list_guided_exercise_skills",
    description_override=(
        "List metadata-only guided exercise skills available for the current "
        "user, channel, installed capabilities, and therapeutic approach. Use "
        "from the Therapeutic Agent when considering whether to offer a guided "
        "exercise. Returns compact metadata only, not full exercise scripts. "
        "Side effects: none. Retry safety: safe."
    ),
)
async def list_guided_exercise_skills(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    therapeutic_approach: str | None = None,
    channel: str | None = None,
) -> GuidedExerciseSkillDiscoveryToolResult:
    """List guided exercise skills available to this run."""

    return await execute_guided_exercise_discovery_tool(
        wrapper.context,
        therapeutic_approach=therapeutic_approach,
        channel=channel,
    )


@function_tool(
    name_override="record_guided_exercise_progress",
    description_override=(
        "Record what happened on the current guided-exercise step. Use only "
        "when the user's latest response changes exercise state: complete, "
        "partial, hold, stuck, exit, or unsafe. Requires expected_skill_id and "
        "expected_step_id from the loaded skill context; the runtime validates "
        "them and computes the next step. Side effects: active skill state "
        "update when validation succeeds. Retry safety: not guaranteed."
    ),
)
async def record_guided_exercise_progress(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    expected_skill_id: str,
    expected_step_id: str,
    outcome: GuidedExerciseProgressOutcome,
    user_response_summary: str,
) -> GuidedExerciseProgressToolResult:
    """Record progress for the active guided exercise step."""

    return await execute_guided_exercise_progress_tool(
        wrapper.context,
        expected_skill_id=expected_skill_id,
        expected_step_id=expected_step_id,
        outcome=outcome,
        user_response_summary=user_response_summary,
    )


@function_tool(
    name_override="load_guided_exercise_skill",
    description_override=(
        "Load the runtime-selected guided-exercise skill block for the current "
        "step and action. Use only when the runtime prompt requires it for a "
        "GuidedExerciseAgent turn. Side effects: none. Retry safety: safe."
    ),
)
async def load_guided_exercise_skill(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    exercise_type: str,
    runtime_action: str,
    current_step_index: int | None = None,
) -> GuidedExerciseSkillToolResult:
    """Load one guided-exercise skill selected by the app runtime."""

    return await execute_guided_exercise_skill_tool(
        wrapper.context,
        exercise_type=exercise_type,
        current_step_index=current_step_index,
        runtime_action=runtime_action,
    )


def build_guided_exercise_discovery_tools() -> list[Any]:
    """Return guided-exercise discovery tools for the primary agent."""

    return [list_guided_exercise_skills]


def build_guided_exercise_tools() -> list[Any]:
    """Return guided-exercise tools for the OpenAI specialist."""

    return [load_guided_exercise_skill, record_guided_exercise_progress]


def _progress_result_for_outcome(
    *,
    definition_steps: tuple[Any, ...],
    skill_id: str,
    step_id: str,
    step_index: int,
    outcome: GuidedExerciseProgressOutcome,
    exercise_state: Mapping[str, Any],
) -> GuidedExerciseProgressToolResult:
    if outcome == "unsafe":
        return GuidedExerciseProgressToolResult(
            status="unsafe",
            runtime_action="crisis",
            skill_id=skill_id,
            previous_step_id=step_id,
            current_step_id=step_id,
            response_instruction=(
                "Stop exercise guidance and follow crisis/safety routing. Do not "
                "continue the exercise in this response."
            ),
            side_effect="none",
            retry_safe=True,
        )
    if outcome == "exit":
        return GuidedExerciseProgressToolResult(
            status="cancelled",
            runtime_action="cancel",
            skill_id=skill_id,
            previous_step_id=step_id,
            exercise_state_delta={"exercise_state": _cleared_exercise_state()},
            response_instruction=(
                "Acknowledge the user's choice to stop, do not continue the "
                "exercise, and hand conversational ownership back to therapeutic support."
            ),
            side_effect="active_skill_state_update",
        )
    if outcome in {"partial", "hold"}:
        return GuidedExerciseProgressToolResult(
            status="active",
            runtime_action="hold",
            skill_id=skill_id,
            previous_step_id=step_id,
            current_step_id=step_id,
            exercise_state_delta={"exercise_state": {}},
            response_instruction=(
                "Stay on the current step. Validate the user's effort and gently "
                "invite a little more engagement without pressuring them."
            ),
            side_effect="none",
            retry_safe=True,
        )
    if outcome == "stuck":
        return GuidedExerciseProgressToolResult(
            status="active",
            runtime_action="simplify",
            skill_id=skill_id,
            previous_step_id=step_id,
            current_step_id=step_id,
            exercise_state_delta={"exercise_state": {}},
            response_instruction=(
                "Stay on the current step and offer a smaller, simpler version "
                "of the same task. Do not advance yet."
            ),
            side_effect="none",
            retry_safe=True,
        )

    next_index = step_index + 1
    if next_index >= len(definition_steps):
        return GuidedExerciseProgressToolResult(
            status="completed",
            runtime_action="complete",
            skill_id=skill_id,
            previous_step_id=step_id,
            exercise_state_delta={"exercise_state": _cleared_exercise_state()},
            response_instruction=(
                "The exercise is complete. Briefly reflect completion, invite the "
                "user to notice how they feel now, and return to therapeutic support."
            ),
            side_effect="active_skill_state_update",
        )

    next_step = definition_steps[next_index]
    delta = {
        "exercise_state": {
            "exercise_type": skill_id,
            "exercise_step": next_index,
            "exercise_step_id": next_step.id,
            "exercise_version": exercise_state.get("exercise_version"),
            "exercise_therapeutic_approach": exercise_state.get(
                "exercise_therapeutic_approach"
            ),
        }
    }
    return GuidedExerciseProgressToolResult(
        status="active",
        runtime_action="advance",
        skill_id=skill_id,
        previous_step_id=step_id,
        current_step_id=next_step.id,
        next_step_id=next_step.id,
        exercise_state_delta=delta,
        response_instruction=(
            "Advance to the next registered step. Use load_guided_exercise_skill "
            "for the new current step before giving step wording."
        ),
        side_effect="active_skill_state_update",
    )


def _cleared_exercise_state() -> dict[str, None]:
    return {
        "exercise_type": None,
        "exercise_step": None,
        "exercise_step_id": None,
        "exercise_version": None,
        "exercise_therapeutic_approach": None,
    }


def _record_progress_result(
    context: OpenAITextRunContext,
    *,
    expected_skill_id: str,
    expected_step_id: str,
    outcome: GuidedExerciseProgressOutcome,
    result: GuidedExerciseProgressToolResult,
) -> None:
    context.record_guided_exercise_progress_tool_result(
        expected_skill_id=expected_skill_id,
        expected_step_id=expected_step_id,
        outcome=outcome,
        status=result.status,
        runtime_action=result.runtime_action,
        exercise_state_delta=result.exercise_state_delta,
        response_instruction=result.response_instruction,
        side_effect=result.side_effect,
        retry_safe=result.retry_safe,
    )


__all__ = [
    "GuidedExerciseProgressToolResult",
    "GuidedExerciseSkillDiscoveryToolResult",
    "GuidedExerciseSkillSummary",
    "GuidedExerciseSkillToolResult",
    "build_guided_exercise_discovery_tools",
    "build_guided_exercise_tools",
    "execute_guided_exercise_discovery_tool",
    "execute_guided_exercise_progress_tool",
    "execute_guided_exercise_skill_tool",
    "list_guided_exercise_skills",
    "load_guided_exercise_skill",
    "record_guided_exercise_progress",
]
