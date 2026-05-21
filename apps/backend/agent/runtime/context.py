"""Local context passed to OpenAI Agents SDK text tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

from agent.models import Channel
from agent.runtime_context import WorkflowContext

if TYPE_CHECKING:
    from agent.state import AgentState


MemoryActionType = Literal[
    "list",
    "status",
    "set_recall",
    "save_preference",
    "forget_by_index",
    "forget_by_query",
    "confirm_pending",
    "cancel_pending",
]
MemoryReadActionType = Literal["list", "status"]
MemoryToolSideEffect = Literal[
    "none",
    "procedural_profile_update",
    "pending_deletion",
    "delete_memory",
    "cancel_pending",
]
GroundedToolStatus = Literal["answered", "no_verified_answer"]
CrisisResourceToolStatus = Literal[
    "not_attempted",
    "found",
    "no_location",
    "location_refused",
    "no_verified_results",
]
GuidedExerciseProgressOutcome = Literal[
    "complete",
    "partial",
    "hold",
    "stuck",
    "exit",
    "unsafe",
]
GuidedExerciseProgressStatus = Literal[
    "active",
    "completed",
    "cancelled",
    "conflict",
    "unsafe",
]
GuidedExerciseRuntimeAction = Literal[
    "advance",
    "hold",
    "simplify",
    "complete",
    "cancel",
    "crisis",
    "conflict",
]


@dataclass(frozen=True, slots=True)
class MemoryToolCallRecord:
    """One memory tool result captured from an SDK run."""

    tool_name: str
    action_type: MemoryActionType
    response_text: str
    memory_control: dict[str, Any]
    procedural_profile: dict[str, Any] | None = None
    side_effect: MemoryToolSideEffect = "none"
    retry_safe: bool = True


@dataclass(frozen=True, slots=True)
class GroundedToolCallRecord:
    """One grounded lookup tool result captured from an SDK run."""

    tool_name: str
    query: str
    response_text: str
    grounded_lookup: dict[str, Any]
    status: GroundedToolStatus
    side_effect: Literal["none"] = "none"
    retry_safe: bool = True


@dataclass(frozen=True, slots=True)
class CrisisResourceToolCallRecord:
    """One crisis-resource lookup result captured from an SDK run."""

    tool_name: str
    response_text: str
    inferred_location: str
    found_resources: list[dict[str, str]]
    resource_lookup_status: CrisisResourceToolStatus
    side_effect: Literal["none"] = "none"
    retry_safe: bool = True


@dataclass(frozen=True, slots=True)
class GuidedExerciseSkillToolCallRecord:
    """One guided-exercise skill load captured from an SDK run."""

    tool_name: str
    exercise_type: str
    current_step_index: int | None
    runtime_action: str
    skill_context: str
    side_effect: Literal["none"] = "none"
    retry_safe: bool = True


@dataclass(frozen=True, slots=True)
class GuidedExerciseProgressToolCallRecord:
    """One guided-exercise progress update captured from an SDK run."""

    tool_name: str
    expected_skill_id: str
    expected_step_id: str
    outcome: GuidedExerciseProgressOutcome
    status: GuidedExerciseProgressStatus
    runtime_action: GuidedExerciseRuntimeAction
    exercise_state_delta: dict[str, Any]
    response_instruction: str
    side_effect: Literal["active_skill_state_update", "none"] = "none"
    retry_safe: bool = False


@dataclass(frozen=True, slots=True)
class TherapeuticResponseSkillToolCallRecord:
    """One therapeutic response skill load captured from an SDK run."""

    tool_name: str
    response_style: str
    therapeutic_approach: str
    skill_context: str
    side_effect: Literal["none"] = "none"
    retry_safe: bool = True


@dataclass(slots=True)
class OpenAITextRunContext:
    """Application-owned context passed to OpenAI text agents.

    The Agents SDK keeps this object local to Python tool handlers. It is not
    automatically shown to the model, which lets OpenCouch keep stores,
    mutation state, and backend clients out of model-visible context.
    """

    thread_id: str
    workflow_context: WorkflowContext
    current_user_message: str
    user_id: str | None = None
    session_id: str | None = None
    channel: Channel = Channel.TEST
    pending_memory_action: Mapping[str, Any] | None = None
    agent_state: "AgentState | None" = None
    installed_skills: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    crisis_guardrail_output: Any | None = None
    crisis_guardrail_triggered: bool = False
    memory_tool_calls: list[MemoryToolCallRecord] = field(default_factory=list)
    grounded_tool_calls: list[GroundedToolCallRecord] = field(default_factory=list)
    crisis_resource_tool_calls: list[CrisisResourceToolCallRecord] = field(
        default_factory=list
    )
    guided_exercise_skill_tool_calls: list[GuidedExerciseSkillToolCallRecord] = field(
        default_factory=list
    )
    guided_exercise_progress_tool_calls: list[GuidedExerciseProgressToolCallRecord] = (
        field(default_factory=list)
    )
    therapeutic_response_skill_tool_calls: list[
        TherapeuticResponseSkillToolCallRecord
    ] = field(default_factory=list)

    def record_memory_tool_result(
        self,
        *,
        action_type: MemoryActionType,
        response_text: str,
        memory_control: Mapping[str, Any],
        procedural_profile: Mapping[str, Any] | None = None,
        side_effect: MemoryToolSideEffect = "none",
        retry_safe: bool = True,
    ) -> None:
        """Remember a memory tool result for state merge/diagnostics."""

        self.memory_tool_calls.append(
            MemoryToolCallRecord(
                tool_name=_memory_tool_name(action_type),
                action_type=action_type,
                response_text=response_text,
                memory_control=dict(memory_control),
                procedural_profile=(
                    dict(procedural_profile) if procedural_profile is not None else None
                ),
                side_effect=side_effect,
                retry_safe=retry_safe,
            )
        )

    def record_grounded_tool_result(
        self,
        *,
        query: str,
        response_text: str,
        grounded_lookup: Mapping[str, Any],
        status: GroundedToolStatus,
    ) -> None:
        """Remember a grounded lookup tool result for state merge/diagnostics."""

        self.grounded_tool_calls.append(
            GroundedToolCallRecord(
                tool_name="answer_grounded_lookup",
                query=query,
                response_text=response_text,
                grounded_lookup=dict(grounded_lookup),
                status=status,
            )
        )

    def latest_grounded_tool_result(self) -> GroundedToolCallRecord | None:
        """Return the latest grounded lookup tool result."""

        return self.grounded_tool_calls[-1] if self.grounded_tool_calls else None

    def record_crisis_resource_tool_result(
        self,
        *,
        response_text: str,
        inferred_location: str,
        found_resources: list[dict[str, str]],
        resource_lookup_status: CrisisResourceToolStatus,
    ) -> None:
        """Remember a crisis-resource lookup result for state merge."""

        self.crisis_resource_tool_calls.append(
            CrisisResourceToolCallRecord(
                tool_name="lookup_crisis_resources",
                response_text=response_text,
                inferred_location=inferred_location,
                found_resources=[dict(resource) for resource in found_resources],
                resource_lookup_status=resource_lookup_status,
            )
        )

    def latest_crisis_resource_tool_result(
        self,
    ) -> CrisisResourceToolCallRecord | None:
        """Return the latest crisis-resource lookup tool result."""

        return (
            self.crisis_resource_tool_calls[-1]
            if self.crisis_resource_tool_calls
            else None
        )

    def record_guided_exercise_skill_tool_result(
        self,
        *,
        exercise_type: str,
        current_step_index: int | None,
        runtime_action: str,
        skill_context: str,
    ) -> None:
        """Remember a guided-exercise skill load for diagnostics."""

        self.guided_exercise_skill_tool_calls.append(
            GuidedExerciseSkillToolCallRecord(
                tool_name="load_guided_exercise_skill",
                exercise_type=exercise_type,
                current_step_index=current_step_index,
                runtime_action=runtime_action,
                skill_context=skill_context,
            )
        )

    def latest_guided_exercise_skill_tool_result(
        self,
    ) -> GuidedExerciseSkillToolCallRecord | None:
        """Return the latest guided-exercise skill tool result."""

        return (
            self.guided_exercise_skill_tool_calls[-1]
            if self.guided_exercise_skill_tool_calls
            else None
        )

    def record_guided_exercise_progress_tool_result(
        self,
        *,
        expected_skill_id: str,
        expected_step_id: str,
        outcome: GuidedExerciseProgressOutcome,
        status: GuidedExerciseProgressStatus,
        runtime_action: GuidedExerciseRuntimeAction,
        exercise_state_delta: Mapping[str, Any],
        response_instruction: str,
        side_effect: Literal["active_skill_state_update", "none"],
        retry_safe: bool,
    ) -> None:
        """Remember a guided-exercise progress update for diagnostics."""

        self.guided_exercise_progress_tool_calls.append(
            GuidedExerciseProgressToolCallRecord(
                tool_name="record_guided_exercise_progress",
                expected_skill_id=expected_skill_id,
                expected_step_id=expected_step_id,
                outcome=outcome,
                status=status,
                runtime_action=runtime_action,
                exercise_state_delta=dict(exercise_state_delta),
                response_instruction=response_instruction,
                side_effect=side_effect,
                retry_safe=retry_safe,
            )
        )

    def latest_guided_exercise_progress_tool_result(
        self,
    ) -> GuidedExerciseProgressToolCallRecord | None:
        """Return the latest guided-exercise progress tool result."""

        return (
            self.guided_exercise_progress_tool_calls[-1]
            if self.guided_exercise_progress_tool_calls
            else None
        )

    def record_therapeutic_response_skill_tool_result(
        self,
        *,
        response_style: str,
        therapeutic_approach: str,
        skill_context: str,
    ) -> None:
        """Remember a therapeutic response skill load for diagnostics."""

        self.therapeutic_response_skill_tool_calls.append(
            TherapeuticResponseSkillToolCallRecord(
                tool_name="load_therapeutic_response_skill",
                response_style=response_style,
                therapeutic_approach=therapeutic_approach,
                skill_context=skill_context,
            )
        )

    def latest_therapeutic_response_skill_tool_result(
        self,
    ) -> TherapeuticResponseSkillToolCallRecord | None:
        """Return the latest therapeutic response skill tool result."""

        return (
            self.therapeutic_response_skill_tool_calls[-1]
            if self.therapeutic_response_skill_tool_calls
            else None
        )


def _memory_tool_name(action_type: MemoryActionType) -> str:
    return {
        "list": "show_saved_memory",
        "status": "show_memory_status",
        "set_recall": "set_proactive_memory_recall",
        "save_preference": "save_response_preference",
        "forget_by_index": "prepare_memory_deletion_by_index",
        "forget_by_query": "prepare_memory_deletion_by_query",
        "confirm_pending": "confirm_memory_deletion",
        "cancel_pending": "cancel_memory_deletion",
    }[action_type]
