"""Local context for dormant OpenAI Agents SDK text definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, cast

from agent.memory.modes import MemoryMode
from agent.models import Channel, CrisisAssessment
from agent.runtime_context import WorkflowContext
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
    installed_skills: list[str] = field(default_factory=list)
    turn_count: int = 0
    memory_tool_calls: list[MemoryToolCallRecord] = field(default_factory=list)
    grounded_tool_calls: list[GroundedToolCallRecord] = field(default_factory=list)

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

    def latest_memory_tool_result(
        self,
        action_type: MemoryActionType,
    ) -> MemoryToolCallRecord | None:
        """Return the latest memory tool result for an action type."""

        for call in reversed(self.memory_tool_calls):
            if call.action_type == action_type:
                return call
        return None

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

    def agent_state_for_memory_action(self, action: Mapping[str, Any]) -> AgentState:
        """Return a minimal LangGraph-state-shaped payload for memory services.

        The OpenAI tools reuse existing memory-control services instead of
        duplicating behavior. Those services currently accept ``AgentState``,
        so this adapter builds only the fields they need plus safe defaults for
        persistent channels.
        """

        memory_control: dict[str, Any] = {"action": dict(action)}
        if self.pending_memory_action is not None:
            memory_control["pending_action"] = dict(self.pending_memory_action)

        return cast(
            AgentState,
            {
                "message": self.current_user_message,
                "channel": self.channel,
                "user_id": self.user_id,
                "session_id": self.session_id or self.thread_id,
                "installed_skills": list(self.installed_skills),
                "working_memory": [],
                "session_memory": {"summary": ""},
                "procedural_profile": {},
                "session_progress": {
                    "turn_count": self.turn_count,
                    "is_guest": self.workflow_context.memory_mode
                    == MemoryMode.INCOGNITO,
                },
                "exercise_state": {},
                "memory_control": memory_control,
                "grounded_lookup": {"query": "", "status": "not_attempted"},
                "crisis": CrisisAssessment(),
                "diagnostics": {},
            },
        )

    def agent_state_for_grounded_lookup(self, query: str) -> AgentState:
        """Return a minimal state payload for grounded lookup services."""

        return cast(
            AgentState,
            {
                "message": self.current_user_message,
                "channel": self.channel,
                "user_id": self.user_id,
                "session_id": self.session_id or self.thread_id,
                "installed_skills": list(self.installed_skills),
                "working_memory": [],
                "session_memory": {"summary": ""},
                "procedural_profile": {},
                "session_progress": {
                    "turn_count": self.turn_count,
                    "is_guest": self.workflow_context.memory_mode
                    == MemoryMode.INCOGNITO,
                },
                "exercise_state": {},
                "memory_control": {
                    "pending_action": (
                        dict(self.pending_memory_action)
                        if self.pending_memory_action is not None
                        else None
                    )
                },
                "grounded_lookup": {"query": query, "status": "not_attempted"},
                "crisis": CrisisAssessment(),
                "diagnostics": {},
            },
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
