"""Local context for dormant OpenAI Agents SDK text definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, cast

from agent.memory.modes import MemoryMode
from agent.models import Channel, CrisisAssessment
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


MemoryReadActionType = Literal["list", "status"]


@dataclass(frozen=True, slots=True)
class MemoryToolCallRecord:
    """One read-only memory tool result captured from an SDK run."""

    tool_name: str
    action_type: MemoryReadActionType
    response_text: str
    memory_control: dict[str, Any]
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

    def record_memory_tool_result(
        self,
        *,
        action_type: MemoryReadActionType,
        response_text: str,
        memory_control: Mapping[str, Any],
    ) -> None:
        """Remember a read-only memory tool result for state merge/diagnostics."""

        tool_name = (
            "show_saved_memory" if action_type == "list" else "show_memory_status"
        )
        self.memory_tool_calls.append(
            MemoryToolCallRecord(
                tool_name=tool_name,
                action_type=action_type,
                response_text=response_text,
                memory_control=dict(memory_control),
            )
        )

    def latest_memory_tool_result(
        self,
        action_type: MemoryReadActionType,
    ) -> MemoryToolCallRecord | None:
        """Return the latest read-only memory tool result for an action type."""

        for call in reversed(self.memory_tool_calls):
            if call.action_type == action_type:
                return call
        return None

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
