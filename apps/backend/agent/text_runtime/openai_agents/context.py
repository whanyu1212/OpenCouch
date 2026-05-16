"""Local context for dormant OpenAI Agents SDK text definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from agent.memory.modes import MemoryMode
from agent.models import Channel, CrisisAssessment
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


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
