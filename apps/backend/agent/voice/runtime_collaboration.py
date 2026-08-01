"""Narrow runtime operations required by the Realtime voice facade."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState, AgentTurnInputState
from llm.base import BaseLLMClient


GetState = Callable[[str], Awaitable[AgentState | None]]
BuildTurnInitialState = Callable[..., AgentTurnInputState]
BuildWorkflowContext = Callable[..., WorkflowContext]
PrepareSessionForTurn = Callable[..., Awaitable[None]]
RememberLlmClient = Callable[[str, BaseLLMClient | None], None]
EnsureSdkTurnRecorded = Callable[..., Awaitable[None]]


@dataclass(frozen=True, slots=True)
class VoiceRuntimeCollaboration:
    """Runtime operations the voice facade needs to coordinate a turn."""

    get_state: GetState
    build_turn_initial_state: BuildTurnInitialState
    build_workflow_context: BuildWorkflowContext
    prepare_session_for_turn: PrepareSessionForTurn
    remember_llm_client: RememberLlmClient
    ensure_sdk_turn_recorded: EnsureSdkTurnRecorded


__all__ = ["VoiceRuntimeCollaboration"]
