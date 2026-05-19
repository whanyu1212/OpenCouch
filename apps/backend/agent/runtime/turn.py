"""Runtime-native turn state construction and one-shot execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    CrisisAssessment,
    MessageRole,
    ResponseCategory,
)
from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.runtime_context import WorkflowContext
from agent.state import AgentTurnInputState
from llm.base import BaseLLMClient


def build_initial_state(
    agent_input: AgentInput,
    *,
    prior_turn_count: int | None = None,
    include_input_history: bool = False,
) -> AgentTurnInputState:
    """Convert public input into the internal text-runtime turn state."""

    current_user_turn = {
        "role": MessageRole.USER.value,
        "content": agent_input.message,
    }
    prior_history_turns = [
        message.model_dump(mode="json") for message in agent_input.history
    ]
    visible_history = (
        [*prior_history_turns, current_user_turn]
        if include_input_history
        else [current_user_turn]
    )

    if prior_turn_count is None:
        prior_user_turns = sum(
            1
            for msg in agent_input.history
            if (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            == "user"
        )
        turn_count = prior_user_turns + 1
    else:
        turn_count = prior_turn_count + 1

    return AgentTurnInputState(
        message=agent_input.message,
        channel=agent_input.channel,
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        installed_skills=list(agent_input.installed_skills),
        transcript=visible_history,
        working_memory=list(agent_input.working_memory),
        session_memory={"summary": ""},
        procedural_profile={
            "procedural_rules": [],
            "proactive_recall_enabled": False,
        },
        session_progress={
            "turn_count": turn_count,
            "is_guest": False,
        },
        exercise_state={},
        memory_control={},
        crisis=CrisisAssessment(),
        therapeutic_approach=None,
        response_style="pending",
        session_action="none",
        response_text="",
        should_persist_memory=False,
        diagnostics={},
        route="",
        turn_lifecycle={"active_flow": "none", "action": "none"},
        memory_reference={"mode": "none"},
        response_guidance="",
        crisis_audit={},
        grounded_lookup={"query": "", "status": "not_attempted"},
        inferred_location="",
        found_resources=[],
        resource_lookup_status="not_attempted",
    )


def state_to_output(state: Mapping[str, Any]) -> AgentOutput:
    """Normalize text-runtime state into the public output contract."""

    raw_crisis = state.get("crisis") or CrisisAssessment()
    crisis = (
        CrisisAssessment.model_validate(raw_crisis)
        if isinstance(raw_crisis, Mapping)
        else raw_crisis
    )
    response_type = (
        ResponseCategory.CRISIS if crisis.level >= 2 else ResponseCategory.THERAPEUTIC
    )
    return AgentOutput(
        response_text=state.get("response_text", ""),
        response_type=response_type,
        crisis=crisis,
        response_style=state.get("response_style"),
        therapeutic_approach=state.get("therapeutic_approach"),
        session_action=state.get("session_action", "none"),
        should_persist_memory=state.get("should_persist_memory", False),
        diagnostics=dict(state.get("diagnostics", {})),
    )


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
    memory_store: MemoryStore | None = None,
    crisis_log_backend: CrisisLogBackend | None = None,
    memory_mode: MemoryMode = MemoryMode.INCOGNITO,
) -> AgentOutput:
    """Run one turn through the OpenAI text runtime without durable state."""

    store = memory_store or OpenCouchMemoryStore()
    crisis_log = crisis_log_backend or InMemoryCrisisLogBackend()
    initial_state = build_initial_state(agent_input, include_input_history=True)
    runtime = OpenAITextRuntime()
    final_state = await runtime.run_turn(
        initial_state,
        config={
            "configurable": {"thread_id": agent_input.session_id},
            "metadata": {
                "channel": agent_input.channel.value,
                "memory_mode": memory_mode.value,
                "streaming": False,
            },
        },
        context=WorkflowContext(
            llm_client=llm_client,
            response_llm=llm_client,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=memory_mode,
        ),
        prior_state=None,
    )

    return state_to_output(final_state)


__all__ = [
    "build_initial_state",
    "run_agent",
    "state_to_output",
]
