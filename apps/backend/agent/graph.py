"""Text-agent state plumbing and one-shot execution helpers.

The legacy graph workflow has been removed from the active text runtime. This
module remains as a compatibility import surface for state construction and the
one-shot ``run_agent`` helper used by tests/evals.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.memory.extraction_service import (
    extract_procedural_rules,
    extract_semantic_facts,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    CrisisAssessment,
    MessageRole,
    ResponseCategory,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentState
from agent.text_runtime.openai_adapter import OpenAITextAgentAdapter
from llm.base import BaseLLMClient


def build_initial_state(
    agent_input: AgentInput,
    *,
    prior_turn_count: int | None = None,
    include_input_history: bool = False,
) -> AgentGraphInputState:
    """Convert public input into the internal text-runtime state."""

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

    return AgentGraphInputState(
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
    """Normalize text runtime state into the public output contract."""

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
    """Run one turn through the OpenAI text runtime without persistent state."""

    store = memory_store or OpenCouchMemoryStore()
    crisis_log = crisis_log_backend or InMemoryCrisisLogBackend()
    initial_state = build_initial_state(agent_input, include_input_history=True)
    adapter = OpenAITextAgentAdapter()
    final_state = await adapter.run_turn(
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

    extraction_state = cast(AgentState, {**dict(initial_state), **dict(final_state)})
    crisis_assessment = final_state.get("crisis")
    if crisis_assessment is not None and getattr(crisis_assessment, "level", 0) >= 2:
        extraction_state["route"] = "crisis"
    extraction_diagnostics = await _run_extraction_synchronously(
        extraction_state,
        llm_client=llm_client,
        memory_store=store,
        memory_mode=memory_mode,
    )
    if extraction_diagnostics:
        merged_diag = {
            **dict(final_state.get("diagnostics", {})),
            **extraction_diagnostics,
        }
        final_state = {**dict(final_state), "diagnostics": merged_diag}
    return state_to_output(final_state)


async def _run_extraction_synchronously(
    state: Mapping[str, Any],
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
) -> dict[str, Any]:
    """Run both extractors sequentially and merge their diagnostics."""

    import asyncio

    semantic_outcome, procedural_outcome = await asyncio.gather(
        extract_semantic_facts(
            state,  # type: ignore[arg-type]
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            embedding_provider=None,
            session_buffer=None,
        ),
        extract_procedural_rules(
            state,  # type: ignore[arg-type]
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            session_buffer=None,
        ),
    )
    return {
        **semantic_outcome.as_diagnostics(),
        **procedural_outcome.as_diagnostics(),
    }


__all__ = [
    "build_initial_state",
    "run_agent",
    "state_to_output",
]
