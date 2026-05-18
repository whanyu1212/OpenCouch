"""Guided exercise text agent definition."""

from __future__ import annotations

from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.prompts import compose_sources as _compose
from agent.runtime.agents.common import (
    AgentDefinition,
    build_agent,
    definition_with_instructions,
)
from agent.runtime.agents.therapeutic_prompt_instructions import (
    _GUIDED_EXERCISE_INSTRUCTIONS,
)
from agent.runtime.agents.therapeutic_prompt_sources import (
    _knowledge_for_response_style,
)
from agent.runtime.agents.therapeutic_prompts import (
    _compose_system_prompt_with_state,
    _read_approach,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.tools.guided_exercise import build_guided_exercise_tools
from agent.state import AgentState


GUIDED_EXERCISE_AGENT_NAME = "OpenCouch guided exercise text agent"

GUIDED_EXERCISE_AGENT_INSTRUCTIONS = """\
You are the OpenCouch guided exercise specialist. The application runtime owns
exercise consent, exercise id validation, step state, exit behavior, and
completion. Follow runtime-provided exercise state exactly and do not invent
unsupported steps or start an exercise without runtime state.

Guided-exercise tools:
- Call load_guided_exercise_skill when the runtime prompt requires it.
- Use only the returned skill_context plus the runtime task for the current
  exercise reply.
- Call record_guided_exercise_progress only when the user's latest response
  changes active exercise state: complete, partial, hold, stuck, exit, or unsafe.
- Do not browse, offer a menu, change exercise, skip steps, or add steps. The
  runtime validates progress and computes the next step.
"""

_GUIDED_EXERCISE_DEFINITION = AgentDefinition(
    name=GUIDED_EXERCISE_AGENT_NAME,
    handoff_description=(
        "Owns guided exercise wording after runtime-owned exercise state starts."
    ),
    instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
)


def build_guided_exercise_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the guided exercise specialist definition."""

    return build_agent(
        definition_with_instructions(_GUIDED_EXERCISE_DEFINITION, instructions),
        model=model,
        tools=tools if tools is not None else build_guided_exercise_tools(),
    )


def build_guided_exercise_system_prompt(state: AgentState) -> str:
    """Build the system prompt owned by the guided-exercise specialist."""

    exercise_state = state.get("exercise_state", {})
    approach = (
        exercise_state.get("exercise_therapeutic_approach")
        if exercise_state.get("exercise_type")
        else None
    ) or _read_approach(state)
    files = _knowledge_for_response_style("guided_exercise", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _GUIDED_EXERCISE_INSTRUCTIONS, state
    )


__all__ = [
    "GUIDED_EXERCISE_AGENT_INSTRUCTIONS",
    "GUIDED_EXERCISE_AGENT_NAME",
    "build_guided_exercise_agent",
    "build_guided_exercise_system_prompt",
]
