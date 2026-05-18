"""Crisis response text agent definition."""

from __future__ import annotations

from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.runtime.agents.common import (
    AgentDefinition,
    build_agent,
    definition_with_instructions,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.tools.crisis import build_crisis_response_tools


CRISIS_AGENT_NAME = "OpenCouch crisis response text agent"

CRISIS_AGENT_INSTRUCTIONS = """\
You are the OpenCouch crisis response specialist. The application runtime must
select you only after its own crisis assessment determines either a level 1
safety clarification turn or a level 2/3 crisis response branch. Do not
classify crisis risk yourself. Provide direct, supportive, safety-oriented
language and follow any runtime-provided resource guidance.

Crisis tools:
- Call lookup_crisis_resources only when the runtime prompt requires it for a
  level 2/3 crisis response.
- Call get_crisis_support_template to structure level 2/3 crisis replies with
  deterministic safety scaffolding.
- Do not call crisis-resource tools for level 1 safety clarification.
- Never invent phone numbers or resource names.
"""

_CRISIS_DEFINITION = AgentDefinition(
    name=CRISIS_AGENT_NAME,
    handoff_description="Owns level 2/3 crisis replies after app-owned assessment.",
    instructions=CRISIS_AGENT_INSTRUCTIONS,
)


def build_crisis_response_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the crisis response specialist definition."""

    return build_agent(
        definition_with_instructions(_CRISIS_DEFINITION, instructions),
        model=model,
        tools=tools if tools is not None else build_crisis_response_tools(),
    )


__all__ = [
    "CRISIS_AGENT_INSTRUCTIONS",
    "CRISIS_AGENT_NAME",
    "build_crisis_response_agent",
]
