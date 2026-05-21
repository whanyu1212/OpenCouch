"""Crisis response text agent definition."""

from __future__ import annotations

from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.guardrails.prompts import build_crisis_response_system_prompt
from agent.specialists.common import (
    AgentDefinition,
    build_agent,
    definition_with_instructions,
)
from agent.runtime.context import OpenAITextRunContext
from agent.specialists.therapeutic_prompts import build_clarifying_system_prompt
from agent.state import AgentState
from agent.tools.crisis import build_crisis_response_tools


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

RUNTIME_CRISIS_INSTRUCTIONS = """\
You are the OpenCouch crisis text specialist for a turn already classified by
the application runtime. The runtime owns crisis assessment, audit logging,
persistence, memory mutation, and guided-exercise state. You own crisis
response wording and may own crisis-resource lookup when the runtime prompt
requires the attached lookup_crisis_resources tool.
Do not reclassify the user or invent crisis resources. Follow the provided
prompt context exactly for either level-1 safety clarification or level-2/3
crisis response.
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


def build_runtime_crisis_agent(
    *,
    state: AgentState,
    runtime_mode: str,
    base_agent: Agent[OpenAITextRunContext],
    enable_resource_tools: bool | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the runtime-specific crisis agent variant for a classified turn."""

    if runtime_mode == "crisis_response":
        system_prompt = build_crisis_response_system_prompt()
        tools = (
            [
                tool
                for tool in base_agent.tools
                if tool.name == "lookup_crisis_resources"
            ]
            if enable_resource_tools is not False
            else []
        )
    elif runtime_mode == "crisis_clarification":
        system_prompt = build_clarifying_system_prompt(state)
        tools = []
    else:
        raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

    instructions = f"{RUNTIME_CRISIS_INSTRUCTIONS}\n\n{system_prompt}"
    return Agent[OpenAITextRunContext](
        name=base_agent.name,
        handoff_description=base_agent.handoff_description,
        instructions=instructions,
        model=base_agent.model,
        tools=tools,
    )


__all__ = [
    "CRISIS_AGENT_INSTRUCTIONS",
    "CRISIS_AGENT_NAME",
    "RUNTIME_CRISIS_INSTRUCTIONS",
    "build_crisis_response_agent",
    "build_runtime_crisis_agent",
]
