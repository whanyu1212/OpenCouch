"""Shared builders for OpenAI text agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.runtime.context import OpenAITextRunContext


@dataclass(frozen=True)
class AgentDefinition:
    """Static metadata for one OpenAI text agent."""

    name: str
    handoff_description: str
    instructions: str


def build_agent(
    definition: AgentDefinition,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build an OpenAI Agents SDK agent from OpenCouch metadata."""

    return Agent[OpenAITextRunContext](
        name=definition.name,
        handoff_description=definition.handoff_description,
        instructions=definition.instructions,
        model=model,
        tools=list(tools or ()),
    )


def definition_with_instructions(
    definition: AgentDefinition,
    instructions: str | None,
) -> AgentDefinition:
    """Override instructions while preserving identity and handoff metadata."""

    if instructions is None:
        return definition
    return AgentDefinition(
        name=definition.name,
        handoff_description=definition.handoff_description,
        instructions=instructions,
    )
