"""OpenAI Agents SDK text agent roster."""

from __future__ import annotations

from dataclasses import dataclass

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.specialists.crisis import build_crisis_response_agent
from agent.specialists.guided_exercise import build_guided_exercise_agent
from agent.specialists.therapeutic import build_therapeutic_agent
from agent.specialists.triage import build_triage_agent
from agent.runtime.context import OpenAITextRunContext


@dataclass(frozen=True)
class OpenAITextAgentRoster:
    """OpenAI text agent definitions."""

    triage_agent: Agent[OpenAITextRunContext]
    therapeutic_agent: Agent[OpenAITextRunContext]
    crisis_agent: Agent[OpenAITextRunContext]
    guided_exercise_agent: Agent[OpenAITextRunContext]


def build_openai_text_agent_roster(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
) -> OpenAITextAgentRoster:
    """Build the OpenAI text agent roster."""

    return OpenAITextAgentRoster(
        triage_agent=build_triage_agent(model=model),
        therapeutic_agent=build_therapeutic_agent(model=model),
        crisis_agent=build_crisis_response_agent(model=model),
        guided_exercise_agent=build_guided_exercise_agent(model=model),
    )


__all__ = [
    "OpenAITextAgentRoster",
    "build_openai_text_agent_roster",
]
