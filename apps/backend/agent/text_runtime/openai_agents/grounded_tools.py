"""OpenAI Agents SDK grounded lookup tools for the text runtime migration."""

from __future__ import annotations

from typing import Any, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.text_runtime.openai_agents.context import (
    GroundedToolStatus,
    OpenAITextRunContext,
)
from agent.tools.grounded_search import answer_factual_lookup


class GroundedLookupToolResult(BaseModel):
    """Structured result returned by grounded lookup tools."""

    response_text: str = Field(description="User-facing grounded lookup answer.")
    grounded_lookup: dict[str, Any] = Field(
        default_factory=dict,
        description="Grounded lookup state delta produced by the shared service.",
    )
    status: GroundedToolStatus = Field(
        description="Whether the lookup produced a verified answer.",
    )
    side_effect: str = Field(
        default="none",
        description="Grounded lookup tools do not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the tool can duplicate side effects.",
    )


async def execute_grounded_lookup_tool(
    context: OpenAITextRunContext,
    *,
    query: str,
) -> GroundedLookupToolResult:
    """Execute a grounded lookup through the existing service layer."""

    text = " ".join(query.strip().split())
    if not text:
        raise ValueError("answer_grounded_lookup requires a non-empty query.")
    llm_client = context.workflow_context.llm_client
    if llm_client is None:
        raise RuntimeError("answer_grounded_lookup requires an LLM client.")

    answer, status = await answer_factual_lookup(
        context.agent_state_for_grounded_lookup(text),
        llm_client=llm_client,
        query=text,
    )
    response_text = (
        answer
        if answer
        else "I couldn't verify that from reliable sources, so I don't want to guess."
    )
    tool_result = GroundedLookupToolResult(
        response_text=response_text,
        grounded_lookup={"query": text, "status": status},
        status=cast(GroundedToolStatus, status),
    )
    context.record_grounded_tool_result(
        query=text,
        response_text=tool_result.response_text,
        grounded_lookup=tool_result.grounded_lookup,
        status=tool_result.status,
    )
    return tool_result


@function_tool(
    name_override="answer_grounded_lookup",
    description_override=(
        "Answer an explicit factual, current, official, source-backed, "
        "resource-seeking, or externally verifiable lookup request using the "
        "OpenCouch grounded lookup service. Use only for explicit lookup "
        "requests, not ordinary therapeutic support. Side effects: none. Retry "
        "safety: safe."
    ),
)
async def answer_grounded_lookup(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    query: str,
) -> GroundedLookupToolResult:
    """Answer one factual lookup request without durable side effects."""

    return await execute_grounded_lookup_tool(wrapper.context, query=query)


def build_grounded_lookup_tools() -> list[Any]:
    """Return grounded lookup tools for the OpenAI text agent."""

    return [answer_grounded_lookup]
