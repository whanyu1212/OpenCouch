"""OpenAI Agents SDK grounded lookup tools for the text runtime migration."""

from __future__ import annotations

import time
from typing import Any, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.runtime.context import (
    GroundedToolStatus,
    OpenAITextRunContext,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.observability.timing import elapsed_ms
from agent.tools.grounded_search import (
    FactualLookupStatus,
    GroundedLookupRequest,
    answer_factual_lookup,
    answer_factual_lookup_request,
)


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


def grounded_lookup_response_delta(
    response_text: str,
    *,
    status: FactualLookupStatus,
    started_at: float,
) -> dict[str, Any]:
    """Return the shared response delta for grounded lookup turns."""

    return {
        "route": "grounded_lookup",
        "grounded_lookup": {"status": status},
        "response_style": "grounded_lookup",
        "response_text": response_text,
        "diagnostics": {"grounded_lookup_ms": elapsed_ms(started_at)},
    }


async def build_grounded_lookup_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Answer a grounded factual lookup and return its state delta."""

    started_at = time.monotonic()
    grounded_lookup = state.get("grounded_lookup", {}) or {}
    query = str(grounded_lookup.get("query") or "").strip()
    if not query:
        raise ValueError("grounded lookup requires grounded_lookup.query.")
    llm_client = context.llm_client

    if llm_client is None:
        raise RuntimeError("grounded lookup requires an LLM client.")

    answer, status = await answer_factual_lookup(
        state,
        llm_client=llm_client,
        query=query,
    )
    if answer:
        return grounded_lookup_response_delta(
            answer,
            status=status,
            started_at=started_at,
        )
    text = "I couldn't verify that from reliable sources, so I don't want to guess."
    return grounded_lookup_response_delta(
        text,
        status="no_verified_answer",
        started_at=started_at,
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

    answer, status = await answer_factual_lookup_request(
        GroundedLookupRequest(
            query=text,
            current_user_message=context.current_user_message,
            transcript=tuple(context.transcript),
        ),
        llm_client=llm_client,
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


__all__ = [
    "GroundedLookupToolResult",
    "answer_grounded_lookup",
    "build_grounded_lookup_delta",
    "build_grounded_lookup_tools",
    "execute_grounded_lookup_tool",
    "grounded_lookup_response_delta",
]
