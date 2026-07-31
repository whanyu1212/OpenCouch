"""OpenAI Agents SDK crisis tools for specialist-owned crisis replies."""

from __future__ import annotations

from typing import Any, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.audit.models import CrisisResourceLookupStatus
from agent.guardrails.crisis_response import (
    CrisisSupportRiskLevel,
    build_crisis_response_plan,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_search import (
    CrisisResourceLookupRequest,
    find_crisis_resources,
    find_crisis_resources_for_request,
)


class CrisisResourceLookupToolResult(BaseModel):
    """Structured result returned by crisis-resource lookup tools."""

    response_text: str = Field(
        description="Crisis-resource guidance for the specialist response agent."
    )
    inferred_location: str = Field(
        default="",
        description="User-stated location used for resource lookup, when available.",
    )
    found_resources: list[dict[str, str]] = Field(
        default_factory=list,
        description="Verified crisis-resource rows from official/reputable sources.",
    )
    resource_lookup_status: CrisisResourceLookupStatus = Field(
        description="Whether local crisis resources were found or why not."
    )
    side_effect: str = Field(
        default="none",
        description="Crisis-resource lookup does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the lookup can duplicate side effects.",
    )


class CrisisSupportTemplateToolResult(BaseModel):
    """Deterministic crisis-response scaffold for specialist-owned replies."""

    risk_level: CrisisSupportRiskLevel = Field(
        description="Runtime-classified crisis risk level for this template."
    )
    opening: str = Field(description="Short, calm opening for the response.")
    validation: str = Field(description="Brief validation without overpromising.")
    immediate_safety_step: str = Field(
        description="Concrete immediate safety step appropriate to the risk level."
    )
    resource_guidance: str = Field(
        description="Resource guidance using only verified resources when supplied."
    )
    one_question: str = Field(
        description="At most one follow-up question for the crisis response."
    )
    avoid: list[str] = Field(
        description="Safety-critical response patterns the agent must avoid."
    )
    response_text: str = Field(
        description="Prompt-ready scaffold assembled from the structured fields."
    )
    side_effect: str = Field(
        default="none",
        description="Template loading does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the template load can duplicate side effects.",
    )


async def build_crisis_resource_lookup_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Resolve local crisis-resource state for the current crisis turn."""

    llm_client = context.llm_client
    if llm_client is None:
        raise RuntimeError("crisis_resource_lookup requires an LLM client.")

    (
        inferred_location,
        found_resources,
        resource_lookup_status,
    ) = await find_crisis_resources(state, llm_client=llm_client)

    return {
        "inferred_location": inferred_location,
        "found_resources": found_resources,
        "resource_lookup_status": resource_lookup_status,
    }


def crisis_response_delta(response_text: str) -> dict[str, Any]:
    """Return the shared response delta for crisis-response turns."""

    return {
        "route": "crisis",
        "response_style": "crisis_response",
        "response_text": response_text,
    }


async def execute_crisis_resource_lookup_tool(
    context: OpenAITextRunContext,
) -> CrisisResourceLookupToolResult:
    """Execute crisis-resource lookup through the existing grounded service."""

    llm_client = context.workflow_context.llm_client
    if llm_client is None:
        raise RuntimeError("lookup_crisis_resources requires an LLM client.")

    (
        inferred_location,
        found_resources,
        status,
    ) = await find_crisis_resources_for_request(
        CrisisResourceLookupRequest(
            current_user_message=context.current_user_message,
            transcript=tuple(context.transcript),
        ),
        llm_client=llm_client,
    )
    result = CrisisResourceLookupToolResult(
        response_text=build_crisis_response_plan(
            inferred_location=inferred_location,
            found_resources=found_resources,
            resource_lookup_status=cast(CrisisResourceLookupStatus, status),
        ).resource_guidance,
        inferred_location=inferred_location,
        found_resources=found_resources,
        resource_lookup_status=cast(CrisisResourceLookupStatus, status),
    )
    context.record_crisis_resource_tool_result(
        response_text=result.response_text,
        inferred_location=result.inferred_location,
        found_resources=result.found_resources,
        resource_lookup_status=result.resource_lookup_status,
    )
    return result


@function_tool(
    name_override="lookup_crisis_resources",
    description_override=(
        "Look up verified local crisis resources for the current crisis turn "
        "using the user's stated location when available. Use only for "
        "runtime-selected level 2/3 crisis response turns, not for level 1 "
        "safety clarification. Side effects: none. Retry safety: safe."
    ),
)
async def lookup_crisis_resources(
    wrapper: RunContextWrapper[OpenAITextRunContext],
) -> CrisisResourceLookupToolResult:
    """Look up crisis resources for one app-classified crisis response."""

    return await execute_crisis_resource_lookup_tool(wrapper.context)


async def execute_crisis_support_template_tool(
    *,
    risk_level: str,
    crisis_level: int | None = None,
    inferred_location: str = "",
    found_resources: list[dict[str, str]] | None = None,
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted",
) -> CrisisSupportTemplateToolResult:
    """Return a deterministic safety scaffold for the crisis specialist."""

    plan = build_crisis_response_plan(
        crisis_level=crisis_level,
        requested_risk_level=risk_level,
        inferred_location=inferred_location,
        found_resources=found_resources,
        resource_lookup_status=resource_lookup_status,
    )
    response_text = "\n\n".join(
        [
            f"Opening: {plan.opening}",
            f"Validation: {plan.validation}",
            f"Immediate safety step: {plan.immediate_safety_step}",
            f"Resource guidance: {plan.resource_guidance}",
            f"Ask one question: {plan.one_question}",
            "Avoid:\n" + "\n".join(f"- {item}" for item in plan.avoid),
        ]
    )
    return CrisisSupportTemplateToolResult(
        risk_level=plan.risk_level,
        opening=plan.opening,
        validation=plan.validation,
        immediate_safety_step=plan.immediate_safety_step,
        resource_guidance=plan.resource_guidance,
        one_question=plan.one_question,
        avoid=list(plan.avoid),
        response_text=response_text,
    )


@function_tool(
    name_override="get_crisis_support_template",
    description_override=(
        "Load a deterministic crisis-response safety scaffold for the current "
        "specialist reply. Use this to structure level 2/3 crisis responses. "
        "It does not replace crisis-resource lookup and must not be used to "
        "invent phone numbers. Parameters: risk_level is moderate, high, or "
        "imminent; optional resource fields must come from verified lookup "
        "results. Side effects: none. Retry safety: safe."
    ),
)
async def get_crisis_support_template(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    risk_level: str,
    inferred_location: str = "",
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted",
    resource_name: str = "",
    resource_phone: str = "",
    resource_url: str = "",
    resource_region: str = "",
) -> CrisisSupportTemplateToolResult:
    """Load a deterministic crisis-response scaffold."""

    found_resources = _single_resource_from_tool_args(
        name=resource_name,
        phone=resource_phone,
        url=resource_url,
        region=resource_region,
    )
    if not found_resources:
        latest_lookup = wrapper.context.latest_crisis_resource_tool_result()
        if latest_lookup is not None:
            found_resources = latest_lookup.found_resources
            if not inferred_location:
                inferred_location = latest_lookup.inferred_location
            if resource_lookup_status == "not_attempted":
                resource_lookup_status = latest_lookup.resource_lookup_status

    return await execute_crisis_support_template_tool(
        risk_level=risk_level,
        crisis_level=_crisis_level_from_state(wrapper.context.agent_state),
        inferred_location=inferred_location,
        found_resources=found_resources,
        resource_lookup_status=resource_lookup_status,
    )


def build_crisis_response_tools() -> list[Any]:
    """Return crisis tools for the OpenAI crisis response specialist."""

    return [lookup_crisis_resources, get_crisis_support_template]


def _single_resource_from_tool_args(
    *,
    name: str,
    phone: str,
    url: str,
    region: str,
) -> list[dict[str, str]]:
    resource = {
        "name": str(name or "").strip(),
        "phone": str(phone or "").strip(),
        "url": str(url or "").strip(),
        "region": str(region or "").strip(),
    }
    if not any(resource.values()):
        return []
    return [resource]


def _crisis_level_from_state(state: AgentState | None) -> int | None:
    crisis = (state or {}).get("crisis")
    level = getattr(crisis, "level", None)
    return int(level) if isinstance(level, int) else None


__all__ = [
    "CrisisResourceLookupToolResult",
    "CrisisSupportTemplateToolResult",
    "build_crisis_resource_lookup_delta",
    "build_crisis_response_tools",
    "crisis_response_delta",
    "execute_crisis_resource_lookup_tool",
    "execute_crisis_support_template_tool",
    "get_crisis_support_template",
    "lookup_crisis_resources",
]
