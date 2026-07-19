"""OpenAI Agents SDK crisis tools for specialist-owned crisis replies."""

from __future__ import annotations

from typing import Any, Literal, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.audit.models import CrisisResourceLookupStatus
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


CrisisSupportRiskLevel = Literal["moderate", "high", "imminent"]


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
        response_text=_resource_lookup_response_text(
            inferred_location=inferred_location,
            found_resources=found_resources,
            status=cast(CrisisResourceLookupStatus, status),
        ),
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
    inferred_location: str = "",
    found_resources: list[dict[str, str]] | None = None,
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted",
) -> CrisisSupportTemplateToolResult:
    """Return a deterministic safety scaffold for the crisis specialist."""

    normalized_risk = _normalize_crisis_support_risk_level(risk_level)
    resources = [dict(resource) for resource in found_resources or []]
    opening, validation, immediate_safety_step, one_question = (
        _crisis_support_template_parts(normalized_risk)
    )
    resource_guidance = _resource_lookup_response_text(
        inferred_location=inferred_location,
        found_resources=resources,
        status=resource_lookup_status,
    )
    avoid = [
        "Do not diagnose the user or make clinical certainty claims.",
        "Do not promise confidentiality or that everything will be okay.",
        "Do not claim OpenCouch has contacted emergency services or another person.",
        "Do not invent phone numbers, URLs, or local crisis resources.",
        "Do not give instructions involving self-harm methods or lethal means.",
    ]
    response_text = "\n\n".join(
        [
            f"Opening: {opening}",
            f"Validation: {validation}",
            f"Immediate safety step: {immediate_safety_step}",
            f"Resource guidance: {resource_guidance}",
            f"Ask one question: {one_question}",
            "Avoid:\n" + "\n".join(f"- {item}" for item in avoid),
        ]
    )
    return CrisisSupportTemplateToolResult(
        risk_level=normalized_risk,
        opening=opening,
        validation=validation,
        immediate_safety_step=immediate_safety_step,
        resource_guidance=resource_guidance,
        one_question=one_question,
        avoid=avoid,
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


def _normalize_crisis_support_risk_level(risk_level: str) -> CrisisSupportRiskLevel:
    value = " ".join(str(risk_level or "").strip().lower().split())
    if value in {"moderate", "level 2", "2"}:
        return "moderate"
    if value in {"imminent", "level 3", "3"}:
        return "imminent"
    return "high"


def _crisis_support_template_parts(
    risk_level: CrisisSupportRiskLevel,
) -> tuple[str, str, str, str]:
    if risk_level == "moderate":
        return (
            "I’m here with you, and I want to help you stay safe right now.",
            "What you’re describing sounds really painful, and you do not have to handle it alone.",
            "Pause and move to a safer place if you can, away from anything you could use to hurt yourself.",
            "Are you somewhere safe enough to keep talking for the next few minutes?",
        )
    if risk_level == "imminent":
        return (
            "Your safety matters most right now.",
            "I’m taking this seriously, and the next step is immediate support from someone nearby or emergency care.",
            "If you might act soon, call local emergency services now, go to the nearest emergency department if safe, or ask a trusted person nearby to stay with you.",
            "Can you contact emergency services or a trusted nearby person right now?",
        )
    return (
        "I’m really glad you said this here.",
        "These feelings can be intense and frightening, and you deserve immediate support.",
        "Please move away from anything you could use to hurt yourself and reach out to someone nearby who can stay with you.",
        "Is there someone nearby you can ask to stay with you while we keep this simple?",
    )


def _resource_lookup_response_text(
    *,
    inferred_location: str,
    found_resources: list[dict[str, str]],
    status: CrisisResourceLookupStatus,
) -> str:
    if found_resources:
        location_label = inferred_location or "the user's region"
        resources = "\n".join(_format_resource_row(row) for row in found_resources)
        return (
            f"Verified local crisis resources for {location_label}:\n"
            f"{resources}\n"
            "Include at least one specific resource above in the response. Do not "
            "modify phone numbers, and do not include phone numbers that are not "
            "listed above."
        )
    if status == "location_refused":
        return (
            "The user declined location-based help. Respect that boundary. Give "
            "immediate safety guidance that does not require location: contact "
            "local emergency services if they might act soon, go to the nearest "
            "emergency department if safe, move away from means, and contact a "
            "trusted person nearby. Do not invent phone numbers."
        )
    if status == "no_location":
        return (
            "The user has not stated their location. Give immediate safety "
            "guidance that does not require location: local emergency services, "
            "nearest emergency department, moving away from means, and asking "
            "someone nearby to stay with them. Do not invent phone numbers."
        )
    if status == "no_verified_results":
        location_label = inferred_location or "the user's stated region"
        return (
            f"The user gave this location: {location_label}. No verified, "
            "actionable local crisis line was found. Give immediate safety "
            "guidance using local emergency services, the nearest emergency "
            "department, moving away from means, and contacting a trusted person "
            "nearby. Briefly state that a local crisis line could not be verified. "
            "Do not invent phone numbers."
        )
    if status == "lookup_error":
        return (
            "Looking up local crisis resources failed due to a temporary issue, "
            "so none could be verified this turn. Do not claim a lookup was "
            "completed or that none exist. Give immediate safety guidance using "
            "local emergency services, the nearest emergency department, moving "
            "away from means, and contacting a trusted person nearby. Do not "
            "invent phone numbers."
        )
    return (
        "No verified local resources were found. Ask once for country or region "
        "only if the user is comfortable sharing it, and do not invent phone "
        "numbers."
    )


def _format_resource_row(resource: dict[str, str]) -> str:
    name = resource.get("name", "Crisis Line")
    phone = resource.get("phone", "")
    url = resource.get("url", "")
    entry = f"- {name}"
    if phone:
        entry += f": {phone}"
    if url:
        entry += f" ({url})"
    return entry


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
