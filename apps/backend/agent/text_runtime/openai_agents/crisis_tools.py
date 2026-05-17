"""OpenAI Agents SDK crisis tools for specialist-owned crisis replies."""

from __future__ import annotations

from typing import Any, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.text_runtime.openai_agents.context import (
    CrisisResourceToolStatus,
    OpenAITextRunContext,
)
from agent.tools.grounded_search import (
    CrisisResourceLookupRequest,
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
    resource_lookup_status: CrisisResourceToolStatus = Field(
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
            status=cast(CrisisResourceToolStatus, status),
        ),
        inferred_location=inferred_location,
        found_resources=found_resources,
        resource_lookup_status=cast(CrisisResourceToolStatus, status),
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


def build_crisis_response_tools() -> list[Any]:
    """Return crisis tools for the OpenAI crisis response specialist."""

    return [lookup_crisis_resources]


def _resource_lookup_response_text(
    *,
    inferred_location: str,
    found_resources: list[dict[str, str]],
    status: CrisisResourceToolStatus,
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
    "build_crisis_response_tools",
    "execute_crisis_resource_lookup_tool",
    "lookup_crisis_resources",
]
