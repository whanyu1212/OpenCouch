"""Framework-neutral policy for one crisis response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.audit.models import CrisisResourceLookupStatus


CrisisSupportRiskLevel = Literal["moderate", "high", "imminent"]

CRISIS_RESPONSE_AVOID = (
    "Do not diagnose the user or make clinical certainty claims.",
    "Do not promise confidentiality or that everything will be okay.",
    "Do not claim OpenCouch has contacted emergency services or another person.",
    "Do not invent phone numbers, URLs, or local crisis resources.",
    "Do not give instructions involving self-harm methods or lethal means.",
)


@dataclass(frozen=True, slots=True)
class CrisisResponsePlan:
    """Safety facts shared by text and voice crisis-response renderers."""

    risk_level: CrisisSupportRiskLevel
    opening: str
    validation: str
    immediate_safety_step: str
    one_question: str
    resource_guidance: str
    text_resource_guidance: str
    max_follow_up_questions: int
    location_question_permitted: bool
    avoid: tuple[str, ...]


def build_crisis_response_plan(
    *,
    crisis_level: int | None = None,
    requested_risk_level: str = "high",
    inferred_location: str = "",
    found_resources: list[dict[str, str]] | None = None,
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted",
) -> CrisisResponsePlan:
    """Derive shared crisis policy from trusted assessment and lookup state."""

    risk_level = _risk_level_for(crisis_level, requested_risk_level)
    resources = [dict(resource) for resource in found_resources or []]
    opening, validation, immediate_safety_step, one_question = _support_parts(
        risk_level
    )
    location_question_permitted = (
        not resources
        and resource_lookup_status in {"no_location", "not_attempted"}
        and risk_level != "imminent"
    )
    return CrisisResponsePlan(
        risk_level=risk_level,
        opening=opening,
        validation=validation,
        immediate_safety_step=immediate_safety_step,
        one_question=one_question,
        resource_guidance=_resource_guidance(
            inferred_location=inferred_location,
            found_resources=resources,
            status=resource_lookup_status,
        ),
        text_resource_guidance=_text_resource_guidance(
            inferred_location=inferred_location,
            found_resources=resources,
            status=resource_lookup_status,
            location_question_permitted=location_question_permitted,
        ),
        max_follow_up_questions=1,
        location_question_permitted=location_question_permitted,
        avoid=CRISIS_RESPONSE_AVOID,
    )


def _risk_level_for(
    crisis_level: int | None, requested_risk_level: str
) -> CrisisSupportRiskLevel:
    if crisis_level is not None:
        if crisis_level >= 3:
            return "imminent"
        if crisis_level >= 2:
            return "moderate"
    value = " ".join(str(requested_risk_level or "").strip().lower().split())
    if value in {"moderate", "level 2", "2"}:
        return "moderate"
    if value in {"imminent", "level 3", "3"}:
        return "imminent"
    return "high"


def _support_parts(risk_level: CrisisSupportRiskLevel) -> tuple[str, str, str, str]:
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


def _resource_guidance(
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


def _text_resource_guidance(
    *,
    inferred_location: str,
    found_resources: list[dict[str, str]],
    status: CrisisResourceLookupStatus,
    location_question_permitted: bool,
) -> str:
    if found_resources:
        location_label = inferred_location or "the user's region"
        resources = "\n".join(_format_resource_row(row) for row in found_resources)
        return (
            f"\nVerified local crisis resources for {location_label}:\n"
            f"{resources}\n"
            "Include at least one of these specific resources in your response. "
            "Do not modify phone numbers. Only include phone numbers that appear "
            "in the verified resources block above. You may say local emergency "
            "services, but do not name or number them unless they appear above "
            "as a verified resource.\n"
        )
    if status == "location_refused":
        return (
            "\nThe user has explicitly declined location-based help. Respect "
            "that boundary without mentioning location sharing again. Give immediate "
            "safety guidance that does not require location: contact local emergency "
            "services if they might act soon, go to the nearest emergency department "
            "if they can do so safely, move away from means, and contact a trusted "
            "person nearby. Do not invent phone numbers. Keep the response focused "
            "on these safety steps.\n"
        )
    if status == "no_location":
        location_instruction = (
            "Ask once, optionally, for their country or region only if they are "
            "comfortable sharing it so local resources can be looked up. Do not "
            "pressure them for location."
            if location_question_permitted
            else "Do not ask for location in this response; prioritize emergency "
            "services, moving away from means, and asking someone nearby to stay "
            "with you."
        )
        return (
            "\nThe user has not stated their location. Give immediate safety "
            "guidance that does not require location: local emergency services, "
            "the nearest emergency department, moving away from means, and "
            f"asking someone nearby to stay with you. {location_instruction} "
            "Do not invent phone numbers.\n"
        )
    if status == "no_verified_results":
        location_label = inferred_location or "the user's stated region"
        return (
            f"\nThe user gave this location: {location_label}. No verified, "
            "actionable local crisis line was found. Give immediate safety guidance "
            "using local emergency services, the nearest emergency department, "
            "moving away from means, and contacting a trusted person nearby. Briefly "
            "state that you could not verify a local crisis line for that region. "
            "Keep this especially brief: 2-4 short sentences, and do not ask a "
            "follow-up question unless it is needed for an immediate safety step. "
            "Do not invent phone numbers.\n"
        )
    if status == "lookup_error":
        return (
            "\nLooking up local crisis resources failed due to a temporary issue, "
            "so none could be verified this turn. Do not claim a lookup was completed "
            "or that no resources exist. Give immediate safety guidance using local "
            "emergency services, the nearest emergency department, moving away from "
            "means, and contacting a trusted person nearby. Keep this especially "
            "brief: 2-4 short sentences, and do not ask a follow-up question unless "
            "it is needed for an immediate safety step. Do not invent phone numbers.\n"
        )
    return (
        "\nNo verified local resources were found. If the user is willing, ask "
        "for their country or region so the right local crisis line can be looked "
        "up. Do not invent phone numbers.\n"
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
    "CRISIS_RESPONSE_AVOID",
    "CrisisResponsePlan",
    "CrisisSupportRiskLevel",
    "build_crisis_response_plan",
]
