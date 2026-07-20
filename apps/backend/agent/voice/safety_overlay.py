"""Product policy and public projections for the Realtime safety overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from agent.audit.models import CrisisResourceLookupStatus
from agent.tools.crisis import crisis_support_template_parts
from agent.voice.concurrent_safety import VoiceConcurrentSafetyResult

VoiceSafetyAction = Literal["continue", "interrupt"]
VoiceSafetyRiskLevel = Literal[2, 3]
VoiceSafetyResourceStatus = Literal[
    "found",
    "no_location",
    "location_refused",
    "no_verified_results",
    "lookup_error",
]


@dataclass(frozen=True, slots=True)
class VoiceSafetySupport:
    """Public-safe deterministic guidance for an interrupted voice turn."""

    headline: str
    validation: str
    immediate_step: str


@dataclass(frozen=True, slots=True)
class VoiceSafetyDecision:
    """Server-owned playback decision derived from a trusted assessment."""

    action: VoiceSafetyAction
    risk_level: VoiceSafetyRiskLevel | None = None
    support: VoiceSafetySupport | None = None


@dataclass(frozen=True, slots=True)
class VoiceSafetyResourceResolution:
    """Public-safe result of a bounded, non-mutating resource lookup."""

    status: VoiceSafetyResourceStatus
    inferred_location: str
    resources: list[dict[str, str]]
    message: str


class VoiceSafetyOverlayService:
    """Own safety-overlay policy independently of HTTP and runtime plumbing."""

    def decide(self, result: VoiceConcurrentSafetyResult) -> VoiceSafetyDecision:
        """Interrupt only for a completed, high-confidence level 2/3 crisis."""

        assessment = result.assessment
        should_interrupt = (
            result.status == "completed"
            and assessment is not None
            and assessment.confidence == "high"
            and assessment.level >= 2
            and assessment.needs_crisis_response
        )
        if not should_interrupt or assessment is None:
            return VoiceSafetyDecision(action="continue")

        risk_level: VoiceSafetyRiskLevel = 3 if assessment.level == 3 else 2
        opening, validation, immediate_step, _ = crisis_support_template_parts(
            "imminent" if risk_level == 3 else "moderate"
        )
        return VoiceSafetyDecision(
            action="interrupt",
            risk_level=risk_level,
            support=VoiceSafetySupport(
                headline=opening,
                validation=validation,
                immediate_step=immediate_step,
            ),
        )

    def resource_resolution(
        self,
        *,
        inferred_location: str,
        resources: list[dict[str, str]],
        status: CrisisResourceLookupStatus,
    ) -> VoiceSafetyResourceResolution:
        """Project verified lookup data without prompt-only guidance fields."""

        public_status: VoiceSafetyResourceStatus = (
            status if status != "not_attempted" else "lookup_error"
        )
        verified_resources: list[dict[str, str]] = []
        for resource in resources:
            name = str(resource.get("name") or "").strip()
            phone = str(resource.get("phone") or "").strip()
            url = str(resource.get("url") or "").strip()
            parsed_url = urlsplit(url)
            if (
                not name
                or not any(character.isdigit() for character in phone)
                or parsed_url.scheme != "https"
                or not parsed_url.hostname
            ):
                continue
            verified_resources.append(
                {
                    "name": name,
                    "phone": phone,
                    "url": url,
                    "region": str(resource.get("region") or "").strip(),
                }
            )
        if public_status == "found" and not verified_resources:
            public_status = "no_verified_results"
        return VoiceSafetyResourceResolution(
            status=public_status,
            inferred_location=inferred_location.strip(),
            resources=verified_resources if public_status == "found" else [],
            message=_resource_message(public_status),
        )


def _resource_message(status: VoiceSafetyResourceStatus) -> str:
    if status == "found":
        return "Verified crisis resources are available below."
    if status == "location_refused":
        return (
            "Location-based help was not requested. If you may act soon, contact "
            "local emergency services or go to the nearest emergency department."
        )
    if status == "no_location":
        return (
            "Location-specific contacts could not be checked without a country or "
            "region. If you may act soon, contact local emergency services or go "
            "to the nearest emergency department."
        )
    if status == "no_verified_results":
        return (
            "No local crisis contact could be verified. If you may act soon, "
            "contact local emergency services or go to the nearest emergency "
            "department."
        )
    return (
        "Local crisis resources could not be checked right now. If you may act "
        "soon, contact local emergency services or go to the nearest emergency "
        "department."
    )


__all__ = [
    "VoiceSafetyAction",
    "VoiceSafetyDecision",
    "VoiceSafetyOverlayService",
    "VoiceSafetyResourceResolution",
    "VoiceSafetyResourceStatus",
    "VoiceSafetyRiskLevel",
    "VoiceSafetySupport",
]
