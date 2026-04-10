"""Crisis resource lookup tool using provider-native search grounding.

Looks up regional crisis hotlines when a user discloses their location during
a crisis conversation. Falls back gracefully to an empty list on any error so
the crisis response node can always produce a safe reply.
"""

from __future__ import annotations

import logging

from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# Maximum number of resources we keep from a single search response. Caps both
# prompt size for downstream nodes and the surface area of any noisy results.
_MAX_RESOURCES = 5
# Characters stripped from the start/end of each parsed line — covers common
# bullet styles LLMs emit when listing items.
_BULLET_CHARS = " -•*"

_LOCATION_EXTRACTION_SYSTEM = (
    "You extract location information from mental health support conversations. "
    "Return only the user's stated country, region, or city as a short plain-text string. "
    "If no location is mentioned, return the empty string."
)

_RESOURCE_LOOKUP_SYSTEM = (
    "You are a factual assistant helping to find official crisis support resources. "
    "Use your web search capability to look up verified hotlines and services. "
    "Respond only with the search-grounded results — never invent phone numbers. "
    "If you cannot find verified results, say so clearly."
)


async def find_local_crisis_resources(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> tuple[str, list[dict[str, str]]]:
    """Find verified crisis hotlines local to whatever location the user mentioned.

    The only public entry point of the module. Chains a cheap location-extraction
    call with a search-grounded resource lookup, returning both so callers can
    persist them in state for observability.

    Returns:
        A ``(location, resources)`` tuple. Either side may be empty if
        extraction or search fails — callers should treat empty results as a
        signal to fall back to deterministic copy.
    """
    location = await _extract_location(state, llm_client=llm_client)
    if not location:
        return "", []
    resources = await _lookup_resources(location, llm_client=llm_client)
    return location, resources


async def _extract_location(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> str:
    """Ask the LLM to pull a location string out of the user's recent turns."""
    message = state.get("message", "")
    history = state.get("history", [])[-4:]
    history_text = "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in history
    )
    prompt = (
        "Extract the user's location from the conversation below. "
        "Return only the location name (city, region, or country). "
        "If no location is mentioned, return an empty string.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Current user message:\nuser: {message}"
    )
    try:
        raw = await llm_client.generate_text(
            prompt=prompt,
            system_instruction=_LOCATION_EXTRACTION_SYSTEM,
            temperature=0,
        )
    except Exception:
        logger.warning("Location extraction failed; proceeding without location.")
        return ""
    return raw.strip()


async def _lookup_resources(
    location: str,
    *,
    llm_client: BaseLLMClient,
) -> list[dict[str, str]]:
    """Use search-grounded generation to find verified hotlines for ``location``."""
    prompt = (
        f"Find official, verified 24/7 mental health crisis hotlines and emergency "
        f"services for someone in {location}. "
        "List each resource as: Name | Phone | Website. "
        "Only include resources you can verify from official government or recognised "
        "charity sources. Do not invent or guess phone numbers."
    )
    try:
        raw = await llm_client.generate_text(
            prompt=prompt,
            system_instruction=_RESOURCE_LOOKUP_SYSTEM,
            temperature=0,
            use_search=True,
        )
    except Exception:
        logger.warning(
            "Crisis resource search failed for location=%r; using empty fallback.",
            location,
        )
        return []
    return _parse_resource_lines(raw, location=location)


def _parse_resource_lines(raw: str, *, location: str) -> list[dict[str, str]]:
    """Parse ``Name | Phone | Website`` lines into structured resource dicts.

    Lines that don't contain a ``|`` separator or have fewer than two fields
    are silently dropped. Caps results at ``_MAX_RESOURCES``.
    """
    resources: list[dict[str, str]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip(_BULLET_CHARS)
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        resources.append(
            {
                "name": parts[0] or "Crisis Line",
                "phone": parts[1],
                "url": parts[2] if len(parts) > 2 else "",
                "region": location,
            }
        )
        if len(resources) >= _MAX_RESOURCES:
            break
    return resources
