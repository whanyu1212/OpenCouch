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


def _clean_field(raw_field: str) -> str:
    """Normalize one pipe-separated field from the LLM's resource output.

    Handles the formatting variations we see in the wild across providers:

    - Markdown bold (``**Samaritans**`` → ``Samaritans``). OpenAI's
      web_search tool emits markdown-formatted bold around field
      values. Phone fields sometimes contain multiple ``**``-wrapped
      numbers (``**0120-A**; **050-B** for IP phones``), so we strip
      ALL ``**`` sequences from the field rather than just the ends.
      There is no legitimate reason to keep markdown bold in a
      structured phone/name/url field.
    - Leading/trailing whitespace.

    Returns the cleaned field; empty string if the field was empty or
    contained only formatting characters.
    """

    # Strip markdown bold markers anywhere in the field, then collapse
    # any whitespace runs created by the removal.
    value = raw_field.replace("**", "")
    return " ".join(value.split())


def _clean_url_field(raw_field: str) -> str:
    """Normalize the URL field, stripping OpenAI-style citation suffixes.

    OpenAI's web_search tool appends a citation tag after the URL in
    the form ``URL ([source.domain](url?utm_source=openai))``. The
    extra syntax confuses downstream URL rendering without adding
    information the CLI can use (the real URL is already the leading
    value). Strip everything from the first space-then-paren onward.

    Also handles markdown bold the same way as :func:`_clean_field`.

    Ordering note: the citation tail is stripped FIRST, then the
    markdown-bold cleanup runs. Reversing the order leaves inner
    ``**`` sequences behind when the URL is wrapped in bold
    (``**URL** ([citation]...)``), because the leading ``**`` gets
    stripped but the trailing ``**`` is no longer at the end of the
    string after the citation is cut.
    """

    # Step 1: strip the citation tail. Look for " (" — a space then
    # an open paren — which is the OpenAI citation-prefix boundary.
    # Non-OpenAI output without this pattern is unaffected because
    # ``find`` returns -1.
    value = raw_field.strip()
    citation_start = value.find(" (")
    if citation_start >= 0:
        value = value[:citation_start].rstrip()

    # Step 2: strip markdown-bold markers from both ends of the
    # now-citation-free value.
    return _clean_field(value)


def _parse_resource_lines(raw: str, *, location: str) -> list[dict[str, str]]:
    """Parse ``Name | Phone | Website`` lines into structured resource dicts.

    Lines that don't contain a ``|`` separator or have fewer than two
    fields are silently dropped. Caps results at ``_MAX_RESOURCES``.

    Handles two LLM output formats:
    - Gemini's search-grounded output (plain pipe-separated lines)
    - OpenAI's search-grounded output (markdown-bold fields with
      citation suffixes, e.g., ``**Name** | **Phone** |
      **URL** ([source.domain](url?utm_source=openai))``)

    Both formats parse to the same normalized dict shape. See the
    ``_clean_field`` and ``_clean_url_field`` helpers for the
    formatting normalizations.
    """
    resources: list[dict[str, str]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip(_BULLET_CHARS)
        if "|" not in line:
            continue
        # Markdown tables bracket each row with a leading and trailing
        # pipe: ``| Name | Phone | Website |``. If we split that on
        # ``|`` we get empty strings at both ends. Strip them so the
        # real columns land at indexes 0/1/2 as the plain format
        # expects. This is a no-op for non-table output.
        line = line.strip("|").strip()
        raw_parts = line.split("|")
        if len(raw_parts) < 2:
            continue
        name = _clean_field(raw_parts[0])
        phone = _clean_field(raw_parts[1])
        url = _clean_url_field(raw_parts[2]) if len(raw_parts) > 2 else ""
        # Skip header rows like ``Name | Phone | Website`` and
        # markdown table separators like ``--- | --- | ---``. Check
        # both the name and phone fields because either can give us
        # away. Also reject "no phone" placeholders that the model
        # sometimes emits when it couldn't find a number — those
        # rows are informational noise, not actionable resources.
        phone_lower = phone.lower()
        name_lower = name.lower()
        if name_lower in ("name", "---"):
            continue
        if phone_lower in ("phone", "---", "number") or not phone:
            continue
        # Skip rows where the model explicitly notes the number
        # couldn't be verified — these confuse the CLI's resource
        # display because there's nothing to dial.
        if "not verified" in phone_lower or "no phone" in phone_lower:
            continue
        resources.append(
            {
                "name": name or "Crisis Line",
                "phone": phone,
                "url": url,
                "region": location,
            }
        )
        if len(resources) >= _MAX_RESOURCES:
            break
    return resources
