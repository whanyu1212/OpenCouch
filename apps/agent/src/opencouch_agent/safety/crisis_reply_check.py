"""Checking a crisis reply before the user sees it (HY-230).

A wrong number in a crisis is the worst mistake this app could make, so the
responder may only name numbers from the vetted catalog. Any other number, or
any link, fails the whole reply, and the fixed fallback is sent instead.
Stripping the bad part out would leave a broken sentence at the worst moment.
"""

import re
from collections.abc import Iterator
from typing import Final, Literal

from opencouch_agent.safety.crisis_resources import COUNTRY_CODES, CrisisCatalog

ReplyProblem = Literal["empty", "unknown_number", "link"]
"""Why a reply failed the check. Logged instead of the reply's text."""

# Three or more digits, possibly split by single spaces, hyphens or dots, with
# an optional leading +. Fail closed: a year or a time also counts as a number.
_NUMBER: Final = re.compile(r"\+?\d(?:[ .\-]?\d){2,}")
_SEPARATOR: Final = re.compile(r"[ .\-]")
# Any web address, with or without a scheme. Fail closed: a missing space
# after a full stop ("now.Please") also reads as one.
_LINK: Final = re.compile(
    r"https?://|www\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b", re.IGNORECASE
)


def allowed_numbers(catalog: CrisisCatalog) -> frozenset[str]:
    """Every catalog number as digits, with and without its country code."""
    return frozenset(_number_forms(catalog))


def reply_problem(text: str, allowed: frozenset[str]) -> ReplyProblem | None:
    """What makes a crisis reply unsafe to send, or None if it may be sent."""
    if not text.strip():
        return "empty"
    if _LINK.search(text):
        return "link"
    for match in _NUMBER.finditer(text):
        if not _is_allowed(match.group(), allowed):
            return "unknown_number"
    return None


def _is_allowed(number: str, allowed: frozenset[str]) -> bool:
    # A short figure after a space isn't part of the number ("1767 24 hours").
    # Nothing else is dropped: a wrong country code or an extension must fail.
    head, _, tail = number.rpartition(" ")
    if head and len(_digits(tail)) < 3:
        number = head
    if _digits(number) in allowed:
        return True
    # Two numbers written side by side ("995 999").
    return all(_digits(group) in allowed for group in _SEPARATOR.split(number))


def _number_forms(catalog: CrisisCatalog) -> Iterator[str]:
    for resource in catalog.resources:
        for channel in resource.channels:
            local = _digits(channel.display)
            # A WhatsApp `number` carries the country code and its display
            # doesn't, so this covers a number written either way.
            yield from (channel.number, local, COUNTRY_CODES[resource.region] + local)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)
