"""Requests for early access from people not yet invited (HY-353).

Only an email address and when it was asked for are kept, so the beta's owner
can invite people by hand. No account is made.
"""

from typing import Final, Literal, Protocol

from pydantic import BaseModel

from opencouch_agent.beta.invites import normalized_email

# The longest address SMTP allows.
MAX_EMAIL_LENGTH: Final = 254


class AccessRequestBody(BaseModel):
    """What the request-access form sends."""

    email: str


class AccessRequestReceived(BaseModel):
    """The reply to every valid request: new, repeated or already invited."""

    status: Literal["received"] = "received"


class AccessRequests(Protocol):
    """Where requests for early access are kept."""

    async def add(self, email: str) -> None:
        """Record a request; asking again changes nothing."""
        ...


class InMemoryAccessRequests:
    """Requests kept in a set. For tests and local runs only."""

    def __init__(self) -> None:
        self.emails: set[str] = set()

    async def add(self, email: str) -> None:
        """Record a request; see `AccessRequests.add`."""
        self.emails.add(email)


def requestable_email(raw: str) -> str | None:
    """The normalised address to record, or None if it can't be an address.

    Only a sanity check: whether it really exists shows when it's invited.
    """
    email = normalized_email(raw)
    local, at, domain = email.partition("@")
    labels = domain.split(".")
    looks_like_address = (
        at == "@"
        and bool(local)
        and len(labels) > 1
        and all(labels)
        and len(labels[-1]) > 1
        and "@" not in domain
        and not any(char.isspace() for char in email)
    )
    if not looks_like_address or len(email) > MAX_EMAIL_LENGTH:
        return None
    return email
