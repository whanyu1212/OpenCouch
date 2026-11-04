"""Shared AG-UI state exchanged between the agent and the UI."""

from datetime import datetime

from pydantic import BaseModel, Field


class CompanionState(BaseModel):
    """Placeholder for the shared state schema.

    Designing this schema (session mode, active exercise and step, pending
    memory approvals) is the first task in #356. Anything the client sends
    back here is untrusted user input: never let it lower a crisis level or
    skip a safety step.
    """

    crisis_resources: list[str] = Field(default_factory=list)
    """Currently visible vetted services, in display order."""
    crisis_resources_until: datetime | None = None
    """When the pinned card expires; None during a crisis."""
    get_help_resources: list[str] = Field(default_factory=list)
    """Services available from Get help for the rest of this conversation."""
    crisis_check_in: str | None = None
    """The server's current check-in offer ID, never an instruction to step down."""
