"""Shared AG-UI state exchanged between the agent and the UI."""

from pydantic import BaseModel, Field


class CompanionState(BaseModel):
    """Placeholder for the shared state schema.

    Designing this schema (session mode, active exercise and step, pending
    memory approvals) is the first task in #356. Anything the client sends
    back here is untrusted user input: never let it lower a crisis level or
    skip a safety step.
    """

    crisis_resources: list[str] = Field(default_factory=list)
    """Catalog keys for the crisis resource card, in the order to show them
    (HY-230). The server sets them on a crisis turn and clears them on every
    other turn; empty means no card."""
