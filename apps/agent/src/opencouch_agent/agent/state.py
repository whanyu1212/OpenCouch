"""Shared AG-UI state exchanged between the agent and the UI."""

from pydantic import BaseModel


class CompanionState(BaseModel):
    """Placeholder for the shared state schema.

    Designing this schema (session mode, active exercise and step, pending
    memory approvals) is the first task in #356. Anything the client sends
    back here is untrusted user input: never let it lower a crisis level or
    skip a safety step.
    """
