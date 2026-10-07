"""Per-run dependencies passed to the companion agent."""

from dataclasses import dataclass, field

from opencouch_agent.agent.state import CompanionState


@dataclass
class CompanionDeps:
    """Run dependencies.

    Must stay a dataclass with a `state` field so the AG-UI adapter can treat
    it as a `StateHandler` and sync shared state with the frontend.
    """

    state: CompanionState = field(default_factory=CompanionState)
