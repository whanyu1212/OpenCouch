"""The crisis audit log: every turn that changed the risk level or went unscreened.

Append-only and kept apart from the conversation, so it outlives a deleted or
incognito thread. Safety: an entry holds levels and the rule's reason, never
anything the user or the agent wrote.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from opencouch_agent.safety.gate import GateDecision
from opencouch_agent.safety.jev import FailureReason
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import RiskState

STORED_LEVEL_KEPT: Final = "screening failed: kept the stored level"


@dataclass(frozen=True)
class AuditEntry:
    """One audited turn.

    Attributes:
        thread_id: The conversation the turn belongs to.
        at: When the turn arrived.
        level_before: The conversation's level before the turn.
        level_after: Its level after the turn.
        reason: What decided `level_after`: the risk rule that changed it, or
            `STORED_LEVEL_KEPT` when screening failed.
        screening_failure: Why screening didn't run, if it didn't.
    """

    thread_id: str
    at: datetime
    level_before: RiskLevel
    level_after: RiskLevel
    reason: str
    screening_failure: FailureReason | None = None


class CrisisAuditLog(Protocol):
    """Append-only storage for audit entries."""

    async def append(self, entry: AuditEntry) -> None:
        """Add one entry to the log."""
        ...


class InMemoryCrisisAuditLog:
    """Audit log backed by a list. For tests and local runs only."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        """Every entry, oldest first."""
        return tuple(self._entries)

    async def append(self, entry: AuditEntry) -> None:
        """Add one entry to the log."""
        self._entries.append(entry)


def audit_entry(
    thread_id: str, before: RiskState, decision: GateDecision, *, at: datetime
) -> AuditEntry | None:
    """The entry for a screened turn, or None if the turn needs none.

    A turn is audited when screening failed or the level changed, down as
    well as up: a step-down from CRISIS is a safety decision too.
    """
    if decision.screening_failure is not None:
        return AuditEntry(
            thread_id=thread_id,
            at=at,
            level_before=before.level,
            level_after=decision.risk.level,
            reason=STORED_LEVEL_KEPT,
            screening_failure=decision.screening_failure,
        )
    if decision.risk.level is before.level:
        return None
    return AuditEntry(
        thread_id=thread_id,
        at=at,
        level_before=before.level,
        level_after=decision.risk.level,
        reason=decision.risk.changes[-1].reason,
    )
