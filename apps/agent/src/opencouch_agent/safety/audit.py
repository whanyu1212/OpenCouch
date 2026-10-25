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
ABANDONED: Final = "gave up rechecking a missed turn: it was never screened"


@dataclass(frozen=True)
class AuditEntry:
    """One audited turn.

    Attributes:
        thread_id: The conversation the turn belongs to.
        at: When the turn arrived.
        level_before: The conversation's level before the turn.
        level_after: Its level after the turn.
        reason: What decided `level_after`: the risk rule that changed it,
            `STORED_LEVEL_KEPT` when screening failed, or `ABANDONED` when a
            missed turn was given up on.
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


def audit_entries(
    thread_id: str, before: RiskState, decision: GateDecision, *, at: datetime
) -> tuple[AuditEntry, ...]:
    """The entries for a screened turn: each change of level, then any failures.

    Changes down are audited as well as up: a step-down from CRISIS is a
    safety decision too. A turn can change the level twice when it also
    rechecks a turn that screening missed, and records each missed turn it
    gives up on.
    """
    entries = [
        AuditEntry(
            thread_id=thread_id,
            at=at,
            level_before=change.from_level,
            level_after=change.to_level,
            reason=change.reason,
        )
        for change in decision.risk.changes[len(before.changes) :]
    ]
    level = decision.risk.level
    entries.extend(
        AuditEntry(
            thread_id=thread_id,
            at=at,
            level_before=level,
            level_after=level,
            reason=ABANDONED,
        )
        for _ in range(decision.abandoned)
    )
    if decision.screening_failure is not None:
        entries.append(
            AuditEntry(
                thread_id=thread_id,
                at=at,
                level_before=level,
                level_after=level,
                reason=STORED_LEVEL_KEPT,
                screening_failure=decision.screening_failure,
            )
        )
    return tuple(entries)
