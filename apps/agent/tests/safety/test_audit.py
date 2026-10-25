from datetime import UTC, datetime

from opencouch_agent.safety import FailureReason, GateDecision, RiskLevel
from opencouch_agent.safety.audit import (
    STORED_LEVEL_KEPT,
    AuditEntry,
    InMemoryCrisisAuditLog,
    audit_entry,
)
from opencouch_agent.safety.risk import RiskChange, RiskState

AT = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
ELEVATED = RiskState(level=RiskLevel.ELEVATED, peak=RiskLevel.ELEVATED, turns=2)
IN_CRISIS = RiskState(
    level=RiskLevel.CRISIS,
    peak=RiskLevel.CRISIS,
    turns=3,
    changes=(RiskChange(3, RiskLevel.ELEVATED, RiskLevel.CRISIS, "CRISIS: intent"),),
)


def test_a_rise_is_audited_with_the_rule_that_raised_it() -> None:
    decision = GateDecision(risk=IN_CRISIS, path="crisis")

    entry = audit_entry("thread-1", ELEVATED, decision, at=AT)

    assert entry == AuditEntry(
        thread_id="thread-1",
        at=AT,
        level_before=RiskLevel.ELEVATED,
        level_after=RiskLevel.CRISIS,
        reason="CRISIS: intent",
    )


def test_a_step_down_from_crisis_is_audited() -> None:
    step_down = RiskChange(
        4, RiskLevel.CRISIS, RiskLevel.ELEVATED, "the user tapped safe on the check-in"
    )
    stepped_down = RiskState(
        level=RiskLevel.ELEVATED,
        peak=RiskLevel.CRISIS,
        turns=4,
        changes=(*IN_CRISIS.changes, step_down),
    )
    decision = GateDecision(risk=stepped_down, path="companion")

    entry = audit_entry("thread-1", IN_CRISIS, decision, at=AT)

    assert entry is not None
    assert (entry.level_before, entry.level_after) == (
        RiskLevel.CRISIS,
        RiskLevel.ELEVATED,
    )
    assert entry.reason == "the user tapped safe on the check-in"


def test_a_failed_screening_is_audited_with_its_reason_and_the_stored_level() -> None:
    decision = GateDecision(
        risk=ELEVATED, path="crisis", screening_failure=FailureReason.TIMEOUT
    )

    entry = audit_entry("thread-1", ELEVATED, decision, at=AT)

    assert entry == AuditEntry(
        thread_id="thread-1",
        at=AT,
        level_before=RiskLevel.ELEVATED,
        level_after=RiskLevel.ELEVATED,
        reason=STORED_LEVEL_KEPT,
        screening_failure=FailureReason.TIMEOUT,
    )


def test_a_screened_turn_that_keeps_the_level_is_not_audited() -> None:
    decision = GateDecision(risk=IN_CRISIS, path="crisis")

    assert audit_entry("thread-1", IN_CRISIS, decision, at=AT) is None


async def test_the_in_memory_log_keeps_entries_in_order() -> None:
    log = InMemoryCrisisAuditLog()
    first = AuditEntry("thread-1", AT, RiskLevel.NONE, RiskLevel.CRISIS, "first")
    second = AuditEntry("thread-2", AT, RiskLevel.NONE, RiskLevel.WATCH, "second")

    await log.append(first)
    await log.append(second)

    assert log.entries == (first, second)
