from datetime import UTC, datetime

from opencouch_agent.safety import FailureReason, GateDecision, RiskLevel
from opencouch_agent.safety.audit import (
    ABANDONED,
    STORED_LEVEL_KEPT,
    AuditEntry,
    InMemoryCrisisAuditLog,
    audit_entries,
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

    entries = audit_entries("thread-1", ELEVATED, decision, at=AT)

    assert entries == (
        AuditEntry(
            thread_id="thread-1",
            at=AT,
            level_before=RiskLevel.ELEVATED,
            level_after=RiskLevel.CRISIS,
            reason="CRISIS: intent",
        ),
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

    [entry] = audit_entries("thread-1", IN_CRISIS, decision, at=AT)

    assert (entry.level_before, entry.level_after) == (
        RiskLevel.CRISIS,
        RiskLevel.ELEVATED,
    )
    assert entry.reason == "the user tapped safe on the check-in"


def test_a_failed_screening_is_audited_with_its_reason_and_the_stored_level() -> None:
    decision = GateDecision(
        risk=ELEVATED, path="crisis", screening_failure=FailureReason.TIMEOUT
    )

    entries = audit_entries("thread-1", ELEVATED, decision, at=AT)

    assert entries == (
        AuditEntry(
            thread_id="thread-1",
            at=AT,
            level_before=RiskLevel.ELEVATED,
            level_after=RiskLevel.ELEVATED,
            reason=STORED_LEVEL_KEPT,
            screening_failure=FailureReason.TIMEOUT,
        ),
    )


def test_a_screened_turn_that_keeps_the_level_is_not_audited() -> None:
    decision = GateDecision(risk=IN_CRISIS, path="crisis")

    assert audit_entries("thread-1", IN_CRISIS, decision, at=AT) == ()


def test_a_recheck_that_raises_the_level_is_audited_before_a_new_failure() -> None:
    # A missed turn rechecked to CRISIS, then the new message fails to screen.
    decision = GateDecision(
        risk=IN_CRISIS, path="crisis", screening_failure=FailureReason.UNAVAILABLE
    )

    entries = audit_entries("thread-1", ELEVATED, decision, at=AT)

    assert [(entry.level_after, entry.screening_failure) for entry in entries] == [
        (RiskLevel.CRISIS, None),
        (RiskLevel.CRISIS, FailureReason.UNAVAILABLE),
    ]


async def test_the_in_memory_log_keeps_entries_in_order() -> None:
    log = InMemoryCrisisAuditLog()
    first = AuditEntry("thread-1", AT, RiskLevel.NONE, RiskLevel.CRISIS, "first")
    second = AuditEntry("thread-2", AT, RiskLevel.NONE, RiskLevel.WATCH, "second")

    await log.append(first)
    await log.append(second)

    assert log.entries == (first, second)


def test_each_missed_turn_given_up_on_is_audited() -> None:
    decision = GateDecision(risk=ELEVATED, path="crisis", abandoned=2)

    entries = audit_entries("thread-1", ELEVATED, decision, at=AT)

    assert [entry.reason for entry in entries] == [ABANDONED, ABANDONED]
    assert all(entry.level_after is RiskLevel.ELEVATED for entry in entries)
