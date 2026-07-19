"""Conservative reconciliation helpers for durable memory writes.

These helpers add post-extraction cleanup without turning the hot path
into a full consolidation system. They answer two narrow questions:

1. Should a new semantic fact bump, supersede, or coexist with active
   semantic records that already exist?
2. Should a new procedural rule append, replace an older weaker rule,
   or be skipped as a duplicate/conflict?

The async helpers use LLM-primary classifiers for product judgment.
Local code keeps only exact duplicate/storage mechanics; classifier
failures raise by default, but callers can opt into conservative
fallback plans explicitly when they prefer continuity over strictness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.memory.types import MemoryWrite, ProceduralRule, SemanticFact
from agent.memory.store import StoreRecord
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SemanticReconciliationPlan:
    """The semantic write action after conservative reconciliation."""

    bump_record: StoreRecord | None = None
    supersede_records: list[StoreRecord] = field(default_factory=list)


SemanticReconciliationFailurePolicy = Literal["raise", "coexist"]
ProceduralRuleAction = Literal["append", "replace", "skip"]
ProceduralReconciliationFailurePolicy = Literal["raise", "append"]


@dataclass(slots=True)
class ProceduralReconciliationPlan:
    """The procedural write action after conservative reconciliation."""

    action: ProceduralRuleAction
    replace_indexes: list[int] = field(default_factory=list)


class SemanticReconciliationDecision(BaseModel):
    """Structured output for semantic memory reconciliation."""

    action: Literal["bump", "supersede", "coexist"]
    record_indexes: list[int] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


class ProceduralReconciliationDecision(BaseModel):
    """Structured output for procedural rule reconciliation."""

    action: ProceduralRuleAction
    replace_indexes: list[int] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


def is_active_semantic_record_value(value: dict[str, Any]) -> bool:
    """Return whether a stored semantic fact is still active.

    Args:
        value (dict[str, Any]): Serialized semantic fact payload.

    Returns:
        bool: ``True`` when the record is visible and not dormant or superseded.
    """

    if not value.get("user_visible", True):
        return False
    if value.get("dormant_at"):
        return False
    if value.get("superseded_by"):
        return False
    return True


def filter_active_semantic_records(records: list[StoreRecord]) -> list[StoreRecord]:
    """Return only semantic records that are still active.

    Args:
        records (list[StoreRecord]): Candidate records to filter.

    Returns:
        list[StoreRecord]: Active semantic records only.
    """

    return [
        record for record in records if is_active_semantic_record_value(record.value)
    ]


def _semantic_subject_matches(
    candidate: MemoryWrite | SemanticFact,
    record: StoreRecord,
) -> bool:
    """Return whether the candidate and record refer to the same subject slot."""

    value = record.value
    record_subject = value.get("subject", {})
    if candidate.subject.type != record_subject.get("type"):
        return False
    if candidate.subject.identifier == record_subject.get("identifier"):
        return True

    # User-subject memories are scoped by the store namespace owner. Treat
    # extractor placeholder aliases (e.g. ``test-user``) as the same slot when
    # either side already uses that authoritative owner id; the namespace
    # remains the isolation boundary between different people.
    record_owner_id = record.namespace[0] if record.namespace else None
    subject_identifiers = {
        candidate.subject.identifier,
        record_subject.get("identifier"),
    }
    return (
        candidate.subject.type == "User"
        and record_owner_id is not None
        and record_owner_id in subject_identifiers
    )


def _semantic_slot_matches(
    candidate: MemoryWrite | SemanticFact,
    record: StoreRecord,
) -> bool:
    """Return whether two semantic records occupy the same conceptual slot.

    Args:
        candidate (MemoryWrite | SemanticFact): New semantic candidate.
        record (StoreRecord): Existing stored semantic record.

    Returns:
        bool: ``True`` when the records share the same conceptual slot.
    """

    value = record.value
    return (
        candidate.category == value.get("category")
        and _semantic_subject_matches(candidate, record)
        and candidate.predicate == value.get("predicate")
        and candidate.object.type == value.get("object", {}).get("type")
    )


def filter_semantic_collision_candidates(
    candidate: MemoryWrite | SemanticFact,
    existing_records: list[StoreRecord],
) -> list[StoreRecord]:
    """Return active semantic records that could collide with the candidate.

    Args:
        candidate (MemoryWrite | SemanticFact): New semantic candidate.
        existing_records (list[StoreRecord]): Existing semantic records.

    Returns:
        list[StoreRecord]: Active records in the same conceptual slot.
    """

    return [
        record
        for record in filter_active_semantic_records(existing_records)
        if _semantic_slot_matches(candidate, record)
    ]


def _normalized_identifier(identifier: str) -> str:
    """Normalize an identifier for exact string comparison.

    Args:
        identifier (str): Raw identifier text.

    Returns:
        str: Lowercase identifier with normalized whitespace.
    """

    return " ".join(identifier.lower().split())


def _normalized_text(value: str) -> str:
    """Normalize text for exact duplicate comparison.

    Args:
        value (str): Raw text.

    Returns:
        str: Lowercase text with normalized whitespace.
    """

    return " ".join(value.lower().split())


def _record_object_identifier(record: StoreRecord) -> str:
    """Return a semantic record's object identifier.

    Args:
        record (StoreRecord): Stored semantic record.

    Returns:
        str: Object identifier, or an empty string when absent.
    """

    return str(record.value.get("object", {}).get("identifier") or "")


def _record_evidence_quote(record: StoreRecord) -> str:
    """Return a semantic record's evidence quote.

    Args:
        record (StoreRecord): Stored semantic record.

    Returns:
        str: Evidence quote, or an empty string when absent.
    """

    return str(record.value.get("evidence_quote") or "")


def _find_exact_duplicate_semantic_record(
    fact: SemanticFact,
    collision_records: list[StoreRecord],
) -> StoreRecord | None:
    """Return the exact duplicate semantic collision, if any.

    Args:
        fact (SemanticFact): New semantic fact.
        collision_records (list[StoreRecord]): Active same-slot records.

    Returns:
        StoreRecord | None: Existing duplicate record, or ``None``.
    """

    fact_identifier = _normalized_identifier(fact.object.identifier)
    fact_evidence = _normalized_text(fact.evidence_quote)
    for record in collision_records:
        if (
            _normalized_identifier(_record_object_identifier(record)) == fact_identifier
            and _normalized_text(_record_evidence_quote(record)) == fact_evidence
        ):
            return record
    return None


def plan_procedural_rule_write_without_llm(
    new_rule: ProceduralRule,
    existing_rules: list[ProceduralRule],
) -> ProceduralReconciliationPlan:
    """Return append-or-skip for no-LLM procedural writes.

    Args:
        new_rule (ProceduralRule): New procedural rule to reconcile.
        existing_rules (list[ProceduralRule]): Existing procedural rules.

    Returns:
        ProceduralReconciliationPlan: ``skip`` for exact duplicate rule text,
        otherwise ``append``.
    """

    new_text = _normalized_text(new_rule.rule)
    if any(
        _normalized_text(existing_rule.rule) == new_text
        for existing_rule in existing_rules
    ):
        return ProceduralReconciliationPlan(action="skip")
    return ProceduralReconciliationPlan(action="append")


def _find_exact_duplicate_procedural_rule_index(
    new_rule: ProceduralRule,
    existing_rules: list[ProceduralRule],
) -> int | None:
    """Return the index of an exact duplicate procedural rule.

    Args:
        new_rule (ProceduralRule): New procedural rule.
        existing_rules (list[ProceduralRule]): Existing procedural rules.

    Returns:
        int | None: Duplicate index, or ``None``.
    """

    new_text = _normalized_text(new_rule.rule)
    for index, existing_rule in enumerate(existing_rules):
        if _normalized_text(existing_rule.rule) == new_text:
            return index
    return None


def _apply_semantic_reconciliation_decision(
    decision: SemanticReconciliationDecision,
    collision_records: list[StoreRecord],
) -> SemanticReconciliationPlan:
    """Convert a structured semantic reconciliation decision into a plan.

    Args:
        decision (SemanticReconciliationDecision): LLM decision.
        collision_records (list[StoreRecord]): Candidate collision records.

    Returns:
        SemanticReconciliationPlan: Final semantic reconciliation plan.
    """

    if decision.confidence == "low" or decision.action == "coexist":
        return SemanticReconciliationPlan()

    valid_indexes = [
        index
        for index in sorted(set(decision.record_indexes))
        if 0 <= index < len(collision_records)
    ]
    if decision.action == "bump":
        if len(valid_indexes) != 1:
            raise ValueError("Semantic reconciliation selected invalid bump index.")
        return SemanticReconciliationPlan(
            bump_record=collision_records[valid_indexes[0]]
        )

    if decision.action == "supersede":
        if not valid_indexes:
            raise ValueError(
                "Semantic reconciliation selected no valid supersede records."
            )
        return SemanticReconciliationPlan(
            supersede_records=[collision_records[index] for index in valid_indexes]
        )

    raise ValueError(f"Unsupported semantic reconciliation action: {decision.action}")


def _apply_procedural_reconciliation_decision(
    decision: ProceduralReconciliationDecision,
    existing_rules: list[ProceduralRule],
) -> ProceduralReconciliationPlan:
    """Convert a structured procedural reconciliation decision into a plan.

    Args:
        decision (ProceduralReconciliationDecision): LLM decision.
        existing_rules (list[ProceduralRule]): Active existing rules.

    Returns:
        ProceduralReconciliationPlan: Final procedural reconciliation plan.
    """

    if decision.confidence == "low":
        return ProceduralReconciliationPlan(action="append")
    if decision.action == "append":
        return ProceduralReconciliationPlan(action="append")
    if decision.action == "skip":
        return ProceduralReconciliationPlan(action="skip")

    valid_indexes = [
        index
        for index in sorted(set(decision.replace_indexes))
        if 0 <= index < len(existing_rules)
    ]
    if not valid_indexes:
        raise ValueError("Procedural reconciliation selected no valid replace indexes.")
    return ProceduralReconciliationPlan(action="replace", replace_indexes=valid_indexes)


def _semantic_reconciliation_prompt(
    fact: SemanticFact,
    collision_records: list[StoreRecord],
) -> str:
    """Build the LLM prompt for semantic reconciliation.

    Args:
        fact (SemanticFact): New semantic fact.
        collision_records (list[StoreRecord]): Active same-slot records.

    Returns:
        str: Prompt for structured reconciliation.
    """

    record_lines = []
    for index, record in enumerate(collision_records):
        value = record.value
        object_value = value.get("object", {}) or {}
        record_lines.append(
            f"{index}. key={record.key!r}; "
            f"object={object_value.get('type')}:{object_value.get('identifier')}; "
            f"evidence={value.get('evidence_quote')!r}"
        )
    records = "\n".join(record_lines) or "(none)"
    return (
        "Decide how to reconcile a new semantic memory with existing active "
        "records in the same conceptual slot.\n\n"
        "Actions:\n"
        "- bump: the new fact is essentially the same memory; refresh one "
        "existing record and do not write a new one.\n"
        "- supersede: the new fact corrects or replaces selected older records; "
        "write the new fact and mark selected records dormant.\n"
        "- coexist: the new fact can validly live alongside the old records.\n\n"
        "Exact duplicates have already been handled before this classifier. "
        "Same-slot records with changed evidence or identifier must be judged "
        "from meaning, not string similarity alone.\n\n"
        "Be conservative. Do not supersede unless the new evidence clearly "
        "corrects, replaces, or makes the old record stale.\n\n"
        "For current-state facts, choose supersede when the new evidence says "
        "the current situation has changed from an older state, even if the "
        "older state may have been historically true. Choose coexist when the "
        "records describe different people, concerns, events, or stable facts "
        "that can all remain true.\n\n"
        f"New fact: category={fact.category}; predicate={fact.predicate}; "
        f"object={fact.object.type}:{fact.object.identifier}; "
        f"evidence={fact.evidence_quote!r}\n\n"
        "Existing records:\n"
        f"{records}"
    )


def _procedural_reconciliation_prompt(
    new_rule: ProceduralRule,
    existing_rules: list[ProceduralRule],
) -> str:
    """Build the LLM prompt for procedural rule reconciliation.

    Args:
        new_rule (ProceduralRule): New procedural rule.
        existing_rules (list[ProceduralRule]): Active existing rules.

    Returns:
        str: Prompt for structured reconciliation.
    """

    rule_lines = []
    for index, rule in enumerate(existing_rules):
        rule_lines.append(
            f"{index}. id={rule.id!r}; rule={rule.rule!r}; evidence={rule.evidence!r}"
        )
    rules = "\n".join(rule_lines) or "(none)"
    return (
        "Decide how to reconcile a new procedural preference rule with the "
        "user's active procedural profile.\n\n"
        "Actions:\n"
        "- append: the new rule is distinct and should be added.\n"
        "- replace: the new rule is a clearer, more specific, or conflicting "
        "replacement for selected existing rules.\n"
        "- skip: the new rule duplicates existing guidance or is weaker than "
        "what is already stored.\n\n"
        "Exact duplicate rule text has already been handled before this "
        "classifier. Judge conflicts, negations, and specificity from meaning, "
        "not marker words alone.\n\n"
        "Be conservative about replace. Only replace when selected existing "
        "rules are genuinely stale, weaker, or contradictory.\n\n"
        f"New rule: id={new_rule.id!r}; rule={new_rule.rule!r}; "
        f"evidence={new_rule.evidence!r}\n\n"
        "Existing active rules:\n"
        f"{rules}"
    )


def _reconciliation_system_prompt() -> str:
    """Return the common system prompt for reconciliation classifiers.

    Returns:
        str: System instruction for structured reconciliation classification.
    """

    return (
        "You are a strict memory reconciliation classifier. Return only the "
        "structured decision. You do not write user-facing text."
    )


async def plan_semantic_write_llm_primary(
    fact: SemanticFact,
    existing_records: list[StoreRecord],
    *,
    llm_client: BaseLLMClient | None,
    failure_policy: SemanticReconciliationFailurePolicy = "raise",
) -> SemanticReconciliationPlan:
    """Return semantic reconciliation using an LLM primary path.

    Args:
        fact (SemanticFact): New semantic fact to reconcile.
        existing_records (list[StoreRecord]): Existing semantic records.
        llm_client (BaseLLMClient | None): Optional classifier client.
        failure_policy (SemanticReconciliationFailurePolicy): Behavior to use
            when reconciliation cannot consult the LLM.

    Returns:
        SemanticReconciliationPlan: Final reconciliation plan.
    """

    collision_records = filter_semantic_collision_candidates(fact, existing_records)
    if not collision_records:
        return SemanticReconciliationPlan()

    exact_duplicate = _find_exact_duplicate_semantic_record(fact, collision_records)
    if exact_duplicate is not None:
        return SemanticReconciliationPlan(bump_record=exact_duplicate)
    if llm_client is None:
        if failure_policy == "coexist":
            logger.warning(
                "Semantic reconciliation requires an LLM client; falling back to coexist."
            )
            return SemanticReconciliationPlan()
        raise RuntimeError("Semantic reconciliation requires an LLM client.")

    try:
        decision: SemanticReconciliationDecision = await llm_client.generate_structured(
            prompt=_semantic_reconciliation_prompt(fact, collision_records),
            response_schema=SemanticReconciliationDecision,
            system_instruction=_reconciliation_system_prompt(),
        )
    except Exception:
        if failure_policy == "coexist":
            logger.warning(
                "Semantic reconciliation LLM classifier failed; falling back to coexist.",
                exc_info=True,
            )
            return SemanticReconciliationPlan()
        logger.warning(
            "Semantic reconciliation LLM classifier failed.",
            exc_info=True,
        )
        raise

    return _apply_semantic_reconciliation_decision(decision, collision_records)


async def plan_procedural_rule_write_llm_primary(
    new_rule: ProceduralRule,
    existing_rules: list[ProceduralRule],
    *,
    llm_client: BaseLLMClient | None,
    failure_policy: ProceduralReconciliationFailurePolicy = "raise",
) -> ProceduralReconciliationPlan:
    """Return procedural reconciliation using an LLM primary path.

    Args:
        new_rule (ProceduralRule): New procedural rule to reconcile.
        existing_rules (list[ProceduralRule]): Existing procedural rules.
        llm_client (BaseLLMClient | None): Optional classifier client.
        failure_policy (ProceduralReconciliationFailurePolicy): Behavior to use
            when reconciliation cannot consult the LLM.

    Returns:
        ProceduralReconciliationPlan: Final reconciliation plan.
    """

    if not existing_rules:
        return ProceduralReconciliationPlan(action="append")

    if (
        _find_exact_duplicate_procedural_rule_index(new_rule, existing_rules)
        is not None
    ):
        return ProceduralReconciliationPlan(action="skip")
    if llm_client is None:
        if failure_policy == "append":
            logger.warning(
                "Procedural reconciliation requires an LLM client; falling back to append."
            )
            return ProceduralReconciliationPlan(action="append")
        raise RuntimeError("Procedural reconciliation requires an LLM client.")

    try:
        decision: ProceduralReconciliationDecision = (
            await llm_client.generate_structured(
                prompt=_procedural_reconciliation_prompt(new_rule, existing_rules),
                response_schema=ProceduralReconciliationDecision,
                system_instruction=_reconciliation_system_prompt(),
            )
        )
    except Exception:
        if failure_policy == "append":
            logger.warning(
                "Procedural reconciliation LLM classifier failed; falling back to append.",
                exc_info=True,
            )
            return ProceduralReconciliationPlan(action="append")
        logger.warning(
            "Procedural reconciliation LLM classifier failed.",
            exc_info=True,
        )
        raise

    return _apply_procedural_reconciliation_decision(decision, existing_rules)
