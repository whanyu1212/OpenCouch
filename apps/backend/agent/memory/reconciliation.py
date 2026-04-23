"""Conservative reconciliation helpers for durable memory writes.

Phase D adds a small amount of post-extraction cleanup without turning
the hot path into a full consolidation system. The helpers here answer
two narrow questions:

1. Should a new semantic fact bump, supersede, or coexist with active
   semantic records that already exist?
2. Should a new procedural rule append, replace an older weaker rule,
   or be skipped as a duplicate/conflict?

The heuristics are deliberately conservative. They only act when the
signals are strong enough to avoid silently deleting valid parallel
facts or preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent.memory.models import MemoryWrite, ProceduralRule, SemanticFact
from agent.memory.store import StoreRecord
from agent.memory.text_tokens import tokenize_meaningful

_SEMANTIC_CORRECTION_MARKERS = (
    "actually",
    "not anymore",
    "no longer",
    "instead",
    "used to",
    "turned out",
)

_PROCEDURAL_NEGATION_MARKERS = (
    "don't",
    "dont",
    "do not",
    "stop",
    "avoid",
    "never",
    "without",
    "no longer",
)


@dataclass(slots=True)
class SemanticReconciliationPlan:
    """The semantic write action after conservative reconciliation."""

    bump_record: StoreRecord | None = None
    supersede_records: list[StoreRecord] = field(default_factory=list)


ProceduralRuleAction = Literal["append", "replace", "skip"]


@dataclass(slots=True)
class ProceduralReconciliationPlan:
    """The procedural write action after conservative reconciliation."""

    action: ProceduralRuleAction
    replace_indexes: list[int] = field(default_factory=list)


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
        and candidate.subject.type == value.get("subject", {}).get("type")
        and candidate.subject.identifier == value.get("subject", {}).get("identifier")
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
    return " ".join(identifier.lower().split())


def _identifier_tokens(identifier: str) -> frozenset[str]:
    return tokenize_meaningful(identifier)


def _identifier_subset_overlap(left: str, right: str) -> bool:
    """Return whether one identifier is a token-subset of the other.

    Args:
        left (str): First identifier.
        right (str): Second identifier.

    Returns:
        bool: ``True`` when either token set is a subset of the other.
    """

    left_tokens = _identifier_tokens(left)
    right_tokens = _identifier_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def _prefer_new_identifier(new_identifier: str, existing_identifier: str) -> bool:
    """Return whether the new identifier is more specific than the existing one.

    Args:
        new_identifier (str): Candidate replacement identifier.
        existing_identifier (str): Existing stored identifier.

    Returns:
        bool: ``True`` when the new identifier is more specific.
    """

    new_tokens = _identifier_tokens(new_identifier)
    existing_tokens = _identifier_tokens(existing_identifier)
    new_specificity = (len(new_tokens), len(new_identifier.strip()))
    existing_specificity = (len(existing_tokens), len(existing_identifier.strip()))
    return new_specificity > existing_specificity


def _has_explicit_correction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SEMANTIC_CORRECTION_MARKERS)


def _semantic_topic_overlap(fact: SemanticFact, record: StoreRecord) -> int:
    """Return rough topical overlap between two semantic facts.

    Args:
        fact (SemanticFact): New semantic fact.
        record (StoreRecord): Existing stored semantic record.

    Returns:
        int: Count of overlapping topical tokens.
    """

    candidate_tokens = tokenize_meaningful(
        f"{fact.object.identifier} {fact.evidence_quote}"
    )
    existing_tokens = tokenize_meaningful(
        " ".join(
            [
                str(record.value.get("object", {}).get("identifier") or ""),
                str(record.value.get("evidence_quote") or ""),
            ]
        )
    )
    return len(candidate_tokens & existing_tokens)


def plan_semantic_write(
    fact: SemanticFact,
    existing_records: list[StoreRecord],
) -> SemanticReconciliationPlan:
    """Return whether a semantic fact should bump, supersede, or coexist.

    Args:
        fact (SemanticFact): New semantic fact to reconcile.
        existing_records (list[StoreRecord]): Existing semantic records.

    Returns:
        SemanticReconciliationPlan: Reconciliation action for the semantic fact.
    """

    plan = SemanticReconciliationPlan()
    correction_records: list[StoreRecord] = []

    for record in filter_semantic_collision_candidates(fact, existing_records):
        existing_identifier = str(
            record.value.get("object", {}).get("identifier") or ""
        )
        if _normalized_identifier(existing_identifier) == _normalized_identifier(
            fact.object.identifier
        ):
            plan.bump_record = record
            return plan

        if _identifier_subset_overlap(fact.object.identifier, existing_identifier):
            if _prefer_new_identifier(fact.object.identifier, existing_identifier):
                plan.supersede_records.append(record)
                continue
            plan.bump_record = record
            return plan

        if (
            _has_explicit_correction(fact.evidence_quote)
            and _semantic_topic_overlap(
                fact,
                record,
            )
            >= 2
        ):
            correction_records.append(record)

    if correction_records:
        plan.supersede_records.extend(
            record
            for record in correction_records
            if record not in plan.supersede_records
        )
    return plan


def _procedural_polarity(text: str) -> Literal["negative", "positive"]:
    lowered = text.lower()
    if any(marker in lowered for marker in _PROCEDURAL_NEGATION_MARKERS):
        return "negative"
    return "positive"


def _procedural_rule_signature(rule: ProceduralRule) -> str:
    """Return one text blob for conflict and dedup checks.

    Args:
        rule (ProceduralRule): Procedural rule to serialize.

    Returns:
        str: Combined procedural rule text and evidence.
    """

    return " ".join([rule.rule, *rule.evidence]).strip()


def _prefer_new_rule(new_rule: str, existing_rule: str) -> bool:
    new_tokens = tokenize_meaningful(new_rule)
    existing_tokens = tokenize_meaningful(existing_rule)
    new_specificity = (len(new_tokens), len(new_rule.strip()))
    existing_specificity = (len(existing_tokens), len(existing_rule.strip()))
    return new_specificity > existing_specificity


def plan_procedural_rule_write(
    new_rule: ProceduralRule,
    existing_rules: list[ProceduralRule],
) -> ProceduralReconciliationPlan:
    """Return whether a procedural rule should append, replace, or skip.

    Args:
        new_rule (ProceduralRule): New procedural rule to reconcile.
        existing_rules (list[ProceduralRule]): Existing procedural rules.

    Returns:
        ProceduralReconciliationPlan: Reconciliation action for the new rule.
    """

    new_signature = _procedural_rule_signature(new_rule)
    new_text = new_signature.lower().strip()
    new_tokens = tokenize_meaningful(new_signature)
    new_polarity = _procedural_polarity(new_signature)
    replace_indexes: list[int] = []

    for index, existing_rule in enumerate(existing_rules):
        existing_signature = _procedural_rule_signature(existing_rule)
        existing_text = existing_signature.lower().strip()
        existing_tokens = tokenize_meaningful(existing_signature)

        if existing_text == new_text:
            return ProceduralReconciliationPlan(action="skip")

        overlap = len(new_tokens & existing_tokens)
        subset_overlap = (
            overlap >= 2
            and new_tokens
            and existing_tokens
            and (new_tokens <= existing_tokens or existing_tokens <= new_tokens)
        )
        if subset_overlap:
            if _prefer_new_rule(new_rule.rule, existing_rule.rule):
                replace_indexes.append(index)
                continue
            return ProceduralReconciliationPlan(action="skip")

        if overlap >= 2 and new_polarity != _procedural_polarity(existing_signature):
            replace_indexes.append(index)

    if replace_indexes:
        return ProceduralReconciliationPlan(
            action="replace",
            replace_indexes=sorted(set(replace_indexes)),
        )
    return ProceduralReconciliationPlan(action="append")
