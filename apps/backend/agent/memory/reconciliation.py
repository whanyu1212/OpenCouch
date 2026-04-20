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

from agent.memory.models import ProceduralRule, SemanticFact
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
    """Return whether a stored semantic fact is still active."""

    if not value.get("user_visible", True):
        return False
    if value.get("dormant_at"):
        return False
    if value.get("superseded_by"):
        return False
    return True


def filter_active_semantic_records(records: list[StoreRecord]) -> list[StoreRecord]:
    """Return only semantic records that are still active."""

    return [
        record for record in records if is_active_semantic_record_value(record.value)
    ]


def _semantic_slot_matches(fact: SemanticFact, record: StoreRecord) -> bool:
    """Return whether two semantic records occupy the same conceptual slot."""

    value = record.value
    return (
        fact.category == value.get("category")
        and fact.subject.type == value.get("subject", {}).get("type")
        and fact.subject.identifier == value.get("subject", {}).get("identifier")
        and fact.predicate == value.get("predicate")
        and fact.object.type == value.get("object", {}).get("type")
    )


def _normalized_identifier(identifier: str) -> str:
    return " ".join(identifier.lower().split())


def _identifier_tokens(identifier: str) -> frozenset[str]:
    return tokenize_meaningful(identifier)


def _identifier_subset_overlap(left: str, right: str) -> bool:
    """Return whether one identifier is a token-subset of the other."""

    left_tokens = _identifier_tokens(left)
    right_tokens = _identifier_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def _prefer_new_identifier(new_identifier: str, existing_identifier: str) -> bool:
    """Return whether the new identifier is more specific than the existing one."""

    new_tokens = _identifier_tokens(new_identifier)
    existing_tokens = _identifier_tokens(existing_identifier)
    new_specificity = (len(new_tokens), len(new_identifier.strip()))
    existing_specificity = (len(existing_tokens), len(existing_identifier.strip()))
    return new_specificity > existing_specificity


def _has_explicit_correction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SEMANTIC_CORRECTION_MARKERS)


def _semantic_topic_overlap(fact: SemanticFact, record: StoreRecord) -> int:
    """Return rough topical overlap between two semantic facts."""

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
    """Return whether a semantic fact should bump, supersede, or coexist."""

    plan = SemanticReconciliationPlan()
    correction_records: list[StoreRecord] = []

    for record in filter_active_semantic_records(existing_records):
        if not _semantic_slot_matches(fact, record):
            continue

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
    """Return one text blob for conflict/dedup checks."""

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
    """Return whether a procedural rule should append, replace, or skip."""

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
