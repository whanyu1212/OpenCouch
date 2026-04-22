"""Procedural memory store helpers.

Thin wrappers around the raw :class:`MemoryStore` protocol that implement
the **profile-as-single-document** storage shape the procedural namespace
uses. Unlike semantic memory (one record per fact) and episodic memory
(one record per session arc), procedural memory stores a single
:class:`ProceduralProfile` per user at a fixed key, and every write is a
load → mutate → put round-trip.

Why profile-as-document instead of one-record-per-rule:

Procedural rules are few (typically 5-15 per user) and need to stay
coherent as a set. If two rules contradict each other, the agent needs
to see both at once to decide which wins. A collection model would make
"get all rules for this user" require a range scan; a profile model
gets them in one atomic read.

The profile also houses the ``proactive_recall_enabled`` toggle (the
``/memory recall on|off`` setting), because the setting is a per-user
preference that belongs alongside the rules in the same atomic unit.
This is why :func:`aset_proactive_recall` and :func:`aget_proactive_recall`
live in this module — they're operations on the procedural profile, not
a separate settings namespace.

Module layout:

- :data:`PROCEDURAL_KEY` — the fixed key under which every user's
  procedural profile is stored. Always ``"user_response_style"``.
- :func:`procedural_namespace` — builds the ``(user_id, "procedural")``
  tuple, keeping the string literal in exactly one place.
- :func:`aget_procedural_profile` — read; returns a fresh empty
  profile when no record exists yet (the common case for new users).
- :func:`aput_procedural_profile` — write; serializes the pydantic
  model to a dict so the store layer stays model-agnostic.
- :func:`aupsert_procedural_rule` — load → reconcile → put convenience
  for phase-D duplicate/conflict handling.
- :func:`aadd_procedural_rule` — compatibility wrapper that routes
  through the upsert helper and returns only the updated profile.
- :func:`aset_proactive_recall` — load → toggle → put convenience.
  The CLI ``/memory recall on|off`` command uses this.
- :func:`aget_proactive_recall` — reads just the toggle without
  pulling the full profile into caller code. Used by the prompt
  builders at response time to decide whether to emit the "do not
  proactively reference memory" constraint.

All helpers are async because the underlying store is async. Callers
that need sync access are doing something wrong — agent nodes and CLI
command handlers are both async, so there's no call site that would
need a sync variant.

Concurrency note: the profile-as-document shape means writes are
**last-write-wins** with no merging. If two code paths load the
profile concurrently, mutate different fields, and put it back, one
side's changes will be lost. In phase 1 this isn't a concern because
the runtime is single-threaded per session and nodes within a turn
run sequentially — but it's worth flagging for phase 4 background
consolidation, where a nightly job might touch the profile while a
user is active. A write-then-read-back verification pattern or a
schema-level ``version`` field is the standard fix; we don't need
either yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.memory.hashing import iso_now
from agent.memory.models import (
    ConfidenceLevel,
    MemoryWriteTiming,
    ProceduralProfile,
    ProceduralRule,
    ProceduralRuleSource,
)
from agent.memory.reconciliation import plan_procedural_rule_write
from agent.memory.store import MemoryStore, Namespace

# The fixed record key for every user's procedural profile.
PROCEDURAL_KEY = "user_response_style"
ProceduralUpsertAction = Literal["added", "replaced", "skipped"]

# Maximum number of active rules before the oldest is evicted to the
# archive. Enforced at write time only — existing oversized profiles
# remain readable but will be trimmed on the next rule write.
MAX_ACTIVE_RULES = 20


@dataclass(slots=True)
class ProceduralUpsertResult:
    """Outcome of reconciling one procedural rule against the profile."""

    profile: ProceduralProfile
    action: ProceduralUpsertAction


def procedural_namespace(user_id: str) -> Namespace:
    """Return the procedural namespace tuple for a user.

    Args:
        user_id (str): User identifier.

    Returns:
        Namespace: ``(user_id, "procedural")`` namespace tuple.
    """

    return (user_id, "procedural")


async def aget_procedural_profile(
    store: MemoryStore,
    *,
    user_id: str,
) -> ProceduralProfile:
    """Return the procedural profile for a user.

    Args:
        store (MemoryStore): Memory store to read from.
        user_id (str): User identifier.

    Returns:
        ProceduralProfile: Existing profile or a fresh empty default profile.
    """

    record = await store.aget(procedural_namespace(user_id), PROCEDURAL_KEY)
    if record is None:
        return ProceduralProfile(
            proactive_recall_enabled=False,
            rules=[],
            archived_rules=[],
            last_consolidated_at=None,
        )
    return ProceduralProfile.model_validate(record.value)


async def aput_procedural_profile(
    store: MemoryStore,
    *,
    user_id: str,
    profile: ProceduralProfile,
) -> None:
    """Persist a procedural profile for a user.

    Args:
        store (MemoryStore): Memory store to write to.
        user_id (str): User identifier.
        profile (ProceduralProfile): Procedural profile to persist.
    """

    await store.aput(
        procedural_namespace(user_id),
        PROCEDURAL_KEY,
        profile.model_dump(mode="json"),
    )


async def aadd_procedural_rule(
    store: MemoryStore,
    *,
    user_id: str,
    rule: ProceduralRule,
) -> ProceduralProfile:
    """Add a procedural rule to a user's profile.

    Args:
        store (MemoryStore): Memory store to update.
        user_id (str): User identifier.
        rule (ProceduralRule): Procedural rule to add.

    Returns:
        ProceduralProfile: Updated procedural profile.
    """

    result = await aupsert_procedural_rule(store, user_id=user_id, rule=rule)
    return result.profile


def _evict_oldest_rules(profile: ProceduralProfile) -> None:
    """Archive the oldest rules if the profile exceeds the active-rule cap.

    Args:
        profile (ProceduralProfile): Profile to mutate in place.
    """

    while len(profile.rules) > MAX_ACTIVE_RULES:
        oldest_idx = min(
            range(len(profile.rules)),
            key=lambda i: profile.rules[i].added_at or "",
        )
        evicted = profile.rules.pop(oldest_idx).model_copy(
            update={
                "dormant_at": iso_now(),
                "superseded_by": "eviction",
                "user_visible": False,
            }
        )
        profile.archived_rules = [*profile.archived_rules, evicted]


async def aupsert_procedural_rule(
    store: MemoryStore,
    *,
    user_id: str,
    rule: ProceduralRule,
) -> ProceduralUpsertResult:
    """Add or reconcile a procedural rule against the user's profile.

    Args:
        store (MemoryStore): Memory store to update.
        user_id (str): User identifier.
        rule (ProceduralRule): Procedural rule to upsert.

    Returns:
        ProceduralUpsertResult: Updated profile plus the reconciliation action taken.
    """

    profile = await aget_procedural_profile(store, user_id=user_id)
    plan = plan_procedural_rule_write(rule, profile.rules)
    if plan.action == "skip":
        return ProceduralUpsertResult(profile=profile, action="skipped")

    if plan.action == "replace":
        archived_rules = list(profile.archived_rules)
        for index in plan.replace_indexes:
            archived_rule = profile.rules[index].model_copy(
                update={
                    "dormant_at": iso_now(),
                    "superseded_by": rule.id,
                    "user_visible": False,
                }
            )
            archived_rules.append(archived_rule)
        profile.rules = [
            existing_rule
            for index, existing_rule in enumerate(profile.rules)
            if index not in plan.replace_indexes
        ]
        profile.archived_rules = archived_rules
        profile.rules.append(rule)
        _evict_oldest_rules(profile)
        await aput_procedural_profile(store, user_id=user_id, profile=profile)
        return ProceduralUpsertResult(profile=profile, action="replaced")

    profile.rules.append(rule)
    _evict_oldest_rules(profile)
    await aput_procedural_profile(store, user_id=user_id, profile=profile)
    return ProceduralUpsertResult(profile=profile, action="added")


async def aset_proactive_recall(
    store: MemoryStore,
    *,
    user_id: str,
    enabled: bool,
) -> ProceduralProfile:
    """Set the proactive-recall toggle for a user.

    Args:
        store (MemoryStore): Memory store to update.
        user_id (str): User identifier.
        enabled (bool): Desired proactive-recall setting.

    Returns:
        ProceduralProfile: Updated procedural profile.
    """

    profile = await aget_procedural_profile(store, user_id=user_id)
    profile.proactive_recall_enabled = enabled
    await aput_procedural_profile(store, user_id=user_id, profile=profile)
    return profile


async def aget_proactive_recall(
    store: MemoryStore,
    *,
    user_id: str,
) -> bool:
    """Return whether proactive recall is enabled for a user.

    Args:
        store (MemoryStore): Memory store to read from.
        user_id (str): User identifier.

    Returns:
        bool: Current proactive-recall setting.
    """

    profile = await aget_procedural_profile(store, user_id=user_id)
    return profile.proactive_recall_enabled


def build_procedural_rule(
    *,
    rule_text: str,
    evidence: list[str],
    confidence: ConfidenceLevel = "high",
    source: ProceduralRuleSource = "explicit_user",
    write_timing: MemoryWriteTiming = "immediate",
    write_reason: str = "",
    policy_version: str = "phase1_v1",
) -> ProceduralRule:
    """Construct a procedural rule with the project's default metadata.

    Args:
        rule_text (str): Rule text to store.
        evidence (list[str]): Evidence snippets supporting the rule.
        confidence (ConfidenceLevel): Confidence level for the rule.
        source (ProceduralRuleSource): Rule source label.
        write_timing (MemoryWriteTiming): Memory write timing label.
        write_reason (str): Reason for the write decision.
        policy_version (str): Policy version associated with the write.

    Returns:
        ProceduralRule: New procedural rule with timestamps and defaults populated.
    """

    return ProceduralRule(
        rule=rule_text,
        evidence=evidence,
        confidence=confidence,
        added_at=iso_now(),
        source=source,
        write_timing=write_timing,
        write_reason=write_reason,
        policy_version=policy_version,
    )
