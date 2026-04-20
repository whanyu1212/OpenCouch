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
gets them in one atomic read. See schema.yaml §2 namespaces.procedural
for the rationale.

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

# The fixed record key for every user's procedural profile. Matches
# schema.yaml §2 namespaces.procedural.record_schema.id.
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
    """Return the ``(user_id, "procedural")`` namespace tuple.

    Keeps the string literal in exactly one place so the rest of the
    codebase doesn't scatter references to the namespace name.
    """

    return (user_id, "procedural")


async def aget_procedural_profile(
    store: MemoryStore,
    *,
    user_id: str,
) -> ProceduralProfile:
    """Return the procedural profile for ``user_id``.

    When no profile exists yet (the common case for a new user),
    returns a **fresh empty profile** with ``proactive_recall_enabled``
    set to the schema default (False) and an empty ``rules`` list.
    The empty profile is not written to the store — callers that want
    to persist it must call :func:`aput_procedural_profile` explicitly.

    This empty-default-on-miss behavior matches the schema's intent:
    every user has a procedural profile; it's just implicit until
    something writes to it. The alternative (returning ``None`` and
    making every caller handle the null case) would scatter defensive
    code across every reader.
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
    """Persist ``profile`` as the procedural profile for ``user_id``.

    Overwrites any existing profile at the same key. Callers that want
    additive semantics should use :func:`aadd_procedural_rule` or
    :func:`aset_proactive_recall` instead, which load the existing
    profile first and mutate it in place.
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
    """Add ``rule`` to the user's procedural profile.

    Convenience helper for the writer node: loads the current profile
    (empty if none), reconciles the new rule against the existing set,
    writes the profile back when it changes, and returns the updated
    profile.

    Phase D nuance: duplicate rules are skipped and weaker/conflicting
    rules can be replaced by the new rule, with the replaced rule moved
    into ``profile.archived_rules`` for auditability. Callers that need
    the exact action should use :func:`aupsert_procedural_rule`.
    """

    result = await aupsert_procedural_rule(store, user_id=user_id, rule=rule)
    return result.profile


def _evict_oldest_rules(profile: ProceduralProfile) -> None:
    """Archive the oldest rules if the profile exceeds MAX_ACTIVE_RULES.

    Evicts by earliest ``added_at`` timestamp. Evicted rules are moved
    to ``archived_rules`` with ``superseded_by="eviction"`` so they
    remain auditable but stop inflating the response prompt.
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
    """Add or reconcile ``rule`` against the user's procedural profile."""

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
    """Set the ``proactive_recall_enabled`` toggle for this user.

    Convenience helper for the ``/memory recall on|off`` CLI command:
    loads the current profile (empty if none), updates the toggle,
    writes the profile back. Returns the updated profile so the CLI
    can render a confirmation message.

    Unlike :func:`aadd_procedural_rule`, this helper does change state
    even on an empty profile — the toggle is part of the profile's
    scalar state, so setting it to True on a new user creates a
    profile with the toggle on and an empty rules list.
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
    """Return whether proactive recall is currently enabled for this user.

    Convenience helper for the prompt builders: most callers just
    want the toggle value, not the full profile. Pulling the full
    profile is still a single store read, so this wrapper is a small
    clarity win rather than a performance optimization.

    Returns ``False`` for users with no profile yet — matches the
    schema default.
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
    """Construct a :class:`ProceduralRule` with sensible defaults for v0.7.

    Hot-path writes use ``source="explicit_user"`` with
    ``confidence="high"``. Session-end promotion can override those
    defaults with ``source="consolidation"`` and a lower confidence.
    This helper centralizes the common metadata so callers don't have
    to spell it out on every call.

    The ``added_at`` timestamp is set to "now" in UTC ISO-8601 format
    with a ``Z`` suffix — matching the timestamp convention used by
    every other memory model in the system.
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
