"""Unit tests for ``agent/memory/procedural_profile.py``.

Covers the profile-as-single-document storage pattern: empty-default-on-
miss, round-trip serialization, additive helpers (``aadd_procedural_rule``,
``aset_proactive_recall``), and the convenience readers used by prompt
builders.

These are shape-only unit tests — no live LLM calls, no pydantic validation
errors. The goal is to lock in the API contract of the helpers so the
writer services and prompt builders can import them with confidence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.types import ProceduralProfile, ProceduralRule
from agent.memory.procedural_profile import (
    PROCEDURAL_KEY,
    aadd_procedural_rule,
    aclear_procedural_rules,
    adelete_procedural_rule,
    aget_procedural_profile,
    aget_proactive_recall,
    aput_procedural_profile,
    aset_proactive_recall,
    aupsert_procedural_rule,
    build_procedural_rule,
    procedural_namespace,
)
from agent.memory.store import OpenCouchMemoryStore
from llm.base import BaseLLMClient, StructuredResponseT


class _BlockingFirstProceduralPutStore(OpenCouchMemoryStore):
    """Store test double that pauses the first procedural profile write."""

    def __init__(self) -> None:
        super().__init__()
        self.first_procedural_put_started = asyncio.Event()
        self.allow_first_procedural_put = asyncio.Event()
        self._blocked_first_put = False

    async def aput(
        self,
        namespace,
        key,
        value,
        *,
        embedding=None,
        embedding_model=None,
    ) -> None:
        if (
            namespace == ("alice", "procedural")
            and key == PROCEDURAL_KEY
            and not self._blocked_first_put
        ):
            self._blocked_first_put = True
            self.first_procedural_put_started.set()
            await self.allow_first_procedural_put.wait()

        await super().aput(
            namespace,
            key,
            value,
            embedding=embedding,
            embedding_model=embedding_model,
        )


class _FakeProceduralReconciliationLLM(BaseLLMClient):
    """Fake structured client for procedural reconciliation tests."""

    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        if response_schema.__name__ != "ProceduralReconciliationDecision":
            raise RuntimeError(f"unexpected schema {response_schema.__name__}")
        self.structured_calls += 1
        return cast(StructuredResponseT, response_schema(**self.decision))


# ─── Namespace + key constants ────────────────────────────────────────────


def test_procedural_namespace_shape() -> None:
    """The namespace helper returns ``(user_id, "procedural")`` exactly."""

    assert procedural_namespace("alice") == ("alice", "procedural")


def test_procedural_key_constant() -> None:
    """The procedural record key is fixed across all users.

    Every user's procedural profile lives at the same key
    ``user_response_style``. This test pins that constant so a
    future rename is visible in the diff.
    """

    assert PROCEDURAL_KEY == "user_response_style"


# ─── aget_procedural_profile — empty and populated ────────────────────────


@pytest.mark.asyncio
async def test_aget_profile_returns_empty_default_for_new_user() -> None:
    """A user with no procedural record gets a fresh empty profile.

    Critical contract: callers should NEVER have to handle ``None``.
    The helper returns an empty profile on miss so every downstream
    caller can write straightforward, null-free code.
    """

    store = OpenCouchMemoryStore()
    profile = await aget_procedural_profile(store, user_id="alice")

    assert isinstance(profile, ProceduralProfile)
    assert profile.rules == []
    assert profile.archived_rules == []
    assert profile.proactive_recall_enabled is False
    assert profile.last_consolidated_at is None


@pytest.mark.asyncio
async def test_aget_profile_empty_default_is_not_persisted() -> None:
    """The empty-default return does NOT write the empty profile to the store.

    Pulling a new user's profile should be a read-only operation that
    doesn't scatter empty-profile records across the namespace. The
    empty default exists only in memory until a caller explicitly
    writes something.
    """

    store = OpenCouchMemoryStore()
    await aget_procedural_profile(store, user_id="alice")

    # Hit the raw store to verify no record was written
    record = await store.aget(("alice", "procedural"), PROCEDURAL_KEY)
    assert record is None


@pytest.mark.asyncio
async def test_aget_profile_deserializes_stored_record() -> None:
    """A previously-written profile round-trips through the helper.

    Writes via the raw store; reads via the helper. Verifies that the
    helper correctly deserializes the dict back into a
    :class:`ProceduralProfile` with all fields populated.
    """

    store = OpenCouchMemoryStore()
    existing = ProceduralProfile(
        proactive_recall_enabled=True,
        rules=[
            ProceduralRule(
                rule="You've said meditation makes you more anxious.",
                evidence=["Please don't suggest meditation again"],
                confidence="high",
                added_at="2026-04-11T12:00:00Z",
                source="explicit_user",
            ),
        ],
        last_consolidated_at=None,
    )
    await store.aput(
        ("alice", "procedural"),
        PROCEDURAL_KEY,
        existing.model_dump(mode="json"),
    )

    loaded = await aget_procedural_profile(store, user_id="alice")

    assert loaded.proactive_recall_enabled is True
    assert len(loaded.rules) == 1
    assert loaded.archived_rules == []
    assert loaded.rules[0].rule == "You've said meditation makes you more anxious."
    assert loaded.rules[0].source == "explicit_user"
    assert loaded.rules[0].confidence == "high"


# ─── aput_procedural_profile — overwrite semantics ────────────────────────


@pytest.mark.asyncio
async def test_aput_profile_overwrites_existing() -> None:
    """``aput_procedural_profile`` is unconditional overwrite.

    Callers that want additive semantics must use
    :func:`aadd_procedural_rule` or :func:`aset_proactive_recall`.
    This test pins the raw overwrite behavior so the contract is
    explicit.
    """

    store = OpenCouchMemoryStore()
    first = ProceduralProfile(
        proactive_recall_enabled=True,
        rules=[
            ProceduralRule(
                rule="You prefer short replies.",
                evidence=["Please keep it short"],
                confidence="high",
                added_at="2026-04-11T12:00:00Z",
                source="explicit_user",
            ),
        ],
        last_consolidated_at=None,
    )
    await aput_procedural_profile(store, user_id="alice", profile=first)

    # Second write with different state
    second = ProceduralProfile(
        proactive_recall_enabled=False,
        rules=[],
        last_consolidated_at=None,
    )
    await aput_procedural_profile(store, user_id="alice", profile=second)

    loaded = await aget_procedural_profile(store, user_id="alice")
    assert loaded.proactive_recall_enabled is False
    assert loaded.rules == []


# ─── aadd_procedural_rule — additive append ───────────────────────────────


@pytest.mark.asyncio
async def test_aadd_rule_on_new_user_creates_profile() -> None:
    """Appending a rule for a new user creates and persists the profile.

    The writer node uses this helper on every rule write, and most
    rule writes are for users whose profile doesn't exist yet. This
    is the critical happy path.
    """

    store = OpenCouchMemoryStore()
    rule = build_procedural_rule(
        rule_text="You've said meditation makes you more anxious.",
        evidence=["Please don't suggest meditation again"],
    )

    updated = await aadd_procedural_rule(store, user_id="alice", rule=rule)

    assert len(updated.rules) == 1
    assert updated.rules[0].rule == "You've said meditation makes you more anxious."

    # Verify persistence — reload and check the profile is there
    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert len(reloaded.rules) == 1
    assert reloaded.rules[0].evidence == ["Please don't suggest meditation again"]


@pytest.mark.asyncio
async def test_aadd_rule_preserves_existing_rules() -> None:
    """Appending a second rule keeps the first rule intact.

    Additive semantics: each call adds to the list without touching
    prior entries. A second rule on the same user grows the list to
    length 2, not replaces.
    """

    store = OpenCouchMemoryStore()
    first = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    second = build_procedural_rule(
        rule_text="You've said meditation makes you more anxious.",
        evidence=["Please don't suggest meditation again"],
    )

    await aadd_procedural_rule(store, user_id="alice", rule=first)
    await aadd_procedural_rule(store, user_id="alice", rule=second)

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert len(reloaded.rules) == 2
    assert reloaded.rules[0].rule == "You prefer short replies."
    assert reloaded.rules[1].rule == ("You've said meditation makes you more anxious.")


@pytest.mark.asyncio
async def test_aadd_rule_without_llm_skips_exact_duplicate_rule_text() -> None:
    """No-LLM additive writes only skip exact duplicate rule text."""

    store = OpenCouchMemoryStore()
    first = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short."],
    )
    duplicate = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Keep replies brief."],
    )

    await aadd_procedural_rule(store, user_id="alice", rule=first)
    await aadd_procedural_rule(store, user_id="alice", rule=duplicate)

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert len(reloaded.rules) == 1
    assert reloaded.rules[0].id == first.id


@pytest.mark.asyncio
async def test_aupsert_rule_replaces_weaker_rule_with_llm() -> None:
    """A stronger replacement should use LLM reconciliation."""

    store = OpenCouchMemoryStore()
    original = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short."],
    )
    replacement = build_procedural_rule(
        rule_text="You prefer short, direct replies without extra validation.",
        evidence=["Please be short and direct."],
    )
    llm = _FakeProceduralReconciliationLLM(
        {
            "action": "replace",
            "replace_indexes": [0],
            "reason": "new rule is clearer and more specific",
            "confidence": "high",
        }
    )

    await aadd_procedural_rule(store, user_id="alice", rule=original)
    result = await aupsert_procedural_rule(
        store,
        user_id="alice",
        rule=replacement,
        llm_client=llm,
    )

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert result.action == "replaced"
    assert llm.structured_calls == 1
    assert len(reloaded.rules) == 1
    assert (
        reloaded.rules[0].rule
        == "You prefer short, direct replies without extra validation."
    )
    assert len(reloaded.archived_rules) == 1
    assert reloaded.archived_rules[0].rule == "You prefer short replies."
    assert reloaded.archived_rules[0].superseded_by == reloaded.rules[0].id
    assert reloaded.archived_rules[0].dormant_at is not None
    assert reloaded.archived_rules[0].user_visible is False


@pytest.mark.asyncio
async def test_aupsert_rule_replaces_conflicting_existing_rule_with_llm() -> None:
    """A newer conflicting rule should use LLM reconciliation."""

    store = OpenCouchMemoryStore()
    original = build_procedural_rule(
        rule_text="Suggest meditation when it seems useful.",
        evidence=["Meditation is okay."],
    )
    replacement = build_procedural_rule(
        rule_text="Don't suggest meditation again.",
        evidence=["Please don't suggest meditation again."],
    )
    llm = _FakeProceduralReconciliationLLM(
        {
            "action": "replace",
            "replace_indexes": [0],
            "reason": "new rule conflicts with older meditation guidance",
            "confidence": "high",
        }
    )

    await aadd_procedural_rule(store, user_id="alice", rule=original)
    result = await aupsert_procedural_rule(
        store,
        user_id="alice",
        rule=replacement,
        llm_client=llm,
    )

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert result.action == "replaced"
    assert llm.structured_calls == 1
    assert len(reloaded.rules) == 1
    assert reloaded.rules[0].rule == "Don't suggest meditation again."
    assert len(reloaded.archived_rules) == 1
    assert reloaded.archived_rules[0].rule == "Suggest meditation when it seems useful."
    assert reloaded.archived_rules[0].superseded_by == reloaded.rules[0].id


@pytest.mark.asyncio
async def test_aadd_rule_preserves_proactive_recall_setting() -> None:
    """Adding a rule must not clobber the proactive_recall_enabled toggle.

    Regression guard: the load → mutate → put idiom is correct ONLY
    if mutation preserves the fields it isn't touching. A naïve
    implementation that constructs a fresh profile with the new rule
    would drop the toggle back to False; this test proves the
    implementation uses in-place mutation correctly.
    """

    store = OpenCouchMemoryStore()
    # Start with recall enabled and no rules
    await aset_proactive_recall(store, user_id="alice", enabled=True)

    # Add a rule — the toggle should stay True
    rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=rule)

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert reloaded.proactive_recall_enabled is True
    assert len(reloaded.rules) == 1


@pytest.mark.asyncio
async def test_concurrent_rule_write_and_recall_toggle_preserve_both_updates() -> None:
    """Concurrent procedural mutations should serialize per user."""

    store = _BlockingFirstProceduralPutStore()
    rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )

    add_task = asyncio.create_task(
        aadd_procedural_rule(store, user_id="alice", rule=rule)
    )
    await store.first_procedural_put_started.wait()

    recall_task = asyncio.create_task(
        aset_proactive_recall(store, user_id="alice", enabled=True)
    )
    await asyncio.sleep(0)
    assert not recall_task.done()

    store.allow_first_procedural_put.set()
    await asyncio.gather(add_task, recall_task)

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert len(reloaded.rules) == 1
    assert reloaded.proactive_recall_enabled is True


# ─── aset_proactive_recall + aget_proactive_recall ────────────────────────


@pytest.mark.asyncio
async def test_aset_recall_on_new_user_creates_profile() -> None:
    """Setting the recall toggle on a new user creates the profile.

    The CLI ``/memory recall on`` command calls this on users who may
    have never had any rules written. The helper must create a profile
    with the toggle set and an empty rules list.
    """

    store = OpenCouchMemoryStore()
    updated = await aset_proactive_recall(store, user_id="alice", enabled=True)

    assert updated.proactive_recall_enabled is True
    assert updated.rules == []


# ─── Rule cap eviction ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eviction_archives_oldest_when_exceeding_cap() -> None:
    """Adding a rule past MAX_ACTIVE_RULES evicts the oldest by added_at."""

    from agent.memory.procedural_profile import MAX_ACTIVE_RULES

    store = OpenCouchMemoryStore()

    # Seed MAX_ACTIVE_RULES rules with sequential timestamps.
    # Each rule uses unique wording to avoid reconciliation dedup.
    unique_topics = [
        "meditation",
        "breathing",
        "journaling",
        "walking",
        "reading",
        "cooking",
        "painting",
        "swimming",
        "gardening",
        "stretching",
        "singing",
        "dancing",
        "knitting",
        "cycling",
        "yoga",
        "running",
        "hiking",
        "drawing",
        "writing",
        "photography",
    ]
    for i in range(MAX_ACTIVE_RULES):
        rule = build_procedural_rule(
            rule_text=f"You prefer {unique_topics[i]} as a coping strategy.",
            evidence=[f"User mentioned {unique_topics[i]} specifically"],
        )
        rule = rule.model_copy(update={"added_at": f"2026-01-01T{i:02d}:00:00Z"})
        await aadd_procedural_rule(store, user_id="alice", rule=rule)

    profile = await aget_procedural_profile(store, user_id="alice")
    assert len(profile.rules) == MAX_ACTIVE_RULES
    assert len(profile.archived_rules) == 0

    # Add one more — should evict the oldest (meditation)
    overflow_rule = build_procedural_rule(
        rule_text="You dislike being interrupted during conversation.",
        evidence=["User said interruptions are frustrating"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=overflow_rule)

    profile = await aget_procedural_profile(store, user_id="alice")
    assert len(profile.rules) == MAX_ACTIVE_RULES
    assert len(profile.archived_rules) == 1
    assert "meditation" in profile.archived_rules[0].rule
    assert profile.archived_rules[0].superseded_by == "eviction"
    assert profile.archived_rules[0].dormant_at is not None
    assert "interrupted" in profile.rules[-1].rule


@pytest.mark.asyncio
async def test_aset_recall_preserves_existing_rules() -> None:
    """Toggling recall must not clobber existing rules.

    Parallel to ``test_aadd_rule_preserves_proactive_recall_setting``:
    the two additive helpers both need to preserve unrelated state
    from the load → mutate → put idiom.
    """

    store = OpenCouchMemoryStore()
    rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=rule)

    # Toggle recall — the rule should stay
    await aset_proactive_recall(store, user_id="alice", enabled=True)

    reloaded = await aget_procedural_profile(store, user_id="alice")
    assert reloaded.proactive_recall_enabled is True
    assert len(reloaded.rules) == 1
    assert reloaded.rules[0].rule == "You prefer short replies."


@pytest.mark.asyncio
async def test_adelete_procedural_rule_removes_exact_rule_and_preserves_toggle() -> (
    None
):
    """Deleting a rule should preserve the recall toggle."""

    store = OpenCouchMemoryStore()
    keep_rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    delete_rule = build_procedural_rule(
        rule_text="Don't suggest meditation again.",
        evidence=["Please don't suggest meditation again"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=keep_rule)
    await aadd_procedural_rule(store, user_id="alice", rule=delete_rule)
    await aset_proactive_recall(store, user_id="alice", enabled=True)

    deleted = await adelete_procedural_rule(
        store,
        user_id="alice",
        rule_id=delete_rule.id,
    )

    assert deleted is not None
    profile, removed_rule = deleted
    assert removed_rule.id == delete_rule.id
    assert profile.proactive_recall_enabled is True
    assert [rule.id for rule in profile.rules] == [keep_rule.id]


@pytest.mark.asyncio
async def test_aclear_procedural_rules_preserves_toggle() -> None:
    """Clearing active rules should not reset recall preferences."""

    store = OpenCouchMemoryStore()
    rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=rule)
    await aset_proactive_recall(store, user_id="alice", enabled=True)

    profile, cleared_count = await aclear_procedural_rules(store, user_id="alice")

    assert cleared_count == 1
    assert profile.rules == []
    assert profile.proactive_recall_enabled is True


@pytest.mark.asyncio
async def test_aget_recall_returns_false_for_new_user() -> None:
    """The recall toggle defaults to False for users with no profile.

    Matches the schema default. Prompt builders rely on this to
    decide whether to emit the "do not proactively reference memory"
    constraint for users who have never toggled anything.
    """

    store = OpenCouchMemoryStore()
    enabled = await aget_proactive_recall(store, user_id="new-user")
    assert enabled is False


@pytest.mark.asyncio
async def test_aget_recall_reflects_written_value() -> None:
    """The convenience reader matches whatever was written."""

    store = OpenCouchMemoryStore()
    await aset_proactive_recall(store, user_id="alice", enabled=True)
    assert await aget_proactive_recall(store, user_id="alice") is True

    await aset_proactive_recall(store, user_id="alice", enabled=False)
    assert await aget_proactive_recall(store, user_id="alice") is False


# ─── build_procedural_rule — default-shape helper ─────────────────────────


def test_build_rule_applies_v07_defaults() -> None:
    """The rule builder applies source and confidence defaults for v0.7.

    v0.7 only produces ``source="explicit_user"`` with
    ``confidence="high"`` rules. These defaults are centralized in
    the helper so the writer node doesn't have to spell them out on
    every call.
    """

    rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    assert rule.source == "explicit_user"
    assert rule.confidence == "high"
    assert rule.rule == "You prefer short replies."
    assert rule.evidence == ["Please keep it short"]
    assert rule.write_timing == "immediate"
    assert rule.write_reason == ""
    assert rule.policy_version == "phase1_v1"
    assert rule.added_at.endswith("Z")  # ISO-8601 UTC with Z suffix


def test_build_rule_overrides_defaults() -> None:
    """Callers can override source and confidence for non-default cases.

    The phase-4 consolidation path will call this with
    ``source="consolidation"`` and a lower confidence. The helper
    supports those overrides without requiring a separate builder.
    """

    rule = build_procedural_rule(
        rule_text="You often mention your sister Sarah.",
        evidence=["fact-123", "fact-456", "fact-789"],
        confidence="medium",
        source="consolidation",
    )
    assert rule.source == "consolidation"
    assert rule.confidence == "medium"
    assert rule.write_timing == "immediate"


# ─── Isolation — two users don't see each other's profiles ────────────────


@pytest.mark.asyncio
async def test_profiles_are_isolated_per_user() -> None:
    """Profile reads/writes for user A must not affect user B.

    Namespace isolation is a store-layer property, but we test it
    through the helpers to lock the contract at the API level.
    """

    store = OpenCouchMemoryStore()

    alice_rule = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short"],
    )
    await aadd_procedural_rule(store, user_id="alice", rule=alice_rule)
    await aset_proactive_recall(store, user_id="alice", enabled=True)

    # Bob has nothing — fresh empty profile
    bob_profile = await aget_procedural_profile(store, user_id="bob")
    assert bob_profile.rules == []
    assert bob_profile.proactive_recall_enabled is False

    # Alice still has her state
    alice_profile = await aget_procedural_profile(store, user_id="alice")
    assert len(alice_profile.rules) == 1
    assert alice_profile.proactive_recall_enabled is True
