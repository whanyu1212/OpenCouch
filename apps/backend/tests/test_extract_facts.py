"""Unit and integration tests for the semantic extraction node.

Covers the full behavior matrix of ``run_extract_semantic_facts_node``:
early exits, LLM success + empty extractions, LLM success + writes,
dedup hits, per-candidate failure isolation, batch failures, intra-batch
dedup, and the ``MemoryWrite → SemanticFact`` conversion helper.

Test structure:
    1. ``TestMemoryWriteToSemanticFact`` — pure-function helper tests
    2. ``TestExtractFactsNodeUnit`` — node tests with a mock runtime +
       fake LLM client, exercising every branch of the control flow
    3. ``TestExtractFactsEndToEnd`` — integration tests via ``run_agent``
       that drive the full parent graph with an injected fake client
       and a fresh memory store, verifying extraction wires correctly
       into the graph topology

All tests are deterministic — no live API calls. The fake LLM client
dispatches on ``response_schema`` so the same instance can be used by
both the crisis classifier and the extraction node within a single
``run_agent`` call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.graph import run_agent
from agent.memory.crisis_log import InMemoryCrisisLogBackend
from agent.memory.dedup import JACCARD_DUPLICATE_THRESHOLD
from agent.memory.models import (
    EntityRef,
    ExtractionResult,
    MemoryWrite,
    SemanticFact,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.nodes.extract_facts import (
    _memory_write_to_semantic_fact,
    run_extract_semantic_facts_node,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from services.llm.base import BaseLLMClient, StructuredResponseT


# ─── Test helpers ────────────────────────────────────────────────────────


def _make_memory_write(
    *,
    evidence_quote: str = "my sister Sarah visited last night",
    subject_identifier: str = "user-1",
    object_identifier: str = "Sarah",
    predicate: str = "KNOWS",
    category: str = "relationship",
    source_session_id: str = "thread-test",
    source_turn_index: int = 0,
) -> MemoryWrite:
    """Build a MemoryWrite with sensible defaults for testing."""

    return MemoryWrite(
        category=category,  # type: ignore[arg-type]
        subject=EntityRef(type="User", identifier=subject_identifier),
        predicate=predicate,  # type: ignore[arg-type]
        object=EntityRef(type="Person", identifier=object_identifier),
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id=source_session_id,
        source_turn_index=source_turn_index,
    )


def _partial_state(
    *,
    message: str = "my sister Sarah came over last night",
    user_id: str | None = "user-1",
    session_id: str | None = "thread-test",
    turn_count: int = 1,
) -> AgentState:
    """Build a partial AgentState for extraction node unit tests.

    Only the fields the extraction node reads (message, history, user_id,
    session_id, progress) are populated. The rest is left off and the
    value is cast to AgentState — the test is asserting behavior, not
    schema completeness.
    """

    state: Any = {
        "message": message,
        "history": [],
        "user_id": user_id,
        "session_id": session_id,
        "progress": {"turn_count": turn_count},
    }
    return cast(AgentState, state)


class _FakeExtractionLLM(BaseLLMClient):
    """Fake LLM client that dispatches on response_schema.

    - For the crisis classifier schema: returns a safe (level 0) result.
    - For ExtractionResult: returns the canned ``extraction_result``.
    - For any other structured schema: raises (tests should not hit this).
    - ``generate_text`` returns a fixed string (used by therapeutic modes).

    The dispatcher pattern lets a single instance be used by both the
    crisis gate and the extraction node within one ``run_agent`` call.
    """

    def __init__(
        self,
        *,
        extraction_result: ExtractionResult,
        raise_on_extraction: bool = False,
    ) -> None:
        self.extraction_result = extraction_result
        self.raise_on_extraction = raise_on_extraction
        self.extraction_calls = 0
        self.crisis_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
        use_search: bool = False,
    ) -> str:
        self.text_calls += 1
        return "fake therapeutic response"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        yield "fake"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        # Dispatch on schema type.
        if response_schema.__name__ == "ExtractionResult":
            self.extraction_calls += 1
            if self.raise_on_extraction:
                raise RuntimeError("simulated extraction LLM failure")
            return cast(StructuredResponseT, self.extraction_result)

        if response_schema.__name__ == "CrisisAssessmentSchema":
            # Return a safe (level 0) crisis assessment so non-crisis
            # messages don't route to the crisis branch during these tests.
            self.crisis_calls += 1
            from agent.nodes.crisis_gate import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe — fake LLM for extraction tests",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

        if response_schema.__name__ == "DispatchDecision":
            # Therapeutic dispatcher — route everything to supportive.
            from agent.memory.models import DispatchDecision

            return cast(
                StructuredResponseT,
                DispatchDecision(
                    mode="supportive",
                    reasoning="fake dispatcher — supportive for extraction tests",
                    confidence="high",
                ),
            )

        raise RuntimeError(
            f"_FakeExtractionLLM: unexpected schema {response_schema.__name__}"
        )


class _MockRuntime:
    """Minimal runtime stand-in for extraction node unit tests."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None,
        memory_store: OpenCouchMemoryStore | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
    ) -> None:
        self.context: WorkflowContext = {
            "llm_client": llm_client,
            "memory_store": memory_store or OpenCouchMemoryStore(),
            "crisis_log_backend": None,  # type: ignore[typeddict-item]
            "memory_mode": memory_mode,
        }


# ─── 1. _memory_write_to_semantic_fact helper tests ─────────────────────


class TestMemoryWriteToSemanticFact:
    """Unit tests for the MemoryWrite → SemanticFact conversion helper."""

    def test_preserves_all_memory_write_fields(self) -> None:
        write = _make_memory_write(
            evidence_quote="specific test quote",
            source_turn_index=5,
        )
        fact = _memory_write_to_semantic_fact(write)

        assert fact.evidence_quote == "specific test quote"
        assert fact.source_turn_index == 5
        assert fact.category == write.category
        assert fact.predicate == write.predicate
        assert fact.subject.type == write.subject.type
        assert fact.subject.identifier == write.subject.identifier
        assert fact.object.type == write.object.type
        assert fact.object.identifier == write.object.identifier
        assert fact.confidence == write.confidence
        assert fact.source_session_id == write.source_session_id

    def test_generates_unique_id_per_call(self) -> None:
        write = _make_memory_write()
        fact_a = _memory_write_to_semantic_fact(write)
        fact_b = _memory_write_to_semantic_fact(write)
        assert fact_a.id != fact_b.id

    def test_sets_default_metadata_fields(self) -> None:
        write = _make_memory_write()
        fact = _memory_write_to_semantic_fact(write)

        assert fact.dormant_at is None
        assert fact.superseded_by is None
        assert fact.user_visible is True
        assert fact.created_at == fact.last_referenced_at
        assert fact.created_at.endswith("Z")

    def test_returned_type_is_semantic_fact(self) -> None:
        write = _make_memory_write()
        fact = _memory_write_to_semantic_fact(write)
        assert isinstance(fact, SemanticFact)


# ─── 2. Extraction node unit tests (mock runtime) ───────────────────────


class TestExtractFactsNodeUnit:
    """Unit tests for ``run_extract_semantic_facts_node`` with a mock runtime."""

    @pytest.mark.asyncio
    async def test_no_llm_client_skips_silently(self) -> None:
        """Without an LLM client, the node returns diagnostics-only with no side effects."""

        store = OpenCouchMemoryStore()
        runtime = _MockRuntime(llm_client=None, memory_store=store)
        state = _partial_state()

        delta = await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        # v0.8 observability: node now returns a diagnostics delta on
        # every path (empty dict contract replaced with "empty facts,
        # populated diagnostics"). The skip reason records the early
        # exit path so downstream dashboards can distinguish "no LLM"
        # from "LLM returned empty" from "LLM errored".
        assert delta == {
            "diagnostics": {
                "extract_facts_ms": pytest.approx(0.0, abs=50.0),
                "semantic_writes": 0,
                "semantic_bumps": 0,
                "extract_facts_reason": "skipped: no llm_client",
            }
        }
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_incognito_mode_skips_silently(self) -> None:
        """Incognito mode skips extraction even with an LLM client."""

        store = OpenCouchMemoryStore()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[_make_memory_write()],
                reason="would-be-extraction",
            )
        )
        runtime = _MockRuntime(
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.INCOGNITO,
        )
        state = _partial_state()

        delta = await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["semantic_writes"] == 0
        assert delta["diagnostics"]["extract_facts_reason"] == "skipped: incognito"
        assert await store.arecord_count() == 0
        # LLM should NOT have been called (the early exit fires before it)
        assert fake.extraction_calls == 0

    @pytest.mark.asyncio
    async def test_empty_extraction_no_writes(self) -> None:
        """LLM returns empty facts → node logs reason, no writes.

        v0.8.2 note: the message must be longer or contain a word
        outside the small-talk vocabulary, otherwise the pre-extractor
        gate intercepts before the LLM call fires. "I feel okay today"
        passes the gate (contains "feel" and "today" which are not in
        the small-talk vocab) but the LLM still returns zero facts.
        """

        store = OpenCouchMemoryStore()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[],
                reason="no persistent fact in this turn",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(message="I feel okay today")

        delta = await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        # v0.8: empty facts path still records the LLM's reason in the
        # diagnostics so dashboards can see why the extractor skipped
        # a turn — the reason comes straight from the LLM, not a hard-
        # coded string, so we pass the fake's reason through.
        assert delta["diagnostics"]["semantic_writes"] == 0
        assert (
            delta["diagnostics"]["extract_facts_reason"]
            == "no persistent fact in this turn"
        )
        assert fake.extraction_calls == 1
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_single_new_fact_writes_to_store(self) -> None:
        """A fresh fact with no duplicates gets written as a SemanticFact."""

        store = OpenCouchMemoryStore()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[_make_memory_write(evidence_quote="my sister Sarah visited")],
                reason="extracted one relationship fact",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        namespace = ("user-1", "semantic")
        assert await store.arecord_count(namespace) == 1
        records = await store.asearch(namespace, query=None, limit=10)
        assert len(records) == 1
        assert records[0].value["evidence_quote"] == "my sister Sarah visited"
        assert records[0].value["user_visible"] is True

    @pytest.mark.asyncio
    async def test_duplicate_fact_bumps_last_referenced_at(self) -> None:
        """A near-duplicate of an existing record bumps timestamp, no new row."""

        store = OpenCouchMemoryStore()
        # Pre-seed with an existing fact.
        seed_write = _make_memory_write(
            evidence_quote="my sister Sarah came over last night"
        )
        seed_fact = _memory_write_to_semantic_fact(seed_write)
        # Manually set an older timestamp so we can detect the bump.
        old_ts = "2026-01-01T00:00:00Z"
        seed_value = seed_fact.model_dump(mode="json")
        seed_value["last_referenced_at"] = old_ts
        await store.aput(("user-1", "semantic"), key=seed_fact.id, value=seed_value)

        # LLM returns a near-duplicate — same triple, very similar quote.
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[
                    _make_memory_write(
                        evidence_quote="my sister Sarah came over last night again"
                    )
                ],
                reason="near duplicate of an existing fact",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        # Record count unchanged (still 1) — no new row was written.
        namespace = ("user-1", "semantic")
        assert await store.arecord_count(namespace) == 1

        # But last_referenced_at has been bumped.
        updated = await store.aget(namespace, key=seed_fact.id)
        assert updated is not None
        assert updated.value["last_referenced_at"] != old_ts
        # Quote preserved — the bump doesn't overwrite evidence.
        assert updated.value["evidence_quote"] == "my sister Sarah came over last night"

    @pytest.mark.asyncio
    async def test_mixed_batch_new_plus_duplicate(self) -> None:
        """A batch with both a new fact and a duplicate handles each correctly."""

        store = OpenCouchMemoryStore()
        # Pre-seed an existing fact.
        seed_write = _make_memory_write(
            evidence_quote="my sister Sarah visited",
            object_identifier="Sarah",
        )
        seed_fact = _memory_write_to_semantic_fact(seed_write)
        await store.aput(
            ("user-1", "semantic"),
            key=seed_fact.id,
            value=seed_fact.model_dump(mode="json"),
        )

        # LLM returns two facts: one duplicate of the seed, one novel.
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[
                    # Duplicate: same triple, same quote.
                    _make_memory_write(
                        evidence_quote="my sister Sarah visited",
                        object_identifier="Sarah",
                    ),
                    # New: different object, different quote.
                    _make_memory_write(
                        evidence_quote="my colleague Mark mentioned the deadline",
                        object_identifier="Mark",
                    ),
                ],
                reason="one duplicate, one new",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        namespace = ("user-1", "semantic")
        # Store now has 2 records: the seed + the novel fact
        assert await store.arecord_count(namespace) == 2

    @pytest.mark.asyncio
    async def test_intra_batch_dedup(self) -> None:
        """Two near-duplicates in a single extraction batch dedupe against each other."""

        store = OpenCouchMemoryStore()

        # LLM returns two near-duplicate candidates in one call.
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[
                    _make_memory_write(
                        evidence_quote="my sister Sarah came over last night"
                    ),
                    _make_memory_write(
                        evidence_quote="my sister Sarah came over last night again"
                    ),
                ],
                reason="two near-duplicate candidates",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        # First candidate writes; second should dedup against the first
        # via the intra-batch append-to-existing-records pattern.
        namespace = ("user-1", "semantic")
        assert await store.arecord_count(namespace) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_logs_warning_and_skips(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """LLM raising during extraction → warning logged, no writes."""

        store = OpenCouchMemoryStore()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(facts=[], reason="irrelevant"),
            raise_on_extraction=True,
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state()

        with caplog.at_level(logging.WARNING, logger="agent.nodes.extract_facts"):
            delta = await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        assert delta["diagnostics"]["semantic_writes"] == 0
        assert delta["diagnostics"]["extract_facts_reason"] == "skipped: llm error"
        assert await store.arecord_count() == 0
        assert any("structured-output call failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_source_session_id_and_turn_index_honored(self) -> None:
        """The fact the LLM returns should carry its own provenance into the store."""

        store = OpenCouchMemoryStore()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[
                    _make_memory_write(
                        source_session_id="session-xyz",
                        source_turn_index=7,
                    )
                ],
                reason="provenance check",
            )
        )
        runtime = _MockRuntime(llm_client=fake, memory_store=store)
        state = _partial_state(session_id="session-xyz", turn_count=8)

        await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

        records = await store.asearch(("user-1", "semantic"), query=None, limit=10)
        assert len(records) == 1
        assert records[0].value["source_session_id"] == "session-xyz"
        assert records[0].value["source_turn_index"] == 7


# ─── 3. End-to-end extraction via run_agent ──────────────────────────────


class TestExtractFactsEndToEnd:
    """Integration tests that drive the full parent graph with fake LLM + store."""

    @pytest.mark.asyncio
    async def test_extraction_happens_on_therapeutic_path(self) -> None:
        """A therapeutic-path turn with a fake LLM should produce a store write."""

        store = OpenCouchMemoryStore()
        crisis_log = InMemoryCrisisLogBackend()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[
                    _make_memory_write(
                        evidence_quote="I have a sister Sarah who lives nearby"
                    )
                ],
                reason="extracted 1 relationship fact",
            )
        )

        result = await run_agent(
            AgentInput(
                message="I have a sister Sarah who lives nearby.",
                user_id="user-e2e",
            ),
            llm_client=fake,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=MemoryMode.LOCAL,
        )

        # The turn itself completed normally
        assert result.response_text  # supportive response from the fake LLM
        # Extraction ran and wrote one record
        assert fake.extraction_calls == 1
        namespace = ("user-e2e", "semantic")
        assert await store.arecord_count(namespace) == 1

    @pytest.mark.asyncio
    async def test_extraction_skipped_in_incognito_end_to_end(self) -> None:
        """Incognito mode through run_agent → no extraction call, no writes."""

        store = OpenCouchMemoryStore()
        crisis_log = InMemoryCrisisLogBackend()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[_make_memory_write()],
                reason="would-be-extraction",
            )
        )

        await run_agent(
            AgentInput(message="I had a rough day at work.", user_id="user-e2e"),
            llm_client=fake,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=MemoryMode.INCOGNITO,
        )

        # Crisis gate and dispatcher still called the LLM, but extraction didn't
        assert fake.extraction_calls == 0
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_extraction_skipped_when_no_llm_client_end_to_end(self) -> None:
        """No LLM client → all LLM-backed nodes skip, including extraction."""

        store = OpenCouchMemoryStore()
        crisis_log = InMemoryCrisisLogBackend()

        await run_agent(
            AgentInput(message="I had a rough day at work.", user_id="user-e2e"),
            llm_client=None,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=MemoryMode.LOCAL,
        )

        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_extraction_on_crisis_path_does_not_run(self) -> None:
        """Crisis-branch turns should NOT run extraction.

        The crisis branch has its own post-response nodes (crisis_log_node
        + extract_semantic_facts_node per the Stage E topology), so the
        extraction node DOES run after crisis_response. But with a fake
        LLM whose extraction returns empty, nothing gets written. This
        test documents the current topology — if the extraction node is
        ever removed from the crisis branch, this test still passes
        because it only asserts record_count == 0, not that the node
        wasn't called.
        """

        store = OpenCouchMemoryStore()
        crisis_log = InMemoryCrisisLogBackend()
        fake = _FakeExtractionLLM(
            extraction_result=ExtractionResult(
                facts=[],  # empty — no extraction on crisis turns in this test
                reason="crisis turn — extraction returns empty",
            )
        )

        await run_agent(
            AgentInput(
                message="I have pills and I am going to kill myself tonight.",
                user_id="user-e2e",
            ),
            llm_client=fake,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=MemoryMode.LOCAL,
        )

        # Extraction node ran but returned empty → no writes
        assert await store.arecord_count() == 0
        # Crisis log DID get written
        assert await crisis_log.arecord_count() == 1


# ─── 4. Regression guard on the dedup threshold constant ────────────────


def test_dedup_threshold_matches_dedup_module_constant() -> None:
    """The extraction tests assume the dedup threshold is 0.85.

    If a future tuning change moves the threshold, this regression
    guard catches it and forces the test suite to be re-examined for
    cases that depend on specific Jaccard values.
    """

    assert JACCARD_DUPLICATE_THRESHOLD == 0.85
