"""End-to-end cross-restart persistence smoke tests (v0.8 Stage E).

These tests validate the full v0.8 contract: data written by one
``PersistentAgentRuntime`` instance survives the runtime's close
and comes back when a second runtime opens the same SQLite files.

This is the test suite that proves "`/memory list` is not empty
after CLI restart" — the user-visible fix v0.8 was scoped to deliver.
It exercises the entire stack:

- ``PersistentAgentRuntime.__init__`` picking SqliteMemoryStore +
  SqliteCrisisLogBackend based on memory_mode (Stage D wiring)
- ``run_turn`` invoking the full LangGraph workflow, which calls
  ``load_memory_node``, the therapeutic dispatcher, a response
  node, and ``extract_semantic_facts_node`` (Stage B reads/writes)
- ``end_session`` invoking the summarizer, which writes an episodic
  arc (Stage C writes)
- Runtime close via ``__aexit__`` releasing the SQLite connections
  without corrupting the data
- A second runtime instance pointing at the same paths, re-opening
  the SQLite files, and reading back every record

What these tests DON'T cover (intentionally):
- LLM quality — all tests use a deterministic fake LLM dispatcher
  that produces canned extractions and summaries
- Specific prompt behavior — that's what the individual node test
  suites cover
- CLI interaction — that's covered by the manual dogfood loop and
  ``test_opencouch_cli.py``

Test strategy: build a fake LLM that dispatches on
``response_schema.__name__`` and returns canned results for every
structured-output call the graph makes (crisis classifier, dispatcher,
extraction, summarization). Use ``tmp_path`` fixtures so all three
SQLite files are isolated per test and clean up automatically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest

from agent.memory.models import (
    EntityRef,
    ExtractionResult,
    MemoryWrite,
    MoodArc,
    SessionArc,
    StoredSessionArc,
    SummarizationResult,
)
from agent.memory.modes import MemoryMode
from agent.memory.sqlite_store import SqliteMemoryStore
from agent.models import Channel
from agent.persistence import PersistentAgentRuntime
from services.llm.base import BaseLLMClient, StructuredResponseT


# ─── Fake LLM client ───────────────────────────────────────────────────


class _FakeCrossRestartLLM(BaseLLMClient):
    """Deterministic LLM client for cross-restart smoke tests.

    Dispatches on ``response_schema.__name__`` and returns canned
    values for every structured-output call the graph makes. The
    class tracks call counts per schema type so tests can assert
    things like "extraction was called exactly once on this turn."

    Canned behaviors:
    - CrisisAssessmentSchema → level 0, no crisis
    - DispatchDecision → supportive mode
    - ExtractionResult → configurable (defaults to a single
      relationship fact about Sarah)
    - SummarizationResult → configurable (defaults to an arc
      summarizing the session as "user talked about Sarah")
    - generate_text → fixed supportive reply
    """

    def __init__(
        self,
        *,
        extraction_result: ExtractionResult | None = None,
        summarization_result: SummarizationResult | None = None,
    ) -> None:
        self.extraction_result = extraction_result or _default_extraction_result()
        self.summarization_result = (
            summarization_result or _default_summarization_result()
        )
        self.crisis_calls = 0
        self.dispatch_calls = 0
        self.extraction_calls = 0
        self.summarization_calls = 0
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
        return "I hear you. Tell me more about what's on your mind."

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        yield "fake stream chunk"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__

        if schema_name == "CrisisAssessmentSchema":
            self.crisis_calls += 1
            from agent.nodes.crisis_gate import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe — fake LLM for cross-restart smoke test",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

        if schema_name == "DispatchDecision":
            self.dispatch_calls += 1
            from agent.memory.models import DispatchDecision

            return cast(
                StructuredResponseT,
                DispatchDecision(
                    mode="supportive",
                    reasoning="fake dispatcher for cross-restart smoke test",
                    confidence="high",
                ),
            )

        if schema_name == "ExtractionResult":
            self.extraction_calls += 1
            return cast(StructuredResponseT, self.extraction_result)

        if schema_name == "SummarizationResult":
            self.summarization_calls += 1
            return cast(StructuredResponseT, self.summarization_result)

        raise RuntimeError(f"_FakeCrossRestartLLM: unexpected schema {schema_name}")


def _default_extraction_result() -> ExtractionResult:
    """Build a canned extraction result with one Sarah-relationship fact."""

    return ExtractionResult(
        facts=[
            MemoryWrite(
                category="relationship",
                subject=EntityRef(type="User", identifier="test-user"),
                predicate="KNOWS",
                object=EntityRef(type="Person", identifier="Sarah"),
                evidence_quote="I have a sister named Sarah",
                confidence="high",
                source_session_id="thread-a",
                source_turn_index=0,
            )
        ],
        reason="extracted 1 relationship fact about Sarah",
    )


def _default_summarization_result() -> SummarizationResult:
    """Build a canned summarization result with a simple session arc."""

    arc = SessionArc(
        session_id="thread-a",
        started_at="2026-04-11T09:00:00Z",
        ended_at="2026-04-11T09:15:00Z",
        duration_seconds=900,
        turn_count=2,
        primary_themes=["family"],
        summary="User mentioned their sister Sarah in passing.",
        mood_arc=MoodArc(opened="curious", closed="grounded"),
        open_loops=[],
        resolved_threads=[],
    )
    return SummarizationResult(
        arc=arc,
        reason="captured 2-turn family arc for cross-restart smoke test",
    )


def _empty_extraction_result() -> ExtractionResult:
    """Build a canned extraction result with zero facts.

    Used by tests that want to simulate "no facts extracted this
    turn" without the dispatcher thinking the whole LLM call failed.
    """

    return ExtractionResult(
        facts=[],
        reason="no extractable facts — fake LLM empty path",
    )


# ─── Helpers ───────────────────────────────────────────────────────────


def _runtime_paths(tmp_path: Path) -> dict[str, Path]:
    """Return a dict of the three SQLite paths for a smoke-test runtime.

    Keeping them as a dict rather than positional args makes the test
    callsites more readable and lets us pass them straight through
    to ``PersistentAgentRuntime`` via ``**kwargs`` style unpacking.
    """

    return {
        "sqlite_path": tmp_path / "threads.sqlite3",
        "memory_sqlite_path": tmp_path / "memory.sqlite3",
        "crisis_log_sqlite_path": tmp_path / "crisis.sqlite3",
    }


# ─── Smoke tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_semantic_facts_survive_runtime_close_and_reopen(tmp_path: Path) -> None:
    """The core v0.8 contract: extract a fact in runtime A, close A,
    open runtime B against the same SQLite files, and verify the fact
    is still readable.

    This is the test that would have caught the v0.4 asymmetric
    persistence bug if it had existed. Now that SQLite backing is
    wired in Stage D, this should pass on the first run."""

    paths = _runtime_paths(tmp_path)

    # ── Runtime A: write a fact, then close ───────────────────────────
    llm_a = _FakeCrossRestartLLM()
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_a:
        result = await runtime_a.run_turn(
            thread_id="thread-a",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            llm_client=llm_a,
        )
        # Sanity check: the turn ran end-to-end and produced output.
        assert result.output.response_text
        # Extraction ran once (crisis_calls is 1 for the crisis gate)
        assert llm_a.extraction_calls == 1
        # The fact landed in the SQLite store
        assert await runtime_a.memory_store.arecord_count() == 1

    # Runtime A is now closed. The SQLite file still exists on disk.
    assert paths["memory_sqlite_path"].exists()

    # ── Runtime B: open the same files, verify the fact came back ─────
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        # The record count should be 1 BEFORE we do anything — the
        # Stage B SqliteMemoryStore reads the existing file on first
        # async call, and the schema DDL is idempotent (CREATE TABLE
        # IF NOT EXISTS) so the existing data is untouched.
        count = await runtime_b.memory_store.arecord_count()
        assert count == 1, f"expected 1 record after restart, got {count}"

        # And we can retrieve it via asearch with a paraphrased query
        results = await runtime_b.memory_store.asearch(
            ("thread-a", "semantic"),
            query="tell me about Sarah",
        )
        assert len(results) == 1
        assert "Sarah" in results[0].value["evidence_quote"]


@pytest.mark.asyncio
async def test_episodic_arc_survives_runtime_close_and_reopen(
    tmp_path: Path,
) -> None:
    """The episodic counterpart to the semantic persistence test.
    An arc written by ``end_session`` in runtime A should be
    readable via ``asearch`` in runtime B."""

    paths = _runtime_paths(tmp_path)

    # Runtime A: run a turn, end the session, close
    llm_a = _FakeCrossRestartLLM()
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_a:
        await runtime_a.run_turn(
            thread_id="thread-a",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            llm_client=llm_a,
        )
        stored_arc = await runtime_a.end_session("thread-a", llm_client=llm_a)
        assert stored_arc is not None, "summarizer should have produced an arc"
        assert isinstance(stored_arc, StoredSessionArc)
        # The episodic record count should be 1
        assert await runtime_a.memory_store.arecord_count(("thread-a", "episodic")) == 1

    # Runtime B: verify the arc is still there
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        arcs = await runtime_b.memory_store.asearch(
            ("thread-a", "episodic"),
            query=None,
            limit=10,
        )
        assert len(arcs) == 1
        arc_value = arcs[0].value
        assert arc_value["summary"] == ("User mentioned their sister Sarah in passing.")
        assert arc_value["primary_themes"] == ["family"]


@pytest.mark.asyncio
async def test_crisis_log_survives_runtime_close_and_reopen(
    tmp_path: Path,
) -> None:
    """Persistence for the crisis log backend. Writes a crisis event
    in runtime A, closes, reopens B, verifies the event is still
    countable. Uses a direct append to the backend rather than
    routing a crisis message through the graph, because the focus
    here is the SQLite persistence contract, not the crisis gate's
    dispatch logic (which is covered by test_crisis_log.py)."""

    paths = _runtime_paths(tmp_path)

    # Runtime A: write a crisis record directly to the backend
    from agent.memory.models import CrisisLogRecord

    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_a:
        record = CrisisLogRecord(
            id="rec-cross-restart-1",
            session_id_opaque="a" * 64,
            user_id_or_null="test-user",
            detected_at="2026-04-11T10:00:00Z",
            level=2,
            override_kind="none",
            classifier_path="deterministic",
            reason="cross-restart smoke test record",
            response_node_completed=True,
            llm_failure_occurred=False,
        )
        await runtime_a.crisis_log_backend.aappend(record)
        assert await runtime_a.crisis_log_backend.arecord_count() == 1

    # Runtime B: reopen, verify the count is still 1
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        assert await runtime_b.crisis_log_backend.arecord_count() == 1
        # And we can look it up by date
        from datetime import date

        day_records = await runtime_b.crisis_log_backend.alist_by_date(
            date(2026, 4, 11)
        )
        assert len(day_records) == 1
        assert day_records[0].id == "rec-cross-restart-1"
        assert day_records[0].level == 2


@pytest.mark.asyncio
async def test_all_three_layers_persist_across_full_lifecycle(
    tmp_path: Path,
) -> None:
    """End-to-end test exercising all three persistence layers in
    one go: LangGraph thread checkpointer, memory store, crisis log.
    This is the authoritative "does v0.8 actually work" test — if
    this passes, the v0.8 promise is kept.

    Flow:
    1. Runtime A: run 2 turns on thread-a
    2. Runtime A: end session (writes episodic arc)
    3. Runtime A: close
    4. Runtime B: open same files
    5. Verify thread-a's transcript was restored by LangGraph
    6. Verify the 1 semantic fact is in the memory store
    7. Verify the 1 episodic arc is in the memory store
    8. Run a new turn on thread-a via runtime B to exercise the
       full graph pipeline against the persisted data
    """

    paths = _runtime_paths(tmp_path)

    # ── Runtime A: write everything ───────────────────────────────────
    llm_a = _FakeCrossRestartLLM()
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_a:
        # Turn 1: substantive message that triggers extraction
        await runtime_a.run_turn(
            thread_id="thread-a",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            llm_client=llm_a,
        )
        # Turn 2: a follow-up — the fake extractor produces the same
        # canned result, but dedup should catch it as a duplicate and
        # bump last_referenced_at instead of writing a second row.
        # (The fake ExtractionResult has the exact same evidence_quote
        # and triple, so find_near_duplicate should match.)
        await runtime_a.run_turn(
            thread_id="thread-a",
            message="tell me more about Sarah",
            channel=Channel.TEST,
            llm_client=llm_a,
        )
        # End the session — this writes an episodic arc
        arc = await runtime_a.end_session("thread-a", llm_client=llm_a)
        assert arc is not None

        # Before close: semantic=1 (dedup held), episodic=1
        semantic_count = await runtime_a.memory_store.arecord_count(
            ("thread-a", "semantic")
        )
        episodic_count = await runtime_a.memory_store.arecord_count(
            ("thread-a", "episodic")
        )
        assert semantic_count == 1, (
            f"expected 1 semantic fact after dedup, got {semantic_count}"
        )
        assert episodic_count == 1

    # ── Runtime B: reopen everything and verify ───────────────────────
    llm_b = _FakeCrossRestartLLM()
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        # LangGraph thread checkpointer restored the transcript
        history = await runtime_b.get_history("thread-a")
        # The transcript should have at least the 2 user turns and
        # 2 assistant replies from runtime A. It may also have the
        # finalize_turn appends. We just check it's non-empty.
        assert len(history) > 0

        # Memory store restored the semantic fact
        assert await runtime_b.memory_store.arecord_count(("thread-a", "semantic")) == 1
        # And the episodic arc
        assert await runtime_b.memory_store.arecord_count(("thread-a", "episodic")) == 1

        # Run a new turn on thread-a through runtime B. This exercises
        # the full graph against the persisted data — load_memory_node
        # will query the SQLite store and should find the Sarah fact
        # via token-recall retrieval.
        result = await runtime_b.run_turn(
            thread_id="thread-a",
            message="what did I say about Sarah",
            channel=Channel.TEST,
            llm_client=llm_b,
        )
        assert result.output.response_text
        # The final state's working_memory should contain the Sarah
        # fact because the query overlaps tokens with the stored
        # evidence quote.
        working_memory = result.state.get("working_memory", [])
        assert any("Sarah" in entry for entry in working_memory), (
            f"expected working_memory to contain a Sarah reference, "
            f"got {working_memory}"
        )


@pytest.mark.asyncio
async def test_fresh_thread_after_restart_sees_prior_records_in_same_namespace(
    tmp_path: Path,
) -> None:
    """When the second runtime uses a DIFFERENT thread_id but the
    same ``user_id`` (explicitly passed), the memory store should
    still serve records written under that user_id. This exercises
    the catch-up-at-first-turn path.

    Note: the current CLI doesn't expose a user_id flag, so this
    path is only reachable via scripted callers that pass user_id
    explicitly. Keeping the test anyway because the scripted path
    IS the v0.4 design intent — the CLI-accessible surface is a
    separate product-level gap documented in the ROADMAP.
    """

    paths = _runtime_paths(tmp_path)

    # Runtime A: write a record under user-alice via thread-old
    llm_a = _FakeCrossRestartLLM(
        extraction_result=ExtractionResult(
            facts=[
                MemoryWrite(
                    category="relationship",
                    subject=EntityRef(type="User", identifier="user-alice"),
                    predicate="KNOWS",
                    object=EntityRef(type="Person", identifier="Emma"),
                    evidence_quote="My best friend Emma lives nearby",
                    confidence="high",
                    source_session_id="thread-old",
                    source_turn_index=0,
                )
            ],
            reason="fake extractor for cross-thread test",
        )
    )
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_a:
        await runtime_a.run_turn(
            thread_id="thread-old",
            message="My best friend Emma lives nearby",
            channel=Channel.TEST,
            user_id="user-alice",
            llm_client=llm_a,
        )
        # Record should land under owner_id="user-alice"
        assert (
            await runtime_a.memory_store.arecord_count(("user-alice", "semantic")) == 1
        )

    # Runtime B: open, run on a FRESH thread_id but SAME user_id
    llm_b = _FakeCrossRestartLLM(extraction_result=_empty_extraction_result())
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        # The Emma record should still be there (owner_id keyed)
        assert (
            await runtime_b.memory_store.arecord_count(("user-alice", "semantic")) == 1
        )

        # Run a fresh-thread turn with the SAME user_id. The history
        # is empty (this thread_id is new), so is_first_turn=True
        # and the catch-up path SHOULD fire. But catch-up only
        # fetches episodic arcs, not semantic facts — so the
        # retrieval hits here come from token-recall instead.
        result = await runtime_b.run_turn(
            thread_id="thread-new",
            message="remind me about my best friend Emma",
            channel=Channel.TEST,
            user_id="user-alice",
            llm_client=llm_b,
        )

        # The working_memory should contain Emma because the query
        # shares the token "Emma" with the stored evidence quote.
        working_memory = result.state.get("working_memory", [])
        assert any("Emma" in entry for entry in working_memory), (
            f"expected working_memory to contain an Emma reference, "
            f"got {working_memory}"
        )


@pytest.mark.asyncio
async def test_incognito_runtime_does_not_persist_to_disk(
    tmp_path: Path,
) -> None:
    """The contrapositive test: in incognito mode, nothing should
    hit disk. This is the privacy contract — even with SQLite paths
    configured, incognito mode uses in-memory backings and the files
    should not exist after the runtime closes."""

    paths = _runtime_paths(tmp_path)

    llm = _FakeCrossRestartLLM()
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.INCOGNITO,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-incognito",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            llm_client=llm,
        )
        # In-memory store — the runtime is holding an OpenCouchMemoryStore
        # instance, NOT a SqliteMemoryStore
        assert not isinstance(runtime.memory_store, SqliteMemoryStore)

    # After the runtime closes, the SQLite files should NOT exist.
    # The LangGraph thread checkpointer in incognito mode uses
    # ``:memory:`` as its sqlite_path, so no file is created.
    # The memory store and crisis log are in-memory only, so their
    # files are never opened either.
    assert not paths["memory_sqlite_path"].exists()
    assert not paths["crisis_log_sqlite_path"].exists()


@pytest.mark.asyncio
async def test_schema_idempotent_across_multiple_opens(tmp_path: Path) -> None:
    """Opening the same SQLite file repeatedly should be safe. The
    ``CREATE TABLE IF NOT EXISTS`` DDL is idempotent by design, so
    the second, third, fourth runtimes pointing at the same file
    should all work without tripping constraint errors."""

    paths = _runtime_paths(tmp_path)
    llm = _FakeCrossRestartLLM()

    # Open and close the runtime 3 times in a row, writing a different
    # fact each time. Each reopen runs the schema DDL, which should
    # be a no-op for the already-created table.
    for i in range(3):
        async with PersistentAgentRuntime(
            **paths,
            memory_mode=MemoryMode.LOCAL,
        ) as runtime:
            # Use a unique thread_id per iteration so the LangGraph
            # checkpointer doesn't accumulate cross-thread state.
            await runtime.run_turn(
                thread_id=f"thread-{i}",
                message=f"iteration {i} message",
                channel=Channel.TEST,
                llm_client=llm,
            )

    # After 3 iterations, check the total record count via a fresh runtime
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as final_runtime:
        # Each iteration's extraction produces the canned Sarah fact
        # under its own thread_id namespace, so we should have 3
        # records total across 3 different namespaces.
        total = await final_runtime.memory_store.arecord_count()
        assert total == 3, f"expected 3 records after 3 iterations, got {total}"
