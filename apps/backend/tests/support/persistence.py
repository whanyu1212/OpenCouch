"""Shared persistence-runtime test helpers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from agent.memory.types import (
    EntityRef,
    ExtractionResult,
    MemoryWrite,
    MoodArc,
    ProceduralExtractionResult,
    SessionArc,
    SummarizationResult,
)
from agent.runtime import RuntimeStoragePaths
from llm.base import BaseLLMClient, StructuredResponseT

_POSTGRES_TEST_URL_ENV = "OPENCOUCH_TEST_POSTGRES_URL"
_POSTGRES_TESTS_ENABLED_ENV = "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS"


def postgres_database_url() -> str | None:
    """Return the explicitly enabled DSN for opt-in Postgres tests."""

    if os.getenv(_POSTGRES_TESTS_ENABLED_ENV) != "1":
        return None
    return os.getenv(_POSTGRES_TEST_URL_ENV)


async def truncate_postgres_tables(dsn: str, *tables: str) -> None:
    """Truncate the named Postgres tables, skipping any that do not exist.

    Used to isolate opt-in Postgres integration tests that share one database:
    each test starts from empty shared tables instead of inheriting rows from
    whatever ran before. Absent tables are skipped so the helper is safe to call
    before any schema has been created.

    Args:
        dsn (str): PostgreSQL connection string.
        tables (str): Table names to truncate (identity sequences are reset).
    """

    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cursor:
            for table in tables:
                await cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = %s) AS present",
                    (table,),
                )
                row = await cursor.fetchone()
                if row and row["present"]:
                    await cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")


class FakeCrossRestartLLM(BaseLLMClient):
    """Deterministic LLM client for persistence runtime smoke tests.

    Dispatches on ``response_schema.__name__`` and returns canned values for
    every structured-output call the graph makes. The class tracks call counts
    per schema type so tests can assert runtime behavior.
    """

    def __init__(
        self,
        *,
        extraction_result: ExtractionResult | None = None,
        procedural_result: ProceduralExtractionResult | None = None,
        summarization_result: SummarizationResult | None = None,
    ) -> None:
        self.extraction_result = (
            extraction_result or _default_cross_restart_extraction_result()
        )
        self.procedural_result = procedural_result or ProceduralExtractionResult(
            rules=[],
            reason="no procedural rules",
        )
        self.summarization_result = (
            summarization_result or _default_cross_restart_summarization_result()
        )
        self.crisis_calls = 0
        self.dispatch_calls = 0
        self.extraction_calls = 0
        self.procedural_calls = 0
        self.summarization_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.text_calls += 1
        return "I hear you. Tell me more about what's on your mind."

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake stream chunk"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__

        if schema_name == "CrisisAssessmentSchema":
            self.crisis_calls += 1
            from agent.guardrails.service import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe - fake LLM for persistence smoke test",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

        if schema_name == "DispatchDecision":
            self.dispatch_calls += 1
            from agent.memory.types import DispatchDecision

            return cast(
                StructuredResponseT,
                DispatchDecision(
                    response_style="supportive",
                    exercise_start_basis="ambiguous_or_none",
                    reasoning="fake dispatcher for persistence smoke test",
                    confidence="high",
                ),
            )

        if schema_name == "ExerciseStepDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                step_state="complete",
                reasoning="fake guided-exercise step classifier",
                confidence="high",
            )

        if schema_name == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                active_flow_action=self._active_flow_action_for_prompt(prompt),
                reasoning="ordinary persistence test turn",
                confidence="high",
            )

        if schema_name == "ExtractionResult":
            self.extraction_calls += 1
            return cast(StructuredResponseT, self.extraction_result)

        if schema_name == "ProceduralExtractionResult":
            self.procedural_calls += 1
            return cast(StructuredResponseT, self.procedural_result)

        if schema_name == "SemanticWritePolicyDecision":
            action = (
                "commit_at_session_end"
                if "family conflict is a big trigger" in prompt.lower()
                else "commit_now"
            )
            return response_schema(  # type: ignore[call-arg,return-value]
                action=action,
                reason="fake semantic write policy for persistence tests",
                confidence="high",
            )

        if schema_name == "ProceduralWritePolicyDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="commit_now",
                reason="fake procedural write policy for persistence tests",
                confidence="high",
            )

        if schema_name == "SemanticReconciliationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="coexist",
                record_indexes=[],
                reason="fake semantic reconciliation for persistence tests",
                confidence="high",
            )

        if schema_name == "ProceduralReconciliationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="append",
                replace_indexes=[],
                reason="fake procedural reconciliation for persistence tests",
                confidence="high",
            )

        if schema_name == "SummarizationResult":
            self.summarization_calls += 1
            return cast(StructuredResponseT, self.summarization_result)

        if schema_name == "TherapeuticResponseLLMOutput":
            return response_schema(  # type: ignore[call-arg,return-value]
                response_text="I hear you. Tell me more about what's on your mind."
            )

        raise RuntimeError(f"FakeCrossRestartLLM: unexpected schema {schema_name}")

    def _active_flow_action_for_prompt(self, prompt: str) -> str:
        """Return active-flow action for deterministic persistence tests.

        Args:
            prompt (str): Turn-dispatch prompt.

        Returns:
            str: Active-flow action for the fake dispatch decision.
        """

        if "Active flow: guided_exercise" in prompt:
            return "clear"
        if "Active flow: pending_memory_action" in prompt:
            return "clear"
        return "none"


def runtime_storage_paths(tmp_path: Path) -> RuntimeStoragePaths:
    """Return grouped SQLite paths for a persistence runtime test."""

    return RuntimeStoragePaths(
        sqlite_path=tmp_path / "threads.sqlite3",
        memory_sqlite_path=tmp_path / "memory.sqlite3",
        crisis_log_sqlite_path=tmp_path / "crisis.sqlite3",
        feedback_sqlite_path=tmp_path / "feedback.sqlite3",
        text_session_sqlite_path=tmp_path / "text_sessions.sqlite3",
    )


def in_memory_runtime_storage_paths() -> RuntimeStoragePaths:
    """Return grouped in-memory SQLite paths for runtime tests."""

    return RuntimeStoragePaths(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
    )


def _default_cross_restart_extraction_result() -> ExtractionResult:
    """Build the default relationship extraction for persistence smoke tests.

    Returns:
        ExtractionResult: Canned extraction result.
    """

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


def _default_cross_restart_summarization_result() -> SummarizationResult:
    """Build the default session summary for persistence smoke tests.

    Returns:
        SummarizationResult: Canned summarization result.
    """

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
        reason="captured 2-turn family arc for persistence smoke test",
    )
