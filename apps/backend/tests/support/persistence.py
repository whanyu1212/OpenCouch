"""Shared persistence-runtime test helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from agent.memory.models import (
    EntityRef,
    ExtractionResult,
    MemoryWrite,
    MoodArc,
    ProceduralExtractionResult,
    SessionArc,
    SummarizationResult,
)
from llm.base import BaseLLMClient, StructuredResponseT


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
            from agent.gates.safety.service import CrisisAssessmentSchema

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
            from agent.memory.models import DispatchDecision

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


def runtime_paths(tmp_path: Path) -> dict[str, Path]:
    """Return the three SQLite paths for a persistence runtime test.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        dict[str, Path]: Keyword arguments accepted by ``PersistentAgentRuntime``.
    """

    return {
        "sqlite_path": tmp_path / "threads.sqlite3",
        "memory_sqlite_path": tmp_path / "memory.sqlite3",
        "crisis_log_sqlite_path": tmp_path / "crisis.sqlite3",
    }


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
