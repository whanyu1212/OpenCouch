"""Evaluate PersistentAgentRuntime persistence trajectories."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from eval.judges.rubric import RubricDimension, RubricJudgeArtifact, RubricLLMJudge
from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.grounded_lookup_fixtures import factual_lookup_fixture_answer
from eval.runners.memory_control_common import (
    grade_store_expectations,
    memory_snapshot,
    seed_memory_store,
)
from eval.runners.therapeutic_common import build_live_therapeutic_llms

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "runtime" / "persistence_trajectory_v1.json"
)
_DEFAULT_MIN_JUDGE_SCORE = 0.8

PersistenceBackend = str


@dataclass(frozen=True)
class RuntimePersistenceStep:
    """One runtime operation in a persistence trajectory."""

    type: str
    thread_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    label: str | None = None
    source: str | None = None
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePersistenceCase:
    """Parsed runtime persistence trajectory case."""

    id: str
    description: str = ""
    modes: list[str] = field(default_factory=lambda: ["scripted"])
    runtime: dict[str, Any] = field(default_factory=dict)
    memory_seed: dict[str, Any] = field(default_factory=dict)
    steps: list[RuntimePersistenceStep] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePaths:
    """Per-case SQLite fallback paths."""

    thread: Path
    memory: Path
    crisis: Path
    feedback: Path


class ScriptedRuntimeLLM:
    """Scripted LLM client for deterministic runtime trajectories."""

    def __init__(self, scripted: Mapping[str, Any]) -> None:
        self.scripted = dict(scripted)
        self.structured_calls: dict[str, int] = {}
        self.text_calls: int = 0
        self.stream_calls: int = 0

    async def generate_text(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
        use_search: bool = False,  # noqa: ARG002 - LLM protocol
    ) -> str:
        self.text_calls += 1
        return str(self.scripted.get("response_text", "scripted runtime response"))

    async def generate_text_stream(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        chunks = self.scripted.get("response_chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                yield str(chunk)
            return
        yield str(self.scripted.get("response_text", "scripted runtime response"))

    async def generate_structured(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        response_schema: type[Any],
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
        use_search: bool = False,  # noqa: ARG002 - LLM protocol
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )

        if schema_name == "CrisisAssessmentSchema":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "crisis",
                    {
                        "level": 0,
                        "confidence": "high",
                        "reason": "scripted safe runtime eval turn",
                        "needs_crisis_response": False,
                        "needs_clarification": False,
                    },
                )
            )
        if schema_name == "TurnDispatchDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "turn_dispatch",
                    {
                        "route": "therapeutic",
                        "reasoning": "scripted therapeutic runtime eval route",
                        "confidence": "high",
                        "active_flow_action": "none",
                    },
                )
            )
        if schema_name == "DispatchDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "therapeutic_dispatch",
                    {
                        "response_style": "supportive",
                        "therapeutic_approach": "none",
                        "exercise_start_basis": "ambiguous_or_none",
                        "reasoning": "scripted supportive response",
                        "confidence": "high",
                    },
                )
            )
        if schema_name == "ExerciseSelectionDecision":
            selected = self.scripted.get("exercise_selection")
            if isinstance(selected, Mapping):
                return response_schema(**selected)
            if selected is None:
                raise RuntimeError("Scripted runtime step needs exercise_selection.")
            return response_schema(
                exercise_type=str(selected),
                reasoning="scripted exercise selection",
                confidence="high",
            )
        if schema_name == "ExerciseStepDecision":
            step_state = self.scripted.get("step_state")
            if step_state is None:
                raise RuntimeError("Scripted runtime step needs step_state.")
            return response_schema(
                step_state=str(step_state),
                reasoning="scripted exercise step decision",
                confidence="high",
            )
        if schema_name == "PreferenceRuleDecision":
            return response_schema(
                rule_text=str(
                    self.scripted.get("preference_rule_text", "Use concise replies.")
                ),
                reasoning="scripted preference rule",
                confidence="high",
            )
        if schema_name == "ExtractionResult":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "semantic_extraction",
                    {"facts": [], "reason": "no scripted semantic facts"},
                )
            )
        if schema_name == "ProceduralExtractionResult":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "procedural_extraction",
                    {"rules": [], "reason": "no scripted procedural rules"},
                )
            )
        if schema_name == "SemanticWritePolicyDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "semantic_write_policy",
                    {
                        "action": "commit_now",
                        "reason": "scripted runtime eval write policy",
                        "confidence": "high",
                    },
                )
            )
        if schema_name == "ProceduralWritePolicyDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "procedural_write_policy",
                    {
                        "action": "commit_now",
                        "reason": "scripted runtime eval write policy",
                        "confidence": "high",
                    },
                )
            )
        if schema_name == "SemanticReconciliationDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "semantic_reconciliation",
                    {
                        "action": "coexist",
                        "record_indexes": [],
                        "reason": "scripted runtime eval reconciliation",
                        "confidence": "high",
                    },
                )
            )
        if schema_name == "ProceduralReconciliationDecision":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "procedural_reconciliation",
                    {
                        "action": "append",
                        "replace_indexes": [],
                        "reason": "scripted runtime eval reconciliation",
                        "confidence": "high",
                    },
                )
            )
        if schema_name == "SummarizationResult":
            return response_schema(
                **_mapping_or_default(
                    self.scripted,
                    "summarization",
                    {"arc": None, "reason": "scripted runtime eval thin session"},
                )
            )

        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


class RuntimePersistenceTrajectoryEvaluator(BaseEvaluator[RuntimePersistenceCase]):
    """Run runtime persistence trajectory checks."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        mode: str,
        backend: PersistenceBackend,
        database_url: str | None,
        judge_mode: str,
        min_judge_score: float | None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"runtime_persistence_trajectory_{backend}_{mode}_{judge_mode}",
        )
        self.mode = mode
        self.backend = backend
        self.database_url = database_url
        self.judge_mode = judge_mode
        self.min_judge_score = min_judge_score
        self._live_llms: tuple[Any, Any] | None = None

    def load_cases(self) -> list[RuntimePersistenceCase]:
        """Load only cases that opt into the selected mode.

        Returns:
            list[RuntimePersistenceCase]: Parsed cases for this mode.
        """

        cases = super().load_cases()
        return [case for case in cases if self.mode in case.modes]

    def parse_case(self, raw_case: Any) -> RuntimePersistenceCase:
        """Parse one runtime persistence case.

        Args:
            raw_case (Any): Raw JSON case object.

        Returns:
            RuntimePersistenceCase: Parsed trajectory case.
        """

        return _parse_case(raw_case)

    def case_id(self, case: RuntimePersistenceCase, index: int) -> str:
        """Return the stable dataset id.

        Args:
            case (RuntimePersistenceCase): Parsed case.
            index (int): Zero-based case index.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: RuntimePersistenceCase) -> EvalResult:
        """Run and grade one runtime trajectory.

        Args:
            case (RuntimePersistenceCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        artifact = await self._run_trajectory(case)
        hard_failures = _grade_case(case, artifact, mode=self.mode)
        failures = list(hard_failures)
        score = 1.0 if not failures else 0.0
        judge_details: dict[str, Any] | None = None

        if self.judge_mode == "live":
            judge_outcome = await _judge_trajectory(
                case,
                artifact,
                hard_failures=hard_failures,
                min_score=self._min_score_for_case(case),
            )
            judge_details = judge_outcome.to_dict()
            failures = judge_outcome.failures
            score = judge_outcome.score

        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=score,
            details={
                "description": case.description,
                "mode": self.mode,
                "backend": self.backend,
                "judge_mode": self.judge_mode,
                "failures": failures,
                "judge": judge_details,
                "artifact": artifact,
            },
        )

    async def _run_trajectory(self, case: RuntimePersistenceCase) -> dict[str, Any]:
        from agent.memory.hashing import hash_session_id
        from agent.persistence import PersistentAgentRuntime

        run_id = f"eval-{case.id}-{uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory(prefix="opencouch-runtime-eval-") as tmp:
            paths = RuntimePaths(
                thread=Path(tmp) / "threads.sqlite3",
                memory=Path(tmp) / "memory.sqlite3",
                crisis=Path(tmp) / "crisis.sqlite3",
                feedback=Path(tmp) / "feedback.sqlite3",
            )

            runtime: PersistentAgentRuntime | None = None
            steps: list[dict[str, Any]] = []

            async def open_runtime() -> PersistentAgentRuntime:
                new_runtime = self._build_runtime(case, paths=paths)
                await new_runtime.__aenter__()
                await _seed_case_memory(new_runtime, case=case, run_id=run_id)
                return new_runtime

            async def reopen_runtime() -> PersistentAgentRuntime:
                nonlocal runtime
                if runtime is not None:
                    await runtime.__aexit__(None, None, None)
                runtime = await open_runtime()
                return runtime

            async def fake_factual_lookup(
                state: dict[str, Any],
                *,
                llm_client: Any,  # noqa: ARG001 - tool protocol
                query: str,
            ) -> tuple[str, str]:
                return factual_lookup_fixture_answer(query)

            async def fake_crisis_resources(
                state: dict[str, Any],
                *,
                llm_client: Any,  # noqa: ARG001 - tool protocol
            ) -> tuple[str, list[dict[str, str]], str]:
                text = _state_text(state).casefold()
                if "singapore" not in text:
                    return "", [], "no_location"
                return (
                    "Singapore",
                    [
                        {
                            "name": "Samaritans of Singapore",
                            "phone": "1767",
                            "website": "https://www.sos.org.sg",
                            "region": "Singapore",
                        }
                    ],
                    "found",
                )

            with (
                patch(
                    "agent.turn_branches.answer_factual_lookup",
                    new=fake_factual_lookup,
                ),
                patch(
                    "agent.nodes.crisis_resource_lookup.find_crisis_resources",
                    new=fake_crisis_resources,
                ),
            ):
                try:
                    runtime = await open_runtime()
                    for index, step in enumerate(case.steps):
                        if step.type == "reopen_runtime":
                            runtime = await reopen_runtime()
                            steps.append(
                                {
                                    "step_index": index + 1,
                                    "type": step.type,
                                    "reopened": True,
                                }
                            )
                            continue

                        steps.append(
                            await self._run_step(
                                runtime,
                                step=step,
                                step_index=index + 1,
                                run_id=run_id,
                            )
                        )

                    assert runtime is not None
                    final = await _collect_final_observations(
                        runtime,
                        case=case,
                        run_id=run_id,
                    )
                finally:
                    if runtime is not None:
                        await runtime.__aexit__(None, None, None)

        return {
            "case_id": case.id,
            "run_id": run_id,
            "backend": self.backend,
            "mode": self.mode,
            "description": case.description,
            "steps": steps,
            "final": final,
            "hashes": {
                "feedback_sessions": [
                    hash_session_id(_format_text(item.get("thread_id", ""), run_id))
                    for item in _mapping_list(
                        case.expected.get("feedback", []),
                        "expected.feedback",
                    )
                ]
            },
        }

    def _build_runtime(
        self,
        case: RuntimePersistenceCase,
        *,
        paths: RuntimePaths,
    ) -> Any:
        from agent.memory.embeddings import NullEmbeddingProvider
        from agent.memory.modes import MemoryMode
        from agent.persistence import PersistentAgentRuntime

        runtime_config = case.runtime
        memory_mode = MemoryMode(str(runtime_config.get("memory_mode", "local")))
        database_url = self.database_url if self.backend == "postgres" else None

        return PersistentAgentRuntime(
            sqlite_path=paths.thread,
            memory_sqlite_path=paths.memory,
            crisis_log_sqlite_path=paths.crisis,
            feedback_sqlite_path=paths.feedback,
            memory_mode=memory_mode,
            memory_backend=self.backend,  # type: ignore[arg-type]
            memory_database_url=database_url,
            thread_persistence_backend=self.backend,  # type: ignore[arg-type]
            thread_database_url=database_url,
            crisis_log_persistence_backend=self.backend,  # type: ignore[arg-type]
            crisis_log_database_url=database_url,
            session_feedback_persistence_backend=self.backend,  # type: ignore[arg-type]
            session_feedback_database_url=database_url,
            embedding_provider=NullEmbeddingProvider(),
            extract_in_foreground=bool(
                runtime_config.get("extract_in_foreground", False)
            ),
            finalize_active_sessions_on_close=bool(
                runtime_config.get("finalize_active_sessions_on_close", False)
            ),
            speculative_memory_prefetch=bool(
                runtime_config.get("speculative_memory_prefetch", False)
            ),
            session_sweep_interval_seconds=3600.0,
        )

    async def _run_step(
        self,
        runtime: Any,
        *,
        step: RuntimePersistenceStep,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        if step.type == "turn":
            return await self._run_turn(
                runtime, step, step_index=step_index, run_id=run_id
            )
        if step.type == "stream_turn":
            return await self._run_stream_turn(
                runtime,
                step,
                step_index=step_index,
                run_id=run_id,
            )
        if step.type == "end_session":
            return await self._end_session(
                runtime,
                step,
                step_index=step_index,
                run_id=run_id,
            )
        if step.type == "record_feedback":
            return await self._record_feedback(
                runtime,
                step,
                step_index=step_index,
                run_id=run_id,
            )
        if step.type == "expect_status":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            return {
                "step_index": step_index,
                "type": step.type,
                "thread_id": thread_id,
                "status_after": (await runtime.session_status(thread_id)).value,
                "expected": _format_templates(step.expected, run_id),
            }
        raise ValueError(f"Unsupported runtime persistence step type {step.type!r}.")

    async def _run_turn(
        self,
        runtime: Any,
        step: RuntimePersistenceStep,
        *,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
        user_id = _format_optional(step.user_id, run_id)
        message = _required_formatted(step.message, "message", run_id)
        control_llm, response_llm = self._llms_for_step(step, run_id=run_id)
        before_crisis = await _crisis_records_for_thread(runtime, thread_id)

        result = await runtime.run_turn(
            thread_id=thread_id,
            message=message,
            user_id=user_id,
            llm_client=control_llm,
            response_llm_client=response_llm,
        )
        after_crisis = await _crisis_records_for_thread(runtime, thread_id)
        status_after = await runtime.session_status(thread_id)

        return _turn_artifact(
            step=step,
            step_index=step_index,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            output=result.output,
            state=result.state,
            status_after=status_after.value,
            crisis_log_delta_count=len(after_crisis) - len(before_crisis),
            structured_calls=getattr(control_llm, "structured_calls", {}),
        )

    async def _run_stream_turn(
        self,
        runtime: Any,
        step: RuntimePersistenceStep,
        *,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        from agent.models import ChunkEvent, DoneEvent, ResponseReadyEvent, StatusEvent

        thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
        user_id = _format_optional(step.user_id, run_id)
        message = _required_formatted(step.message, "message", run_id)
        control_llm, response_llm = self._llms_for_step(step, run_id=run_id)
        before_crisis = await _crisis_records_for_thread(runtime, thread_id)

        statuses: list[str] = []
        chunks: list[str] = []
        ready_count = 0
        done_count = 0
        done_output: Any | None = None
        async for event in runtime.run_turn_stream(
            thread_id=thread_id,
            message=message,
            user_id=user_id,
            llm_client=control_llm,
            response_llm_client=response_llm,
        ):
            if isinstance(event, StatusEvent):
                statuses.append(event.stage)
            elif isinstance(event, ChunkEvent):
                chunks.append(event.text)
            elif isinstance(event, ResponseReadyEvent):
                ready_count += 1
            elif isinstance(event, DoneEvent):
                done_count += 1
                done_output = event.output

        final_state = await runtime.get_state(thread_id)
        if final_state is None:
            raise RuntimeError(f"stream turn produced no state for {thread_id!r}.")
        after_crisis = await _crisis_records_for_thread(runtime, thread_id)
        status_after = await runtime.session_status(thread_id)

        artifact = _turn_artifact(
            step=step,
            step_index=step_index,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            output=done_output,
            state=final_state,
            status_after=status_after.value,
            crisis_log_delta_count=len(after_crisis) - len(before_crisis),
            structured_calls=getattr(control_llm, "structured_calls", {}),
        )
        artifact["stream"] = {
            "stages": statuses,
            "chunk_text": "".join(chunks),
            "response_ready_count": ready_count,
            "done_count": done_count,
        }
        return artifact

    async def _end_session(
        self,
        runtime: Any,
        step: RuntimePersistenceStep,
        *,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
        control_llm, _response_llm = self._llms_for_step(step, run_id=run_id)
        stored_arc = await runtime.end_session(thread_id, llm_client=control_llm)
        state = await runtime.get_state(thread_id)
        status_after = await runtime.session_status(thread_id)
        return {
            "step_index": step_index,
            "type": step.type,
            "thread_id": thread_id,
            "stored_arc": _jsonify(stored_arc),
            "stored_arc_exists": stored_arc is not None,
            "state_after": _state_summary(state),
            "status_after": status_after.value,
            "expected": _format_templates(step.expected, run_id),
            "structured_calls": getattr(control_llm, "structured_calls", {}),
        }

    async def _record_feedback(
        self,
        runtime: Any,
        step: RuntimePersistenceStep,
        *,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        from agent.memory.hashing import hash_session_id

        thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
        label = _required_formatted(step.label, "label", run_id)
        source = _required_formatted(step.source, "source", run_id)
        record = await runtime.record_session_feedback(
            thread_id,
            label=label,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
        )
        records = await runtime.session_feedback_backend.alist_by_session(
            hash_session_id(thread_id)
        )
        return {
            "step_index": step_index,
            "type": step.type,
            "thread_id": thread_id,
            "record": _jsonify(record),
            "feedback_records": _jsonify(records),
            "feedback_count": len(records),
            "expected": _format_templates(step.expected, run_id),
        }

    def _llms_for_step(
        self,
        step: RuntimePersistenceStep,
        *,
        run_id: str,
    ) -> tuple[Any, Any]:
        if self.mode == "scripted":
            llm = ScriptedRuntimeLLM(_format_templates(step.scripted, run_id))
            return llm, llm

        if self._live_llms is None:
            self._live_llms = build_live_therapeutic_llms()
        return self._live_llms

    def _min_score_for_case(self, case: RuntimePersistenceCase) -> float:
        if self.min_judge_score is not None:
            return self.min_judge_score
        expected_score = case.rubric.get("min_judge_score")
        if expected_score is not None:
            return float(expected_score)
        return _DEFAULT_MIN_JUDGE_SCORE


def _parse_case(raw_case: Any) -> RuntimePersistenceCase:
    if not isinstance(raw_case, Mapping):
        raise TypeError("Runtime persistence eval cases must be JSON objects.")
    return RuntimePersistenceCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        modes=[str(mode) for mode in raw_case.get("modes", ["scripted"])],
        runtime=dict(_optional_mapping(raw_case, "runtime")),
        memory_seed=dict(_optional_mapping(raw_case, "memory_seed")),
        steps=[
            _parse_step(item)
            for item in _mapping_list(raw_case.get("steps", []), "steps")
        ],
        expected=dict(_optional_mapping(raw_case, "expected")),
        rubric=dict(_optional_mapping(raw_case, "rubric")),
    )


def _parse_step(raw_step: Mapping[str, Any]) -> RuntimePersistenceStep:
    return RuntimePersistenceStep(
        type=str(raw_step["type"]),
        thread_id=_optional_str(raw_step.get("thread_id")),
        user_id=_optional_str(raw_step.get("user_id")),
        message=_optional_str(raw_step.get("message")),
        label=_optional_str(raw_step.get("label")),
        source=_optional_str(raw_step.get("source")),
        scripted=dict(_optional_mapping(raw_step, "scripted")),
        expected=dict(_optional_mapping(raw_step, "expected")),
    )


async def _seed_case_memory(
    runtime: Any,
    *,
    case: RuntimePersistenceCase,
    run_id: str,
) -> None:
    seed = _format_templates(case.memory_seed, run_id)
    owners = seed.get("owners")
    if not isinstance(owners, list):
        return
    for owner_seed in owners:
        if not isinstance(owner_seed, Mapping):
            continue
        owner_id = str(owner_seed.get("owner_id") or "")
        if not owner_id:
            continue
        payload = owner_seed.get("seed")
        if isinstance(payload, Mapping):
            await seed_memory_store(
                runtime.memory_store, owner_id=owner_id, seed=payload
            )


async def _collect_final_observations(
    runtime: Any,
    *,
    case: RuntimePersistenceCase,
    run_id: str,
) -> dict[str, Any]:
    expected = _format_templates(case.expected, run_id)
    memory_snapshots: list[dict[str, Any]] = []
    for spec in _mapping_list(expected.get("memory_snapshots", []), "memory_snapshots"):
        owner_id = str(spec.get("owner_id") or "")
        if owner_id:
            memory_snapshots.append(
                {
                    "owner_id": owner_id,
                    "snapshot": await memory_snapshot(
                        runtime.memory_store,
                        owner_id=owner_id,
                    ),
                }
            )

    thread_states: list[dict[str, Any]] = []
    for spec in _mapping_list(expected.get("thread_states", []), "thread_states"):
        thread_id = str(spec.get("thread_id") or "")
        if not thread_id:
            continue
        state = await runtime.get_state(thread_id)
        status = await runtime.session_status(thread_id)
        thread_states.append(
            {
                "thread_id": thread_id,
                "exists": state is not None,
                "status": status.value,
                "state": _state_summary(state),
            }
        )

    crisis_logs: list[dict[str, Any]] = []
    for spec in _mapping_list(expected.get("crisis_logs", []), "crisis_logs"):
        thread_id = str(spec.get("thread_id") or "")
        if not thread_id:
            continue
        crisis_logs.append(
            {
                "thread_id": thread_id,
                "records": _jsonify(
                    await _crisis_records_for_thread(runtime, thread_id)
                ),
            }
        )

    feedback: list[dict[str, Any]] = []
    for spec in _mapping_list(expected.get("feedback", []), "feedback"):
        from agent.memory.hashing import hash_session_id

        thread_id = str(spec.get("thread_id") or "")
        if not thread_id:
            continue
        records = await runtime.session_feedback_backend.alist_by_session(
            hash_session_id(thread_id)
        )
        feedback.append(
            {
                "thread_id": thread_id,
                "records": _jsonify(records),
            }
        )

    return {
        "memory_snapshots": memory_snapshots,
        "thread_states": thread_states,
        "crisis_logs": crisis_logs,
        "feedback": feedback,
        "expected": expected,
    }


async def _crisis_records_for_thread(runtime: Any, thread_id: str) -> list[Any]:
    from agent.memory.hashing import hash_session_id

    session_hash = hash_session_id(thread_id)
    records: list[Any] = []
    today = datetime.now(UTC).date()
    for day in (today - timedelta(days=1), today, today + timedelta(days=1)):
        for record in await runtime.crisis_log_backend.alist_by_date(day):
            if getattr(record, "session_id_opaque", None) == session_hash:
                records.append(record)
    return records


def _turn_artifact(
    *,
    step: RuntimePersistenceStep,
    step_index: int,
    run_id: str,
    thread_id: str,
    user_id: str | None,
    message: str,
    output: Any,
    state: Mapping[str, Any],
    status_after: str,
    crisis_log_delta_count: int,
    structured_calls: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "type": step.type,
        "thread_id": thread_id,
        "user_id": user_id,
        "message": message,
        "output": _jsonify(output),
        "state_after": _state_summary(state),
        "route": state.get("route"),
        "response_text": getattr(output, "response_text", None)
        if output is not None
        else state.get("response_text"),
        "response_style": getattr(output, "response_style", None)
        if output is not None
        else state.get("response_style"),
        "status_after": status_after,
        "crisis_log_delta_count": crisis_log_delta_count,
        "structured_calls": dict(structured_calls),
        "expected": _format_templates(step.expected, run_id),
    }


def _state_summary(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}

    transcript = state.get("transcript") or []
    assistant_turns = [
        item
        for item in transcript
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    session_progress = state.get("session_progress") or {}
    return {
        "route": state.get("route"),
        "response_style": state.get("response_style"),
        "therapeutic_approach": state.get("therapeutic_approach"),
        "transcript": _jsonify(transcript),
        "transcript_length": len(transcript) if isinstance(transcript, list) else None,
        "assistant_turn_count": len(assistant_turns),
        "turn_count": session_progress.get("turn_count")
        if isinstance(session_progress, Mapping)
        else None,
        "working_memory": _jsonify(state.get("working_memory") or []),
        "working_memory_count": len(state.get("working_memory") or []),
        "exercise_state": _jsonify(state.get("exercise_state") or {}),
        "memory_control": _jsonify(state.get("memory_control") or {}),
        "grounded_lookup": _jsonify(state.get("grounded_lookup") or {}),
        "turn_lifecycle": _jsonify(state.get("turn_lifecycle") or {}),
        "crisis": _jsonify(state.get("crisis")),
        "diagnostics": _jsonify(state.get("diagnostics") or {}),
    }


def _grade_case(
    case: RuntimePersistenceCase,
    artifact: Mapping[str, Any],
    *,
    mode: str,
) -> list[str]:
    failures: list[str] = []
    steps = artifact.get("steps")
    if not isinstance(steps, list):
        return ["artifact.steps is not a list"]

    expected_steps = [step for step in case.steps if step.type != "reopen_runtime"]
    observed_steps = [
        step for step in steps if isinstance(step, Mapping) and step.get("expected")
    ]
    for index, step_case in enumerate(expected_steps):
        if index >= len(observed_steps):
            failures.append(f"step {index + 1}: missing artifact")
            continue
        _grade_step(
            failures,
            step_case=step_case,
            artifact=observed_steps[index],
            mode=mode,
        )

    final = artifact.get("final")
    if isinstance(final, Mapping):
        _grade_final(failures, final=final, mode=mode)
    else:
        failures.append("artifact.final is not a mapping")
    return failures


def _grade_step(
    failures: list[str],
    *,
    step_case: RuntimePersistenceStep,
    artifact: Mapping[str, Any],
    mode: str,
) -> None:
    expected = artifact.get("expected")
    if not isinstance(expected, Mapping):
        return

    label = f"step {artifact.get('step_index')}"
    if mode == "scripted":
        _expect_equal(failures, label, "route", artifact.get("route"), expected)
        _expect_equal(
            failures,
            label,
            "response_style",
            artifact.get("response_style"),
            expected,
        )
        _expect_equal(
            failures,
            label,
            "crisis_log_delta_count",
            artifact.get("crisis_log_delta_count"),
            expected,
        )

    _expect_equal(
        failures, label, "status_after", artifact.get("status_after"), expected
    )
    _grade_text_collection(
        failures,
        label=f"{label}.response_text",
        values=artifact.get("response_text"),
        contains=expected.get("response_text_contains"),
        absent=expected.get("response_text_not_contains"),
    )
    _expect_equal(
        failures,
        label,
        "feedback_count",
        artifact.get("feedback_count"),
        expected,
    )
    _expect_equal(
        failures,
        label,
        "stored_arc_exists",
        artifact.get("stored_arc_exists"),
        expected,
    )
    _grade_stored_arc(
        failures,
        label=f"{label}.stored_arc",
        actual=artifact.get("stored_arc"),
        expected=expected.get("stored_arc"),
    )

    state = artifact.get("state_after")
    if isinstance(state, Mapping):
        _expect_equal(
            failures,
            label,
            "transcript_length",
            state.get("transcript_length"),
            expected,
        )
        _expect_equal(
            failures,
            label,
            "assistant_turn_count",
            state.get("assistant_turn_count"),
            expected,
        )
        _expect_equal(failures, label, "turn_count", state.get("turn_count"), expected)
        _expect_equal(
            failures,
            label,
            "working_memory_count",
            state.get("working_memory_count"),
            expected,
        )
        _grade_minimum(
            failures,
            label=f"{label}.working_memory_count",
            actual=state.get("working_memory_count"),
            expected=expected.get("working_memory_count_min"),
        )
        _grade_mapping_expectation(
            failures,
            label=f"{label}.exercise_state",
            actual=state.get("exercise_state"),
            expected=expected.get("exercise_state"),
        )
        _grade_mapping_expectation(
            failures,
            label=f"{label}.memory_control",
            actual=state.get("memory_control"),
            expected=expected.get("memory_control"),
        )
        _grade_mapping_expectation(
            failures,
            label=f"{label}.grounded_lookup",
            actual=state.get("grounded_lookup"),
            expected=expected.get("grounded_lookup"),
        )
        _grade_mapping_expectation(
            failures,
            label=f"{label}.turn_lifecycle",
            actual=state.get("turn_lifecycle"),
            expected=expected.get("turn_lifecycle"),
        )
        _grade_text_collection(
            failures,
            label=f"{label}.working_memory",
            values=state.get("working_memory"),
            contains=expected.get("working_memory_contains"),
            absent=expected.get("working_memory_not_contains"),
        )
        diagnostics = state.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            for key in (
                "retrieval_path",
                "semantic_hits",
                "episodic_hits",
                "procedural_count",
                "proactive_recall",
            ):
                _expect_equal(failures, label, key, diagnostics.get(key), expected)
            _grade_minimum(
                failures,
                label=f"{label}.semantic_hits",
                actual=diagnostics.get("semantic_hits"),
                expected=expected.get("semantic_hits_min"),
            )
            _grade_minimum(
                failures,
                label=f"{label}.episodic_hits",
                actual=diagnostics.get("episodic_hits"),
                expected=expected.get("episodic_hits_min"),
            )

    stream = artifact.get("stream")
    if isinstance(stream, Mapping):
        _expect_equal(failures, label, "stream_stages", stream.get("stages"), expected)
        _expect_equal(
            failures,
            label,
            "response_ready_count",
            stream.get("response_ready_count"),
            expected,
        )
        _expect_equal(failures, label, "done_count", stream.get("done_count"), expected)
        if mode == "scripted" and expected.get("chunk_text_contains"):
            needle = str(expected["chunk_text_contains"])
            if needle not in str(stream.get("chunk_text", "")):
                failures.append(f"{label}.stream.chunk_text missing {needle!r}")

    if step_case.type == "record_feedback":
        records = artifact.get("feedback_records")
        _grade_records(
            failures,
            label=f"{label}.feedback_records",
            records=records,
            expected=expected,
        )


def _grade_final(
    failures: list[str],
    *,
    final: Mapping[str, Any],
    mode: str,
) -> None:
    expected = final.get("expected")
    if not isinstance(expected, Mapping):
        return

    _grade_final_memory(failures, final=final, expected=expected, mode=mode)
    _grade_final_thread_states(failures, final=final, expected=expected, mode=mode)
    _grade_final_crisis_logs(failures, final=final, expected=expected, mode=mode)
    _grade_final_feedback(failures, final=final, expected=expected, mode=mode)


def _grade_final_memory(
    failures: list[str],
    *,
    final: Mapping[str, Any],
    expected: Mapping[str, Any],
    mode: str,  # noqa: ARG001 - memory snapshots are deterministic in all modes
) -> None:
    observations = {
        item.get("owner_id"): item.get("snapshot")
        for item in _mapping_list(final.get("memory_snapshots", []), "memory_snapshots")
    }
    for spec in _mapping_list(expected.get("memory_snapshots", []), "memory_snapshots"):
        owner_id = spec.get("owner_id")
        snapshot = observations.get(owner_id)
        if not isinstance(snapshot, Mapping):
            failures.append(f"final.memory[{owner_id!r}] missing snapshot")
            continue
        grade_store_expectations(
            failures,
            snapshot=snapshot,
            expected={"store": spec},
            prefix="store",
        )


def _grade_final_thread_states(
    failures: list[str],
    *,
    final: Mapping[str, Any],
    expected: Mapping[str, Any],
    mode: str,  # noqa: ARG001 - future mode-specific checks
) -> None:
    observations = {
        item.get("thread_id"): item
        for item in _mapping_list(final.get("thread_states", []), "thread_states")
    }
    for spec in _mapping_list(expected.get("thread_states", []), "thread_states"):
        thread_id = spec.get("thread_id")
        observed = observations.get(thread_id)
        if not isinstance(observed, Mapping):
            failures.append(f"final.thread[{thread_id!r}] missing state observation")
            continue
        state = (
            observed.get("state") if isinstance(observed.get("state"), Mapping) else {}
        )
        for key in ("exists", "status"):
            if key in spec and observed.get(key) != spec[key]:
                failures.append(
                    f"final.thread[{thread_id!r}].{key}: expected "
                    f"{spec[key]!r}, got {observed.get(key)!r}"
                )
        for key in ("transcript_length", "assistant_turn_count", "turn_count"):
            if key in spec and state.get(key) != spec[key]:
                failures.append(
                    f"final.thread[{thread_id!r}].{key}: expected "
                    f"{spec[key]!r}, got {state.get(key)!r}"
                )
        if (
            "therapeutic_approach" in spec
            and state.get("therapeutic_approach") != spec["therapeutic_approach"]
        ):
            failures.append(
                f"final.thread[{thread_id!r}].therapeutic_approach: expected "
                f"{spec['therapeutic_approach']!r}, got "
                f"{state.get('therapeutic_approach')!r}"
            )
        _grade_mapping_expectation(
            failures,
            label=f"final.thread[{thread_id!r}].exercise_state",
            actual=state.get("exercise_state"),
            expected=spec.get("exercise_state"),
        )


def _grade_final_crisis_logs(
    failures: list[str],
    *,
    final: Mapping[str, Any],
    expected: Mapping[str, Any],
    mode: str,
) -> None:
    if mode != "scripted":
        return
    observations = {
        item.get("thread_id"): item.get("records")
        for item in _mapping_list(final.get("crisis_logs", []), "crisis_logs")
    }
    for spec in _mapping_list(expected.get("crisis_logs", []), "crisis_logs"):
        thread_id = spec.get("thread_id")
        records = observations.get(thread_id)
        if not isinstance(records, list):
            failures.append(f"final.crisis_logs[{thread_id!r}] missing records")
            continue
        if "count" in spec and len(records) != spec["count"]:
            failures.append(
                f"final.crisis_logs[{thread_id!r}].count: expected "
                f"{spec['count']!r}, got {len(records)!r}"
            )
        if not records:
            continue
        _grade_records(
            failures,
            label=f"final.crisis_logs[{thread_id!r}]",
            records=[records[-1]],
            expected=spec,
        )


def _grade_final_feedback(
    failures: list[str],
    *,
    final: Mapping[str, Any],
    expected: Mapping[str, Any],
    mode: str,
) -> None:
    if mode != "scripted":
        return
    observations = {
        item.get("thread_id"): item.get("records")
        for item in _mapping_list(final.get("feedback", []), "feedback")
    }
    for spec in _mapping_list(expected.get("feedback", []), "feedback"):
        thread_id = spec.get("thread_id")
        records = observations.get(thread_id)
        if not isinstance(records, list):
            failures.append(f"final.feedback[{thread_id!r}] missing records")
            continue
        if "count" in spec and len(records) != spec["count"]:
            failures.append(
                f"final.feedback[{thread_id!r}].count: expected "
                f"{spec['count']!r}, got {len(records)!r}"
            )
        _grade_records(
            failures,
            label=f"final.feedback[{thread_id!r}]",
            records=records[-1:] if records else [],
            expected=spec,
        )


def _grade_records(
    failures: list[str],
    *,
    label: str,
    records: Any,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(records, list):
        failures.append(f"{label} is not a list")
        return
    if not records:
        if any(
            key in expected
            for key in (
                "level",
                "label",
                "user_id_or_null",
                "response_node_completed",
                "llm_failure_occurred",
                "turn_count_at_end",
            )
        ):
            failures.append(f"{label} is empty")
        return

    record = records[-1]
    if not isinstance(record, Mapping):
        failures.append(f"{label} last record is not a mapping")
        return
    for key in (
        "level",
        "label",
        "user_id_or_null",
        "response_node_completed",
        "llm_failure_occurred",
        "turn_count_at_end",
    ):
        if key in expected and record.get(key) != expected[key]:
            failures.append(
                f"{label}.{key}: expected {expected[key]!r}, got {record.get(key)!r}"
            )


def _grade_mapping_expectation(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not isinstance(expected, Mapping):
        return
    actual_map = actual if isinstance(actual, Mapping) else {}
    for key, expected_value in expected.items():
        actual_value = actual_map.get(key)
        if isinstance(expected_value, Mapping):
            _grade_mapping_expectation(
                failures,
                label=f"{label}.{key}",
                actual=actual_value,
                expected=expected_value,
            )
        elif actual_value != expected_value:
            failures.append(
                f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )


def _grade_stored_arc(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not isinstance(expected, Mapping):
        return
    if not isinstance(actual, Mapping):
        failures.append(f"{label}: expected stored arc, got {actual!r}")
        return

    for key in (
        "session_id",
        "turn_count",
        "approach_used",
        "crisis_level_max",
        "primary_themes",
        "open_loops",
        "resolved_threads",
    ):
        _expect_equal(failures, label, key, actual.get(key), expected)

    _grade_text_collection(
        failures,
        label=f"{label}.summary",
        values=actual.get("summary"),
        contains=expected.get("summary_contains"),
        absent=expected.get("summary_not_contains"),
    )
    for key in ("primary_themes", "open_loops", "resolved_threads"):
        _grade_text_collection(
            failures,
            label=f"{label}.{key}",
            values=actual.get(key),
            contains=expected.get(f"{key}_contains"),
            absent=expected.get(f"{key}_not_contains"),
        )

    _grade_mapping_expectation(
        failures,
        label=f"{label}.mood_arc",
        actual=actual.get("mood_arc"),
        expected=expected.get("mood_arc"),
    )
    _grade_text_collection(
        failures,
        label=f"{label}.mood_arc",
        values=actual.get("mood_arc"),
        contains=expected.get("mood_arc_contains"),
        absent=expected.get("mood_arc_not_contains"),
    )
    _grade_mapping_expectation(
        failures,
        label=f"{label}.approach_context",
        actual=actual.get("approach_context"),
        expected=expected.get("approach_context"),
    )
    _grade_text_collection(
        failures,
        label=f"{label}.approach_context",
        values=actual.get("approach_context"),
        contains=expected.get("approach_context_contains"),
        absent=expected.get("approach_context_not_contains"),
    )


def _grade_text_collection(
    failures: list[str],
    *,
    label: str,
    values: Any,
    contains: Any,
    absent: Any,
) -> None:
    haystack = " ".join(_flatten_text(values)).casefold()
    for phrase in _as_list(contains):
        if str(phrase).casefold() not in haystack:
            failures.append(f"{label} missing {phrase!r}")
    for phrase in _as_list(absent):
        if str(phrase).casefold() in haystack:
            failures.append(f"{label} contains forbidden {phrase!r}")


def _grade_minimum(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if expected is None:
        return
    if not isinstance(actual, int | float) or actual < expected:
        failures.append(f"{label}: expected >= {expected!r}, got {actual!r}")


async def _judge_trajectory(
    case: RuntimePersistenceCase,
    artifact: Mapping[str, Any],
    *,
    hard_failures: list[str],
    min_score: float,
) -> Any:
    from config import create_configured_control_llm_client

    judge = RubricLLMJudge(llm_client=create_configured_control_llm_client())
    verdict = await judge.judge(
        RubricJudgeArtifact(
            task=(
                "Judge a runtime persistence trajectory for a mental-health "
                "support agent. Hard checks validate durable state, memory, "
                "logs, feedback, streaming, and active-session liveness."
            ),
            input={
                "case_id": case.id,
                "description": case.description,
                "hard_check_note": (
                    "Hard checks are authoritative for persistence invariants. "
                    "Judge whether the observed runtime behavior is coherent "
                    "and operationally trustworthy. Do not require exact "
                    "scripted wording or response style; for reopen cases, "
                    "continuity can be shown by persisted transcript state and "
                    "a response that reasonably follows the prior user turn."
                ),
            },
            output=_judge_output(artifact, hard_failures=hard_failures),
            rubric=_rubric_dimensions(case),
            hard_failures=hard_failures,
        )
    )
    return judge.combine(
        verdict=verdict,
        hard_failures=hard_failures,
        min_score=min_score,
    )


def _judge_output(
    artifact: Mapping[str, Any],
    *,
    hard_failures: list[str],
) -> dict[str, Any]:
    return {
        "case_id": artifact.get("case_id"),
        "backend": artifact.get("backend"),
        "mode": artifact.get("mode"),
        "description": artifact.get("description"),
        "hard_checks": {"passed": not hard_failures, "failures": hard_failures},
        "steps": _compact_judge_steps(artifact.get("steps")),
        "final": _compact_judge_final(artifact.get("final")),
    }


def _compact_judge_steps(steps: Any) -> list[dict[str, Any]]:
    """Return judge-facing step summaries without noisy runtime diagnostics.

    Args:
        steps (Any): Raw artifact steps.

    Returns:
        list[dict[str, Any]]: Compact summaries for LLM judging.
    """

    if not isinstance(steps, list):
        return []

    compact: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        state = _mapping_or_empty(raw_step.get("state_after"))
        output = _mapping_or_empty(raw_step.get("output"))
        stream = _mapping_or_empty(raw_step.get("stream"))
        entry: dict[str, Any] = {
            "step_index": raw_step.get("step_index"),
            "type": raw_step.get("type"),
            "thread_id": raw_step.get("thread_id"),
            "message": raw_step.get("message"),
            "status_after": raw_step.get("status_after"),
            "route": raw_step.get("route"),
            "response_style": raw_step.get("response_style"),
            "expected": raw_step.get("expected"),
        }
        if output:
            entry["response_text"] = output.get("response_text")
        if stream:
            entry["stream"] = {
                "stages": stream.get("stages"),
                "response_ready_count": stream.get("response_ready_count"),
                "done_count": stream.get("done_count"),
            }
        if state:
            entry["state_after"] = _compact_judge_state(state)
        if "stored_arc" in raw_step:
            entry["stored_arc"] = raw_step.get("stored_arc")
        if "reopened" in raw_step:
            entry["reopened"] = raw_step.get("reopened")
        compact.append(
            {key: value for key, value in entry.items() if value is not None}
        )
    return compact


def _compact_judge_final(final: Any) -> dict[str, Any]:
    """Return a compact final-state summary for LLM judging.

    Args:
        final (Any): Raw final artifact payload.

    Returns:
        dict[str, Any]: Final persistence summary.
    """

    final_map = _mapping_or_empty(final)
    thread_states = final_map.get("thread_states")
    compact_threads: list[dict[str, Any]] = []
    if isinstance(thread_states, list):
        for raw_thread in thread_states:
            if not isinstance(raw_thread, Mapping):
                continue
            state = _mapping_or_empty(raw_thread.get("state"))
            compact_thread = {
                "thread_id": raw_thread.get("thread_id"),
                "exists": raw_thread.get("exists"),
                "status": raw_thread.get("status"),
            }
            if state:
                compact_thread["state"] = _compact_judge_state(state)
            compact_threads.append(
                {
                    key: value
                    for key, value in compact_thread.items()
                    if value is not None
                }
            )

    return {
        "memory_snapshots": final_map.get("memory_snapshots"),
        "thread_states": compact_threads,
        "crisis_logs": final_map.get("crisis_logs"),
        "feedback": final_map.get("feedback"),
        "expected": final_map.get("expected"),
    }


def _compact_judge_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the state fields that matter for persistence judging.

    Args:
        state (Mapping[str, Any]): Raw state summary.

    Returns:
        dict[str, Any]: Compact state summary.
    """

    return {
        "route": state.get("route"),
        "response_style": state.get("response_style"),
        "therapeutic_approach": state.get("therapeutic_approach"),
        "transcript_length": state.get("transcript_length"),
        "assistant_turn_count": state.get("assistant_turn_count"),
        "turn_count": state.get("turn_count"),
        "working_memory_count": state.get("working_memory_count"),
        "exercise_state": state.get("exercise_state"),
        "memory_control": state.get("memory_control"),
        "grounded_lookup": state.get("grounded_lookup"),
        "turn_lifecycle": state.get("turn_lifecycle"),
        "crisis": state.get("crisis"),
        "transcript_tail": _transcript_tail(state.get("transcript")),
    }


def _transcript_tail(transcript: Any, *, limit: int = 4) -> list[dict[str, Any]]:
    """Return a small transcript tail for continuity judging.

    Args:
        transcript (Any): Raw transcript payload.
        limit (int): Maximum number of tail entries.

    Returns:
        list[dict[str, Any]]: Compact transcript entries.
    """

    if not isinstance(transcript, list):
        return []
    tail: list[dict[str, Any]] = []
    for item in transcript[-limit:]:
        if not isinstance(item, Mapping):
            continue
        tail.append(
            {
                "role": item.get("role"),
                "content": item.get("content"),
                "response_style": item.get("response_style"),
            }
        )
    return tail


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return a mapping payload or an empty mapping.

    Args:
        value (Any): Candidate mapping.

    Returns:
        Mapping[str, Any]: Original mapping or empty mapping.
    """

    return value if isinstance(value, Mapping) else {}


def _rubric_dimensions(case: RuntimePersistenceCase) -> list[RubricDimension]:
    raw_dimensions = case.rubric.get("dimensions")
    if isinstance(raw_dimensions, list):
        return [RubricDimension.model_validate(item) for item in raw_dimensions]
    return [
        RubricDimension(
            name="runtime_durability",
            question=(
                "Does the trajectory show durable runtime behavior across turns, "
                "reopens, and session lifecycle operations?"
            ),
            weight=1.0,
        ),
        RubricDimension(
            name="state_hygiene",
            question=(
                "Does checkpoint, memory, log, and feedback state stay scoped "
                "to the intended thread/user without stale leakage?"
            ),
            weight=1.0,
        ),
    ]


def _expect_equal(
    failures: list[str],
    label: str,
    key: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    if key not in expected:
        return
    if actual != expected[key]:
        failures.append(f"{label}.{key}: expected {expected[key]!r}, got {actual!r}")


def _mapping_or_default(
    source: Mapping[str, Any],
    key: str,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = source.get(key)
    if isinstance(value, Mapping):
        return value
    return default


def _optional_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a JSON object.")
    return value


def _mapping_list(value: Any, field_name: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON list.")
    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be JSON objects.")
        items.append(item)
    return items


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _required_formatted(value: str | None, field_name: str, run_id: str) -> str:
    if value is None:
        raise ValueError(f"Runtime persistence step missing {field_name}.")
    return _format_text(value, run_id)


def _format_optional(value: str | None, run_id: str) -> str | None:
    if value is None:
        return None
    return _format_text(value, run_id)


def _format_text(value: str, run_id: str) -> str:
    return value.replace("{run_id}", run_id)


def _format_templates(value: Any, run_id: str) -> Any:
    if isinstance(value, str):
        return _format_text(value, run_id)
    if isinstance(value, list):
        return [_format_templates(item, run_id) for item in value]
    if isinstance(value, Mapping):
        return {key: _format_templates(item, run_id) for key, item in value.items()}
    return value


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return _jsonify(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        texts: list[str] = []
        for item in value.values():
            texts.extend(_flatten_text(item))
        return texts
    if isinstance(value, list | tuple):
        texts: list[str] = []
        for item in value:
            texts.extend(_flatten_text(item))
        return texts
    return [str(value)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _state_text(state: Mapping[str, Any]) -> str:
    pieces = [str(state.get("message", ""))]
    transcript = state.get("transcript") or []
    if isinstance(transcript, list):
        pieces.extend(
            str(item.get("content", ""))
            for item in transcript
            if isinstance(item, Mapping)
        )
    return " ".join(pieces)


def _resolve_database_url(backend: str, override: str | None) -> str | None:
    if backend == "sqlite":
        return None
    if override:
        return override

    from config import MISSING_MEMORY_DATABASE_URL_MESSAGE, get_settings

    database_url = os.getenv("OPENCOUCH_MEMORY_DATABASE_URL")
    if not database_url:
        database_url = get_settings().memory_database_url
    if not database_url:
        raise ValueError(MISSING_MEMORY_DATABASE_URL_MESSAGE)
    return database_url


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser(
        "Evaluate runtime and persistence trajectories against the real runtime."
    )
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="Use scripted LLM fixtures or configured live LLM clients.",
    )
    parser.add_argument(
        "--backend",
        choices=("postgres", "sqlite"),
        default="postgres",
        help="Persistence backend to evaluate. Postgres is the primary target.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres DSN override. Defaults to OPENCOUCH_MEMORY_DATABASE_URL.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("none", "live"),
        default="none",
        help="Optionally run an LLM judge over the trajectory artifact.",
    )
    parser.add_argument(
        "--min-judge-score",
        type=float,
        default=None,
        help="Minimum LLM judge score when --judge-mode live is enabled.",
    )
    return parser


def _build_evaluator(args: argparse.Namespace) -> RuntimePersistenceTrajectoryEvaluator:
    dataset = args.dataset or _DEFAULT_DATASET
    return RuntimePersistenceTrajectoryEvaluator(
        dataset_path=dataset,
        mode=args.mode,
        backend=args.backend,
        database_url=_resolve_database_url(args.backend, args.database_url),
        judge_mode=args.judge_mode,
        min_judge_score=args.min_judge_score,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    return run_evaluator_cli(_build_evaluator, parser=parser, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
