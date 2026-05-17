"""Evaluate runtime recovery, liveness, and concurrency trajectories."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.runtime_persistence_trajectory_eval import ScriptedRuntimeLLM

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "runtime" / "recovery_trajectory_v1.json"
)


@dataclass(frozen=True)
class RuntimeRecoveryStep:
    """One operation in a runtime recovery trajectory."""

    type: str
    thread_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    expected_liveness: str | None = None
    session_transcript_soft_limit: int | None = None
    scripted: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRecoveryCase:
    """Parsed runtime recovery trajectory case."""

    id: str
    description: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    steps: list[RuntimeRecoveryStep] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePaths:
    """Per-case SQLite fallback paths."""

    thread: Path
    memory: Path
    crisis: Path
    feedback: Path


class _FailingTextAdapter:
    async def run_turn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("scripted text-runtime failure")

    async def run_turn_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("scripted text-runtime failure")
        yield


class RuntimeRecoveryTrajectoryEvaluator(BaseEvaluator[RuntimeRecoveryCase]):
    """Run scripted runtime recovery trajectories."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        backend: str,
        database_url: str | None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"runtime_recovery_trajectory_{backend}",
        )
        self.backend = backend
        self.database_url = database_url

    def parse_case(self, raw_case: Any) -> RuntimeRecoveryCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Runtime recovery eval cases must be JSON objects.")
        return RuntimeRecoveryCase(
            id=str(raw_case["id"]),
            description=str(raw_case.get("description", "")),
            runtime=dict(_optional_mapping(raw_case, "runtime")),
            steps=[
                _parse_step(item)
                for item in _mapping_list(raw_case.get("steps", []), "steps")
            ],
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: RuntimeRecoveryCase, index: int) -> str:
        return case.id

    async def run_case(self, case: RuntimeRecoveryCase) -> EvalResult:
        artifact = await self._run_trajectory(case)
        failures = _grade_case(case, artifact)
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "backend": self.backend,
                "failures": failures,
                "artifact": artifact,
            },
        )

    async def _run_trajectory(self, case: RuntimeRecoveryCase) -> dict[str, Any]:
        from agent.persistence import PersistentAgentRuntime

        run_id = f"eval-{case.id}-{uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory(prefix="opencouch-runtime-recovery-") as tmp:
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
                return new_runtime

            async def reopen_runtime() -> PersistentAgentRuntime:
                nonlocal runtime
                if runtime is not None:
                    await runtime.__aexit__(None, None, None)
                runtime = await open_runtime()
                return runtime

            try:
                runtime = await open_runtime()
                for index, step in enumerate(case.steps):
                    if step.type == "reopen_runtime":
                        runtime = await reopen_runtime()
                        steps.append({"step_index": index + 1, "type": step.type})
                        continue
                    steps.append(
                        await self._run_step(
                            runtime,
                            step=step,
                            step_index=index + 1,
                            run_id=run_id,
                        )
                    )
                final = await _thread_observations(
                    runtime,
                    _mapping_list(case.expected.get("thread_states", []), "expected"),
                    run_id=run_id,
                )
            finally:
                if runtime is not None:
                    await runtime.__aexit__(None, None, None)

        return {
            "case_id": case.id,
            "run_id": run_id,
            "backend": self.backend,
            "steps": steps,
            "final": {
                "thread_states": final,
                "expected": _format_templates(case.expected, run_id),
            },
        }

    def _build_runtime(self, case: RuntimeRecoveryCase, *, paths: RuntimePaths) -> Any:
        from agent.memory.embeddings import NullEmbeddingProvider
        from agent.memory.modes import MemoryMode
        from agent.persistence import PersistentAgentRuntime

        runtime_config = case.runtime
        database_url = self.database_url if self.backend == "postgres" else None
        excluded_tokens = [
            str(item)
            for item in runtime_config.get("auto_finalize_excluded_contains", [])
        ]
        exclude = (
            (lambda thread_id: any(token in thread_id for token in excluded_tokens))
            if excluded_tokens
            else None
        )

        return PersistentAgentRuntime(
            sqlite_path=paths.thread,
            memory_sqlite_path=paths.memory,
            crisis_log_sqlite_path=paths.crisis,
            feedback_sqlite_path=paths.feedback,
            memory_mode=MemoryMode(str(runtime_config.get("memory_mode", "local"))),
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
            auto_finalize_excluded=exclude,
            speculative_memory_prefetch=bool(
                runtime_config.get("speculative_memory_prefetch", False)
            ),
            session_sweep_interval_seconds=3600.0,
        )

    async def _run_step(
        self,
        runtime: Any,
        *,
        step: RuntimeRecoveryStep,
        step_index: int,
        run_id: str,
    ) -> dict[str, Any]:
        if step.type == "turn":
            artifact = await _run_turn(runtime, _step_as_turn(step), run_id=run_id)
            return _step_artifact(
                step,
                step_index=step_index,
                artifact=artifact,
                run_id=run_id,
            )

        if step.type == "expect_turn_exception":
            artifact = await _run_turn(runtime, _step_as_turn(step), run_id=run_id)
            return _step_artifact(
                step,
                step_index=step_index,
                artifact=artifact,
                run_id=run_id,
            )

        if step.type == "failed_turn":
            previous_adapter = runtime._text_agent_adapter  # noqa: SLF001
            runtime._text_agent_adapter = _FailingTextAdapter()  # noqa: SLF001
            try:
                artifact = await _run_turn(runtime, _step_as_turn(step), run_id=run_id)
            finally:
                runtime._text_agent_adapter = previous_adapter  # noqa: SLF001
            return _step_artifact(
                step,
                step_index=step_index,
                artifact=artifact,
                run_id=run_id,
            )

        if step.type == "concurrent_turns":
            artifacts = await asyncio.gather(
                *(_run_turn(runtime, turn, run_id=run_id) for turn in step.turns)
            )
            thread_states = await _thread_observations(
                runtime,
                _mapping_list(step.expected.get("thread_states", []), "thread_states"),
                run_id=run_id,
            )
            return {
                "step_index": step_index,
                "type": step.type,
                "turns": artifacts,
                "result_count": sum(1 for item in artifacts if item.get("ok")),
                "exception_count": sum(
                    1 for item in artifacts if item.get("exception_type")
                ),
                "thread_states": thread_states,
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "inject_foreign_mutation":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            await runtime._active_session_manager.set_active_session_mutation(  # noqa: SLF001
                thread_id,
                mutation_token="foreign-runtime:eval:999999",
                mutation_kind="turn",
            )
            return {
                "step_index": step_index,
                "type": step.type,
                "thread_id": thread_id,
                "status_after": (await runtime.session_status(thread_id)).value,
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "end_session":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            llm = ScriptedRuntimeLLM(_format_templates(step.scripted, run_id))
            stored_arc = await runtime.end_session(thread_id, llm_client=llm)
            return {
                "step_index": step_index,
                "type": step.type,
                "thread_id": thread_id,
                "stored_arc_exists": stored_arc is not None,
                "status_after": (await runtime.session_status(thread_id)).value,
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "finalize_active_sessions":
            before = await runtime._list_active_thread_ids()  # noqa: SLF001
            llm = ScriptedRuntimeLLM(_format_templates(step.scripted, run_id))
            await runtime.finalize_active_sessions(llm_client=llm)
            after = await runtime._list_active_thread_ids()  # noqa: SLF001
            thread_states = await _thread_observations(
                runtime,
                _mapping_list(step.expected.get("thread_states", []), "thread_states"),
                run_id=run_id,
            )
            return {
                "step_index": step_index,
                "type": step.type,
                "active_before": before,
                "active_after": after,
                "thread_states": thread_states,
                "expected": _format_templates(step.expected, run_id),
            }

        raise ValueError(f"Unsupported runtime recovery step type {step.type!r}.")


async def _run_turn(
    runtime: Any,
    raw_turn: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    thread_id = _required_formatted(raw_turn.get("thread_id"), "thread_id", run_id)
    user_id = _format_optional(raw_turn.get("user_id"), run_id)
    message = _required_formatted(raw_turn.get("message"), "message", run_id)
    llm = ScriptedRuntimeLLM(
        _format_templates(_optional_mapping(raw_turn, "scripted"), run_id)
    )

    try:
        result = await runtime.run_turn(
            thread_id=thread_id,
            message=message,
            user_id=user_id,
            llm_client=llm,
            response_llm_client=llm,
            expected_liveness=_format_optional(
                raw_turn.get("expected_liveness"), run_id
            ),
            session_transcript_soft_limit=raw_turn.get("session_transcript_soft_limit"),
        )
    except Exception as exc:  # noqa: BLE001 - eval artifact needs exact error
        return {
            "ok": False,
            "thread_id": thread_id,
            "message": message,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "status_after": (await runtime.session_status(thread_id)).value,
            "structured_calls": getattr(llm, "structured_calls", {}),
        }

    return {
        "ok": True,
        "thread_id": thread_id,
        "message": message,
        "state_after": _state_summary(result.state),
        "status_after": (await runtime.session_status(thread_id)).value,
        "structured_calls": getattr(llm, "structured_calls", {}),
    }


async def _thread_observations(
    runtime: Any,
    specs: list[Mapping[str, Any]],
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for spec in specs:
        thread_id = _required_formatted(spec.get("thread_id"), "thread_id", run_id)
        state = await runtime.get_state(thread_id)
        observations.append(
            {
                "thread_id": thread_id,
                "status": (await runtime.session_status(thread_id)).value,
                "exists": state is not None,
                "state": _state_summary(state),
            }
        )
    return observations


def _parse_step(raw_step: Mapping[str, Any]) -> RuntimeRecoveryStep:
    return RuntimeRecoveryStep(
        type=str(raw_step["type"]),
        thread_id=_optional_str(raw_step.get("thread_id")),
        user_id=_optional_str(raw_step.get("user_id")),
        message=_optional_str(raw_step.get("message")),
        expected_liveness=_optional_str(raw_step.get("expected_liveness")),
        session_transcript_soft_limit=raw_step.get("session_transcript_soft_limit"),
        scripted=dict(_optional_mapping(raw_step, "scripted")),
        turns=[
            dict(item) for item in _mapping_list(raw_step.get("turns", []), "turns")
        ],
        expected=dict(_optional_mapping(raw_step, "expected")),
    )


def _step_as_turn(step: RuntimeRecoveryStep) -> dict[str, Any]:
    return {
        "thread_id": step.thread_id,
        "user_id": step.user_id,
        "message": step.message,
        "expected_liveness": step.expected_liveness,
        "session_transcript_soft_limit": step.session_transcript_soft_limit,
        "scripted": step.scripted,
    }


def _step_artifact(
    step: RuntimeRecoveryStep,
    *,
    step_index: int,
    artifact: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "type": step.type,
        **dict(artifact),
        "expected": _format_templates(step.expected, run_id),
    }


def _grade_case(
    case: RuntimeRecoveryCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for step in _mapping_list(artifact.get("steps", []), "steps"):
        _grade_expected(failures, label=f"step {step.get('step_index')}", item=step)

    final = artifact.get("final")
    if isinstance(final, Mapping):
        _grade_thread_states(
            failures,
            label="final",
            observations=final.get("thread_states"),
            expected=_optional_mapping(final, "expected").get("thread_states", []),
        )
    else:
        failures.append("final artifact is not a mapping")
    return failures


def _grade_expected(
    failures: list[str],
    *,
    label: str,
    item: Mapping[str, Any],
) -> None:
    expected = item.get("expected")
    if not isinstance(expected, Mapping):
        return
    if item.get("exception_type") and "exception_type" not in expected:
        failures.append(
            f"{label}.exception_type: unexpected {item.get('exception_type')!r}"
        )
    for key in (
        "status_after",
        "exception_type",
        "result_count",
        "exception_count",
        "stored_arc_exists",
    ):
        if key in expected and item.get(key) != expected[key]:
            failures.append(
                f"{label}.{key}: expected {expected[key]!r}, got {item.get(key)!r}"
            )
    _grade_thread_states(
        failures,
        label=label,
        observations=item.get("thread_states"),
        expected=expected.get("thread_states", []),
    )


def _grade_thread_states(
    failures: list[str],
    *,
    label: str,
    observations: Any,
    expected: Any,
) -> None:
    if not expected:
        return
    observed_by_id = {
        item.get("thread_id"): item
        for item in _mapping_list(observations, f"{label}.thread_states")
    }
    for spec in _mapping_list(expected, f"{label}.expected.thread_states"):
        thread_id = spec.get("thread_id")
        observed = observed_by_id.get(thread_id)
        if not isinstance(observed, Mapping):
            failures.append(f"{label}.thread[{thread_id!r}] missing observation")
            continue
        state = (
            observed.get("state") if isinstance(observed.get("state"), Mapping) else {}
        )
        for key in ("status", "exists"):
            if key in spec and observed.get(key) != spec[key]:
                failures.append(
                    f"{label}.thread[{thread_id!r}].{key}: expected "
                    f"{spec[key]!r}, got {observed.get(key)!r}"
                )
        for key in ("transcript_length", "assistant_turn_count", "turn_count"):
            if key in spec and state.get(key) != spec[key]:
                failures.append(
                    f"{label}.thread[{thread_id!r}].{key}: expected "
                    f"{spec[key]!r}, got {state.get(key)!r}"
                )


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
        "transcript_length": len(transcript) if isinstance(transcript, list) else None,
        "assistant_turn_count": len(assistant_turns),
        "turn_count": session_progress.get("turn_count")
        if isinstance(session_progress, Mapping)
        else None,
        "exercise_state": _jsonify(state.get("exercise_state") or {}),
    }


def _format_templates(value: Any, run_id: str) -> Any:
    if isinstance(value, str):
        return value.replace("{run_id}", run_id)
    if isinstance(value, list):
        return [_format_templates(item, run_id) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _format_templates(item, run_id) for key, item in value.items()
        }
    return value


def _required_formatted(value: Any, field: str, run_id: str) -> str:
    if value is None:
        raise ValueError(f"Missing required {field}.")
    return str(_format_templates(str(value), run_id))


def _format_optional(value: Any, run_id: str) -> str | None:
    if value is None:
        return None
    return str(_format_templates(str(value), run_id))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if item is None:
        return {}
    if not isinstance(item, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return item


def _mapping_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list.")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{field} must contain only objects.")
    return list(value)


def _jsonify(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonify(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _resolve_database_url(backend: str, override: str | None) -> str | None:
    if backend == "sqlite":
        return None
    return (
        override
        or os.getenv("OPENCOUCH_TEST_POSTGRES_DSN")
        or os.getenv("OPENCOUCH_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://opencouch:opencouch@localhost:5432/opencouch"
    )


def _build_evaluator(args: argparse.Namespace) -> RuntimeRecoveryTrajectoryEvaluator:
    return RuntimeRecoveryTrajectoryEvaluator(
        dataset_path=args.dataset or _DEFAULT_DATASET,
        backend=args.backend,
        database_url=_resolve_database_url(args.backend, args.database_url),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser(
        "Evaluate runtime recovery, liveness, and concurrency trajectories."
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
        help="Override Postgres DSN. Defaults to env vars or local compose DSN.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    return run_evaluator_cli(_build_evaluator, parser=parser, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
