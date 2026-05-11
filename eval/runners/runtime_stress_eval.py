"""Run scripted long-session runtime stress checks."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
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

_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "runtime" / "stress_v1.json"


@dataclass(frozen=True)
class RuntimeStressCase:
    """Parsed runtime stress case."""

    id: str
    thread_id: str
    user_id: str | None
    turn_count: int
    message_template: str
    description: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePaths:
    """Per-case SQLite fallback paths."""

    thread: Path
    memory: Path
    crisis: Path
    feedback: Path


class RuntimeStressEvaluator(BaseEvaluator[RuntimeStressCase]):
    """Run scripted long-session runtime stress checks."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        backend: str,
        database_url: str | None,
    ) -> None:
        super().__init__(dataset_path=dataset_path, name=f"runtime_stress_{backend}")
        self.backend = backend
        self.database_url = database_url

    def parse_case(self, raw_case: Any) -> RuntimeStressCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Runtime stress eval cases must be JSON objects.")
        return RuntimeStressCase(
            id=str(raw_case["id"]),
            thread_id=str(raw_case["thread_id"]),
            user_id=_optional_str(raw_case.get("user_id")),
            turn_count=int(raw_case["turn_count"]),
            message_template=str(raw_case["message_template"]),
            description=str(raw_case.get("description", "")),
            runtime=dict(_optional_mapping(raw_case, "runtime")),
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: RuntimeStressCase, index: int) -> str:
        return case.id

    async def run_case(self, case: RuntimeStressCase) -> EvalResult:
        artifact = await self._run_stress_case(case)
        failures = _grade_case(artifact)
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

    async def _run_stress_case(self, case: RuntimeStressCase) -> dict[str, Any]:
        run_id = f"eval-{case.id}-{uuid4().hex[:8]}"
        thread_id = _format_text(case.thread_id, run_id)
        user_id = _format_optional(case.user_id, run_id)
        with tempfile.TemporaryDirectory(prefix="opencouch-runtime-stress-") as tmp:
            paths = RuntimePaths(
                thread=Path(tmp) / "threads.sqlite3",
                memory=Path(tmp) / "memory.sqlite3",
                crisis=Path(tmp) / "crisis.sqlite3",
                feedback=Path(tmp) / "feedback.sqlite3",
            )
            runtime = self._build_runtime(case, paths=paths)
            async with runtime:
                started = time.perf_counter()
                turn_durations_ms: list[float] = []
                for index in range(case.turn_count):
                    turn_started = time.perf_counter()
                    llm = ScriptedRuntimeLLM(
                        {"response_text": (f"Scripted stress response {index + 1}.")}
                    )
                    await runtime.run_turn(
                        thread_id=thread_id,
                        user_id=user_id,
                        message=_format_text(
                            case.message_template,
                            run_id,
                            turn_index=index + 1,
                        ),
                        llm_client=llm,
                        response_llm_client=llm,
                    )
                    turn_durations_ms.append(
                        (time.perf_counter() - turn_started) * 1000
                    )

                state = await runtime.get_state(thread_id)
                history = await runtime.get_history(thread_id)
                status = await runtime.session_status(thread_id)

        total_ms = (time.perf_counter() - started) * 1000
        assistant_turns = [
            item
            for item in (state or {}).get("transcript", [])
            if isinstance(item, Mapping) and item.get("role") == "assistant"
        ]
        session_progress = (state or {}).get("session_progress") or {}
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "turn_count_requested": case.turn_count,
            "status": status.value,
            "history_count": len(history),
            "transcript_length": len((state or {}).get("transcript") or []),
            "assistant_turn_count": len(assistant_turns),
            "turn_count": session_progress.get("turn_count")
            if isinstance(session_progress, Mapping)
            else None,
            "total_ms": total_ms,
            "avg_turn_ms": total_ms / case.turn_count,
            "max_turn_ms": max(turn_durations_ms) if turn_durations_ms else 0.0,
            "expected": _format_templates(case.expected, run_id),
        }

    def _build_runtime(self, case: RuntimeStressCase, *, paths: RuntimePaths) -> Any:
        from agent.memory.embeddings import NullEmbeddingProvider
        from agent.memory.modes import MemoryMode
        from agent.persistence import PersistentAgentRuntime

        database_url = self.database_url if self.backend == "postgres" else None
        runtime_config = case.runtime
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
            speculative_memory_prefetch=bool(
                runtime_config.get("speculative_memory_prefetch", False)
            ),
            session_sweep_interval_seconds=3600.0,
        )


def _grade_case(artifact: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = artifact.get("expected")
    if not isinstance(expected, Mapping):
        return failures
    for key in (
        "status",
        "history_count",
        "transcript_length",
        "assistant_turn_count",
        "turn_count",
    ):
        if key in expected and artifact.get(key) != expected[key]:
            failures.append(
                f"{key}: expected {expected[key]!r}, got {artifact.get(key)!r}"
            )
    return failures


def _format_text(value: str, run_id: str, *, turn_index: int | None = None) -> str:
    result = value.replace("{run_id}", run_id)
    if turn_index is not None:
        result = result.replace("{turn_index}", str(turn_index))
    return result


def _format_optional(value: str | None, run_id: str) -> str | None:
    if value is None:
        return None
    return _format_text(value, run_id)


def _format_templates(value: Any, run_id: str) -> Any:
    if isinstance(value, str):
        return _format_text(value, run_id)
    if isinstance(value, list):
        return [_format_templates(item, run_id) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _format_templates(item, run_id) for key, item in value.items()
        }
    return value


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


def _build_evaluator(args: argparse.Namespace) -> RuntimeStressEvaluator:
    return RuntimeStressEvaluator(
        dataset_path=args.dataset or _DEFAULT_DATASET,
        backend=args.backend,
        database_url=_resolve_database_url(args.backend, args.database_url),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Run scripted long-session runtime stress checks.")
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
