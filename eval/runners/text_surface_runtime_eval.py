"""Evaluate text API and CLI surfaces against the persistent runtime."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from httpx import ASGITransport, AsyncClient

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

_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "surface" / "text_runtime_v1.json"


@dataclass(frozen=True)
class TextSurfaceStep:
    """One API or CLI operation in a text-surface trajectory."""

    type: str
    thread_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    feedback: str | None = None
    command: str | None = None
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextSurfaceCase:
    """Parsed text-surface runtime case."""

    id: str
    description: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    steps: list[TextSurfaceStep] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePaths:
    """Per-case SQLite fallback paths."""

    thread: Path
    memory: Path
    crisis: Path
    feedback: Path


class SurfaceScriptedRuntimeLLM(ScriptedRuntimeLLM):
    """Scripted runtime LLM with explicit failure hooks for surface evals."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        if self.scripted.get("raise_on_text"):
            self.text_calls += 1
            raise RuntimeError(_scripted_error(self.scripted))
        return await super().generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            use_search=use_search,
        )

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        if self.scripted.get("raise_on_stream"):
            self.stream_calls += 1
            raise RuntimeError(_scripted_error(self.scripted))
        if self.scripted.get("raise_after_stream_chunks"):
            self.stream_calls += 1
            chunks = self.scripted.get("response_chunks")
            if not isinstance(chunks, list):
                chunks = [self.scripted.get("response_text", "partial stream reply")]
            for chunk in chunks:
                yield str(chunk)
            raise RuntimeError(_scripted_error(self.scripted))
        async for chunk in super().generate_text_stream(
            prompt=prompt,
            system_instruction=system_instruction,
        ):
            yield chunk

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
        if _should_raise_on_schema(self.scripted, schema_name):
            self.structured_calls[schema_name] = (
                self.structured_calls.get(schema_name, 0) + 1
            )
            raise RuntimeError(_scripted_error(self.scripted))

        if schema_name == "LookupPreflightDecision":
            self.structured_calls[schema_name] = (
                self.structured_calls.get(schema_name, 0) + 1
            )
            return response_schema(
                **_scripted_mapping(
                    self.scripted,
                    "lookup_preflight",
                    {
                        "status": "search",
                        "search_query": "scripted surface lookup",
                        "answer": "",
                        "reasoning": "scripted lookup preflight",
                    },
                )
            )
        if schema_name == "GroundedLookupResult":
            self.structured_calls[schema_name] = (
                self.structured_calls.get(schema_name, 0) + 1
            )
            return response_schema(
                **_scripted_mapping(
                    self.scripted,
                    "grounded_lookup_result",
                    {
                        "status": "answered",
                        "answer": (
                            "Scripted grounded surface answer.\n\n"
                            "Sources:\n- Evaluation source"
                        ),
                        "sources": ["Evaluation source"],
                        "source_quality": "reputable",
                        "reasoning": "scripted grounded result",
                    },
                )
            )
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


class _FakeWebSocket:
    """Minimal WebSocket test double for calling the route handler directly."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self.close_code: int | None = None
        self.close_reason = ""

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        return dict(self.payload)

    async def send_json(self, data: Mapping[str, Any]) -> None:
        self.sent.append(dict(data))

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.close_reason = reason


class TextSurfaceRuntimeEvaluator(BaseEvaluator[TextSurfaceCase]):
    """Run text-surface runtime checks."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        backend: str,
        database_url: str | None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"text_surface_runtime_{backend}",
        )
        self.backend = backend
        self.database_url = database_url

    def parse_case(self, raw_case: Any) -> TextSurfaceCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Text surface eval cases must be JSON objects.")
        return TextSurfaceCase(
            id=str(raw_case["id"]),
            description=str(raw_case.get("description", "")),
            runtime=dict(_optional_mapping(raw_case, "runtime")),
            steps=[
                _parse_step(item)
                for item in _mapping_list(raw_case.get("steps", []), "steps")
            ],
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: TextSurfaceCase, index: int) -> str:
        return case.id

    async def run_case(self, case: TextSurfaceCase) -> EvalResult:
        artifact = await self._run_trajectory(case)
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

    async def _run_trajectory(self, case: TextSurfaceCase) -> dict[str, Any]:
        run_id = f"eval-{case.id}-{uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory(prefix="opencouch-text-surface-") as tmp:
            paths = RuntimePaths(
                thread=Path(tmp) / "threads.sqlite3",
                memory=Path(tmp) / "memory.sqlite3",
                crisis=Path(tmp) / "crisis.sqlite3",
                feedback=Path(tmp) / "feedback.sqlite3",
            )
            runtime = self._build_runtime(case, paths=paths)
            llm = SurfaceScriptedRuntimeLLM({})
            llm_ref: dict[str, Any] = {"client": llm}
            async with runtime:
                client = await _build_api_client(runtime=runtime, llm_ref=llm_ref)
                try:
                    steps = [
                        await self._run_step(
                            step,
                            runtime=runtime,
                            client=client,
                            llm=llm,
                            llm_ref=llm_ref,
                            run_id=run_id,
                        )
                        for step in case.steps
                    ]
                    final = await _final_observations(
                        runtime,
                        expected=_format_templates(case.expected, run_id),
                    )
                finally:
                    await client.aclose()

        return {
            "case_id": case.id,
            "run_id": run_id,
            "steps": steps,
            "final": final,
        }

    def _build_runtime(self, case: TextSurfaceCase, *, paths: RuntimePaths) -> Any:
        from agent.memory.embeddings import NullEmbeddingProvider
        from agent.memory.modes import MemoryMode
        from agent.persistence import PersistentAgentRuntime

        runtime_config = case.runtime
        database_url = self.database_url if self.backend == "postgres" else None
        llm = SurfaceScriptedRuntimeLLM({})
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
            default_llm_client=llm,
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
        step: TextSurfaceStep,
        *,
        runtime: Any,
        client: AsyncClient,
        llm: SurfaceScriptedRuntimeLLM,
        llm_ref: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        if step.type == "api_chat":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            step_llm = _set_step_llm(
                step,
                llm_ref=llm_ref,
                default_llm=llm,
                run_id=run_id,
            )
            body = {
                "thread_id": thread_id,
                "message": _required_formatted(step.message, "message", run_id),
            }
            user_id = _format_optional(step.user_id, run_id)
            if user_id is not None:
                body["user_id"] = user_id
            response = await client.post("/api/chat", json=body)
            return {
                "type": step.type,
                "status_code": response.status_code,
                "json": _response_json(response),
                "error_code": _json_error_code(_response_json(response)),
                "status_after": (await runtime.session_status(thread_id)).value,
                "has_active_session": await runtime.has_active_session(thread_id),
                "structured_calls": dict(step_llm.structured_calls),
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "api_chat_stream":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            step_llm = _set_step_llm(
                step,
                llm_ref=llm_ref,
                default_llm=llm,
                run_id=run_id,
            )
            body = {
                "thread_id": thread_id,
                "message": _required_formatted(step.message, "message", run_id),
            }
            user_id = _format_optional(step.user_id, run_id)
            if user_id is not None:
                body["user_id"] = user_id
            websocket = _FakeWebSocket(body)
            from api.routes.chat import chat_stream

            await chat_stream(
                websocket,  # type: ignore[arg-type]
                runtime=runtime,
                llm_client=step_llm,
                response_llm_clients={"fast": step_llm},
            )
            event_types = [str(event.get("type")) for event in websocket.sent]
            error = next(
                (event for event in websocket.sent if event.get("type") == "error"),
                {},
            )
            return {
                "type": step.type,
                "accepted": websocket.accepted,
                "events": websocket.sent,
                "event_types": event_types,
                "chunk_text": "".join(
                    str(event.get("text", ""))
                    for event in websocket.sent
                    if event.get("type") == "chunk"
                ),
                "done_count": event_types.count("done"),
                "error_code": error.get("code"),
                "close_code": websocket.close_code,
                "close_reason": websocket.close_reason,
                "status_after": (await runtime.session_status(thread_id)).value,
                "has_active_session": await runtime.has_active_session(thread_id),
                "structured_calls": dict(step_llm.structured_calls),
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "api_history":
            _set_step_llm(step, llm_ref=llm_ref, default_llm=llm, run_id=run_id)
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            response = await client.get(f"/api/threads/{thread_id}/history")
            payload = _response_json(response)
            return {
                "type": step.type,
                "status_code": response.status_code,
                "message_count": len(payload) if isinstance(payload, list) else None,
                "json": payload,
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "api_end":
            _set_step_llm(step, llm_ref=llm_ref, default_llm=llm, run_id=run_id)
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            body = {}
            if step.feedback is not None:
                body["feedback"] = _required_formatted(
                    step.feedback, "feedback", run_id
                )
            response = await client.post(f"/api/threads/{thread_id}/end", json=body)
            return {
                "type": step.type,
                "status_code": response.status_code,
                "json": _response_json(response),
                "feedback_count": await _feedback_count(runtime, thread_id),
                "status_after": (await runtime.session_status(thread_id)).value,
                "has_active_session": await runtime.has_active_session(thread_id),
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "api_memory_status":
            _set_step_llm(step, llm_ref=llm_ref, default_llm=llm, run_id=run_id)
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            params = {"thread_id": thread_id}
            user_id = _format_optional(step.user_id, run_id)
            if user_id is not None:
                params["user_id"] = user_id
            response = await client.get("/api/memory/status", params=params)
            return {
                "type": step.type,
                "status_code": response.status_code,
                "json": _response_json(response),
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "api_session_status":
            _set_step_llm(step, llm_ref=llm_ref, default_llm=llm, run_id=run_id)
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            response = await client.get(f"/api/threads/{thread_id}/session-status")
            payload = _response_json(response)
            has_active_session = (
                bool(payload.get("has_active_session"))
                if isinstance(payload, Mapping)
                else None
            )
            return {
                "type": step.type,
                "status_code": response.status_code,
                "json": payload,
                "has_active_session": has_active_session,
                "status_after": (await runtime.session_status(thread_id)).value,
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "seed_turn":
            thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
            step_llm = SurfaceScriptedRuntimeLLM(
                _format_templates(step.scripted, run_id)
            )
            result = await runtime.run_turn(
                thread_id=thread_id,
                message=_required_formatted(step.message, "message", run_id),
                user_id=_format_optional(step.user_id, run_id),
                llm_client=step_llm,
                response_llm_client=step_llm,
            )
            return {
                "type": step.type,
                "status_after": (await runtime.session_status(thread_id)).value,
                "has_active_session": await runtime.has_active_session(thread_id),
                "transcript_length": len(result.history),
                "expected": _format_templates(step.expected, run_id),
            }

        if step.type == "cli_command":
            return await _run_cli_command(
                step,
                runtime=runtime,
                llm=llm,
                run_id=run_id,
            )

        raise ValueError(f"Unsupported text surface step type {step.type!r}.")


async def _build_api_client(*, runtime: Any, llm_ref: Mapping[str, Any]) -> AsyncClient:
    from fastapi import FastAPI

    from api.dependencies import get_llm_client, get_response_llm_clients, get_runtime
    from api.router import api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[get_llm_client] = lambda: llm_ref["client"]
    app.dependency_overrides[get_response_llm_clients] = lambda: {
        "fast": llm_ref["client"]
    }

    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


def _set_step_llm(
    step: TextSurfaceStep,
    *,
    llm_ref: dict[str, Any],
    default_llm: SurfaceScriptedRuntimeLLM,
    run_id: str,
) -> SurfaceScriptedRuntimeLLM:
    if step.scripted:
        step_llm = SurfaceScriptedRuntimeLLM(_format_templates(step.scripted, run_id))
    else:
        step_llm = default_llm
    llm_ref["client"] = step_llm
    return step_llm


def _response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _json_error_code(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, Mapping):
        return None
    code = detail.get("code")
    return str(code) if code is not None else None


def _scripted_error(scripted: Mapping[str, Any]) -> str:
    return str(scripted.get("error", "scripted surface failure"))


def _should_raise_on_schema(scripted: Mapping[str, Any], schema_name: str) -> bool:
    configured = scripted.get("raise_on_schema")
    if isinstance(configured, str):
        return configured == schema_name
    if isinstance(configured, list):
        return schema_name in {str(item) for item in configured}
    return False


def _scripted_mapping(
    scripted: Mapping[str, Any],
    key: str,
    default: Mapping[str, Any],
) -> dict[str, Any]:
    value = scripted.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return dict(default)


async def _run_cli_command(
    step: TextSurfaceStep,
    *,
    runtime: Any,
    llm: Any,
    run_id: str,
) -> dict[str, Any]:
    from opencouch_cli.app import RunnerSession, handle_command

    thread_id = _required_formatted(step.thread_id, "thread_id", run_id)
    session = RunnerSession(
        requested_mode="eval",
        resolved_mode="eval",
        llm_client=llm,
        thread_id=thread_id,
        sqlite_path=":memory:",
        memory_mode="persistent",
        persistence_backend="postgres",
        user_id=_format_optional(step.user_id, run_id),
        response_llm_client=llm,
    )
    rendered: list[dict[str, str]] = []
    feedback = step.feedback

    with (
        patch("opencouch_cli.app._prompt_for_session_feedback", lambda: feedback),
        patch(
            "opencouch_cli.app.render_info",
            lambda message, style="panel": rendered.append(
                {"style": style, "message": message}
            ),
        ),
        patch("opencouch_cli.app.render_session_summary", lambda arc: None),
    ):
        should_continue = await handle_command(
            _required_formatted(step.command, "command", run_id),
            session,
            runtime,
        )

    return {
        "type": step.type,
        "should_continue": should_continue,
        "status_after": (await runtime.session_status(thread_id)).value,
        "feedback_count": await _feedback_count(runtime, thread_id),
        "rendered": rendered,
        "expected": _format_templates(step.expected, run_id),
    }


async def _final_observations(
    runtime: Any,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for spec in _mapping_list(expected.get("thread_states", []), "thread_states"):
        thread_id = str(spec.get("thread_id") or "")
        if not thread_id:
            continue
        state = await runtime.get_state(thread_id)
        observations.append(
            {
                "thread_id": thread_id,
                "status": (await runtime.session_status(thread_id)).value,
                "has_active_session": await runtime.has_active_session(thread_id),
                "exists": state is not None,
                "history_count": len(await runtime.get_history(thread_id)),
                "feedback_count": await _feedback_count(runtime, thread_id),
            }
        )
    return {"thread_states": observations, "expected": expected}


async def _feedback_count(runtime: Any, thread_id: str) -> int:
    from agent.memory.hashing import hash_session_id

    return len(
        await runtime.session_feedback_backend.alist_by_session(
            hash_session_id(thread_id)
        )
    )


def _parse_step(raw_step: Mapping[str, Any]) -> TextSurfaceStep:
    return TextSurfaceStep(
        type=str(raw_step["type"]),
        thread_id=_optional_str(raw_step.get("thread_id")),
        user_id=_optional_str(raw_step.get("user_id")),
        message=_optional_str(raw_step.get("message")),
        feedback=_optional_str(raw_step.get("feedback")),
        command=_optional_str(raw_step.get("command")),
        scripted=dict(_optional_mapping(raw_step, "scripted")),
        expected=dict(_optional_mapping(raw_step, "expected")),
    )


def _grade_case(artifact: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for index, step in enumerate(_mapping_list(artifact.get("steps", []), "steps")):
        _grade_expected(failures, label=f"step {index + 1}", item=step)

    final = artifact.get("final")
    if isinstance(final, Mapping):
        expected = _optional_mapping(final, "expected")
        _grade_final_threads(
            failures,
            observations=final.get("thread_states"),
            expected=expected.get("thread_states", []),
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
    for key in (
        "status_code",
        "message_count",
        "feedback_count",
        "status_after",
        "has_active_session",
        "should_continue",
        "transcript_length",
        "accepted",
        "close_code",
        "done_count",
        "error_code",
    ):
        if key in expected and item.get(key) != expected[key]:
            failures.append(
                f"{label}.{key}: expected {expected[key]!r}, got {item.get(key)!r}"
            )
    if "json_contains" in expected:
        text = str(item.get("json", ""))
        for phrase in expected["json_contains"]:
            if str(phrase) not in text:
                failures.append(f"{label}.json missing {str(phrase)!r}")
    if "event_types_contains" in expected:
        observed = {str(item) for item in item.get("event_types", [])}
        for event_type in expected["event_types_contains"]:
            if str(event_type) not in observed:
                failures.append(f"{label}.event_types missing {str(event_type)!r}")
    if "event_types_not_contains" in expected:
        observed = {str(item) for item in item.get("event_types", [])}
        for event_type in expected["event_types_not_contains"]:
            if str(event_type) in observed:
                failures.append(f"{label}.event_types contained {str(event_type)!r}")
    if "chunk_text_contains" in expected:
        text = str(item.get("chunk_text", ""))
        for phrase in expected["chunk_text_contains"]:
            if str(phrase) not in text:
                failures.append(f"{label}.chunk_text missing {str(phrase)!r}")


def _grade_final_threads(
    failures: list[str],
    *,
    observations: Any,
    expected: Any,
) -> None:
    observed = {
        item.get("thread_id"): item
        for item in _mapping_list(observations, "final.thread_states")
    }
    for spec in _mapping_list(expected, "final.expected.thread_states"):
        thread_id = spec.get("thread_id")
        item = observed.get(thread_id)
        if not isinstance(item, Mapping):
            failures.append(f"final.thread[{thread_id!r}] missing observation")
            continue
        for key in (
            "status",
            "has_active_session",
            "exists",
            "history_count",
            "feedback_count",
        ):
            if key in spec and item.get(key) != spec[key]:
                failures.append(
                    f"final.thread[{thread_id!r}].{key}: expected "
                    f"{spec[key]!r}, got {item.get(key)!r}"
                )


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


def _build_evaluator(args: argparse.Namespace) -> TextSurfaceRuntimeEvaluator:
    return TextSurfaceRuntimeEvaluator(
        dataset_path=args.dataset or _DEFAULT_DATASET,
        backend=args.backend,
        database_url=_resolve_database_url(args.backend, args.database_url),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser(
        "Evaluate text API and CLI surfaces against PersistentAgentRuntime."
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
