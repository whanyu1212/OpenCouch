"""In-process execution of a scripted multi-turn case through `run_turn` (HY-239)."""

import json
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

from ag_ui.core import BaseEvent, RunErrorEvent, StateSnapshotEvent, ToolCallStartEvent
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    infer_model,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from starlette.requests import Request
from starlette.responses import StreamingResponse
from typesafe_sdk import Noul

from evals.runner.cases import (
    ConversationCase,
    ExpectedPath,
    LevelName,
    RecordsSeed,
    TurnScript,
)
from evals.runner.runs import (
    FailureClass,
    ModelCallRecord,
    RunStatus,
    SavedRunRow,
    TurnRecord,
)
from opencouch_agent.coach import CoachDeps, build_coach_agent
from opencouch_agent.program.models import ProgramPosition
from opencouch_agent.program.records import InMemoryRecordStore, RecordStore
from opencouch_agent.safety import JevClient
from opencouch_agent.safety.audit import InMemoryCrisisAuditLog
from opencouch_agent.safety.crisis import CrisisResponder, build_crisis_agent
from opencouch_agent.safety.crisis_resources import load_shipped_catalog
from opencouch_agent.safety.screening import user_check_state
from opencouch_agent.sessions import InMemorySessionStore
from opencouch_agent.turn import TurnAgents, run_turn

DEFAULT_BASE_TIME: Final = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
DEFAULT_TURN_STEP: Final = timedelta(minutes=1)
DEFAULT_TURN_TIMEOUT: Final = timedelta(seconds=60)
_LATEST_MARKER: Final = user_check_state([], "").rsplit("\n\n", 1)[1]


class ServedModelMismatchError(RuntimeError):
    """The model that answered a call did not match the requested model."""


@dataclass
class EvalInfraError(RuntimeError):
    """An infrastructure failure during a turn, recorded in `errors.jsonl`."""

    failure_class: FailureClass
    message: str
    turn_index: int | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


class ModelRecorder(WrapperModel):
    """Wraps a Pydantic AI `Model` to verify served models and record usage/timing."""

    def __init__(self, wrapped: Model | str, *, requested_model: str) -> None:
        super().__init__(infer_model(wrapped))
        self.requested_model = requested_model
        self.expected_prefix = requested_model.split(":", 1)[-1]
        self.calls: list[ModelCallRecord] = []
        self.last_error: BaseException | None = None

    def take_calls(self) -> list[ModelCallRecord]:
        """Return and clear the model calls recorded since the last turn."""
        recorded = list(self.calls)
        self.calls.clear()
        return recorded

    def take_error(self) -> BaseException | None:
        """Return and clear any exception raised during the last turn."""
        error = self.last_error
        self.last_error = None
        return error

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Record a non-streamed model request and check the served model."""
        start = time.perf_counter()
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            self._record_response(response, elapsed_ms)
            return response
        except Exception as error:
            self.last_error = error
            raise

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Record a streamed model request and check the served model."""
        start = time.perf_counter()
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                try:
                    yield stream
                finally:
                    elapsed_ms = round((time.perf_counter() - start) * 1000)
                    self._record_response(stream.get(), elapsed_ms)
        except Exception as error:
            self.last_error = error
            raise

    def _record_response(self, response: ModelResponse, elapsed_ms: int) -> None:
        served = response.model_name or self.wrapped.model_name
        if not _served_matches(self.expected_prefix, served):
            raise ServedModelMismatchError(
                f"Requested model {self.requested_model!r}, but served {served!r}"
            )
        usage = response.usage
        retries = len(response.failed_attempts) if response.failed_attempts else 0
        self.calls.append(
            ModelCallRecord(
                served_model=served,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                reasoning_tokens=usage.output_reasoning_tokens,
                total_tokens=usage.total_tokens,
                finish_reason=response.finish_reason,
                model_ms=elapsed_ms,
                retries=retries,
            )
        )


def _served_matches(expected: str, served: str) -> bool:
    if expected == served or served.startswith(f"{expected}-"):
        return True
    return expected == "test" and served.startswith("function:")


class _TimedJev:
    """Measures gate time and overrides readings when a scripted turn provides one."""

    def __init__(self, inner: JevClient | None) -> None:
        self._inner = inner
        self.override: Mapping[str, float] | None = None
        self.last_ms: int = 0

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        start = time.perf_counter()
        try:
            if self.override is not None:
                reading = self.override
                return {name: reading.get(name, 0.0) for name in questions}
            if self._inner is not None:
                return await self._inner.ask(state, questions)
            latest = state.rsplit(_LATEST_MARKER, 1)[-1]
            _ = latest
            return dict.fromkeys(questions, 0.0)
        finally:
            self.last_ms = round((time.perf_counter() - start) * 1000)


@dataclass
class _StreamFold:
    reply: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    first_token_ms: int | None = None
    run_error: tuple[str | None, str] | None = None


async def play_case(
    case: ConversationCase,
    *,
    rep: int = 1,
    requested_model: str = "test",
    coach_model: Model | str | None = None,
    crisis_model: Model | str | None = None,
    coach_agent: Agent[CoachDeps, str] | None = None,
    jev: JevClient | None = None,
    turn_timeout: timedelta = DEFAULT_TURN_TIMEOUT,
) -> SavedRunRow:
    """Play a scripted conversation case through the real `run_turn` pipeline.

    Creates a fresh in-memory session store, record store, audit log and thread
    for `(case, rep)`. Raises `EvalInfraError` if a turn fails due to an
    infrastructure error or served-model mismatch.
    """
    store = InMemorySessionStore()
    records = InMemoryRecordStore()
    audit = InMemoryCrisisAuditLog()
    thread_id = f"{case.id}-rep-{rep}"
    user_id = f"eval-user-{case.id}-rep-{rep}"
    await _seed_records(records, user_id, case.records, DEFAULT_BASE_TIME)

    coach_recorder = ModelRecorder(
        coach_model or requested_model, requested_model=requested_model
    )
    crisis_recorder = ModelRecorder(
        crisis_model or coach_model or requested_model,
        requested_model=requested_model,
    )
    coach = (
        coach_agent.override(model=coach_recorder) if coach_agent is not None else None
    )
    built_coach = (
        coach_agent if coach_agent is not None else build_coach_agent(coach_recorder)
    )
    agents = TurnAgents(
        coach=built_coach,
        crisis=CrisisResponder.from_catalog(
            build_crisis_agent(crisis_recorder), load_shipped_catalog()
        ),
    )
    timed_jev = _TimedJev(jev)

    if coach is not None:
        with coach:
            return await _play_turns(
                case,
                rep=rep,
                requested_model=requested_model,
                thread_id=thread_id,
                user_id=user_id,
                agents=agents,
                timed_jev=timed_jev,
                coach_recorder=coach_recorder,
                crisis_recorder=crisis_recorder,
                store=store,
                records=records,
                audit=audit,
                turn_timeout=turn_timeout,
            )
    return await _play_turns(
        case,
        rep=rep,
        requested_model=requested_model,
        thread_id=thread_id,
        user_id=user_id,
        agents=agents,
        timed_jev=timed_jev,
        coach_recorder=coach_recorder,
        crisis_recorder=crisis_recorder,
        store=store,
        records=records,
        audit=audit,
        turn_timeout=turn_timeout,
    )


async def _play_turns(
    case: ConversationCase,
    *,
    rep: int,
    requested_model: str,
    thread_id: str,
    user_id: str,
    agents: TurnAgents,
    timed_jev: _TimedJev,
    coach_recorder: ModelRecorder,
    crisis_recorder: ModelRecorder,
    store: InMemorySessionStore,
    records: RecordStore,
    audit: InMemoryCrisisAuditLog,
    turn_timeout: timedelta,
) -> SavedRunRow:
    turn_records: list[TurnRecord] = []
    at = DEFAULT_BASE_TIME
    for index, script in enumerate(case.turns):
        if script.minutes is not None:
            at = DEFAULT_BASE_TIME + timedelta(minutes=script.minutes)
        elif index > 0:
            at = at + DEFAULT_TURN_STEP
        session_before = await store.load(thread_id)
        offer_id = session_before.risk.check_in_offer
        record = await _run_one_turn(
            script,
            turn_index=index,
            at=at,
            offer_id=offer_id,
            thread_id=thread_id,
            user_id=user_id,
            agents=agents,
            timed_jev=timed_jev,
            coach_recorder=coach_recorder,
            crisis_recorder=crisis_recorder,
            store=store,
            records=records,
            audit=audit,
            turn_timeout=turn_timeout,
        )
        turn_records.append(record)

    served_models = sorted(
        {call.served_model for turn in turn_records for call in turn.model_calls}
    )
    status: RunStatus = (
        "truncated" if any(turn.truncated for turn in turn_records) else "ok"
    )
    return SavedRunRow(
        case_id=case.id,
        rep=rep,
        split=case.split,
        tags=list(case.tags),
        requested_model=requested_model,
        served_models=served_models,
        status=status,
        turns=turn_records,
    )


async def _run_one_turn(
    script: TurnScript,
    *,
    turn_index: int,
    at: datetime,
    offer_id: str | None,
    thread_id: str,
    user_id: str,
    agents: TurnAgents,
    timed_jev: _TimedJev,
    coach_recorder: ModelRecorder,
    crisis_recorder: ModelRecorder,
    store: InMemorySessionStore,
    records: RecordStore,
    audit: InMemoryCrisisAuditLog,
    turn_timeout: timedelta,
) -> TurnRecord:
    timed_jev.override = script.jev_reading if script.jev_reading else None
    timed_jev.last_ms = 0
    coach_recorder.take_calls()
    crisis_recorder.take_calls()
    coach_recorder.take_error()
    crisis_recorder.take_error()

    request = _build_ag_ui_request(
        thread_id=thread_id,
        turn_index=turn_index,
        text=script.user,
        check_in=script.check_in,
        offer_id=offer_id,
    )
    start = time.perf_counter()
    try:
        response = await run_turn(
            request,
            user_id=user_id,
            agents=agents,
            jev=timed_jev,
            store=store,
            records=records,
            audit=audit,
            turn_timeout=turn_timeout,
            at=at,
        )
        fold = await _consume_response(cast(StreamingResponse, response), start)
    except EvalInfraError as error:
        error.turn_index = turn_index
        raise
    except Exception as error:
        raise _classify_error(error, turn_index=turn_index) from error

    turn_ms = round((time.perf_counter() - start) * 1000)
    _raise_turn_failure(
        turn_index,
        fold,
        coach_recorder.take_error() or crisis_recorder.take_error(),
    )

    session_after = await store.load(thread_id)
    risk_level = cast(LevelName, session_after.risk.level.name)
    coach_calls = coach_recorder.take_calls()
    crisis_calls = crisis_recorder.take_calls()
    model_calls = [*coach_calls, *crisis_calls]
    path: ExpectedPath = (
        "crisis"
        if crisis_calls or session_after.risk.level.name == "CRISIS"
        else "coach"
    )
    truncated = any(call.finish_reason == "length" for call in model_calls)
    return TurnRecord(
        turn_index=turn_index,
        user=script.user,
        reply="".join(fold.reply),
        path=path,
        risk_level=risk_level,
        state=fold.state or dict(session_after.state),
        tool_calls=fold.tool_calls,
        first_token_ms=fold.first_token_ms,
        turn_ms=turn_ms,
        gate_ms=timed_jev.last_ms,
        model_calls=model_calls,
        truncated=truncated,
    )


def _raise_turn_failure(
    turn_index: int, fold: _StreamFold, recorder_error: BaseException | None
) -> None:
    if recorder_error is not None:
        raise _classify_error(recorder_error, turn_index=turn_index)
    if fold.run_error is not None:
        code, message = fold.run_error
        failure: FailureClass = "timeout" if code == "turn_timeout" else "api_error"
        raise EvalInfraError(
            failure_class=failure, message=message, turn_index=turn_index
        )


async def _seed_records(
    records: RecordStore, user_id: str, seed: RecordsSeed, at: datetime
) -> None:
    if seed.position is not None:
        await records.save_position(
            user_id, ProgramPosition(at, seed.position.week, seed.position.step)
        )
    for worry in seed.worries:
        await records.add_worry(
            user_id, worry.text, source=worry.source, captured_at=at
        )
    for entry in seed.formulation:
        await records.add_formulation_entry(
            user_id, entry.kind, entry.text, confirmed_at=at
        )


def _build_ag_ui_request(
    *,
    thread_id: str,
    turn_index: int,
    text: str,
    check_in: str | None,
    offer_id: str | None,
) -> Request:
    forwarded: dict[str, Any] = {}
    if check_in is not None:
        forwarded["crisis_check_in"] = {
            "offer_id": offer_id or "unoffered",
            "answer": check_in,
        }
    payload = {
        "threadId": thread_id,
        "runId": f"run-{turn_index}",
        "state": {},
        "messages": [{"id": f"m-{turn_index}", "role": "user", "content": text}],
        "tools": [],
        "context": [],
        "forwardedProps": forwarded,
    }
    body = json.dumps(payload).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/agent",
        "headers": [
            (b"content-type", b"application/json"),
            (b"accept", b"text/event-stream"),
        ],
    }
    return Request(scope, receive)


async def _consume_response(response: StreamingResponse, start: float) -> _StreamFold:
    fold = _StreamFold()
    buffer = ""
    async for chunk in response.body_iterator:
        text = (
            chunk.decode("utf-8")
            if isinstance(chunk, bytes | bytearray)
            else str(chunk)
        )
        buffer += text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            _fold_sse_line(fold, line.rstrip("\r"), start)
    if buffer.strip():
        _fold_sse_line(fold, buffer.rstrip("\r"), start)
    return fold


def _fold_sse_line(fold: _StreamFold, line: str, start: float) -> None:
    if not line.startswith("data: "):
        return
    raw = json.loads(line.removeprefix("data: "))
    kind = raw.get("type")
    if kind == "TEXT_MESSAGE_CONTENT":
        if fold.first_token_ms is None:
            fold.first_token_ms = round((time.perf_counter() - start) * 1000)
        fold.reply.append(str(raw.get("delta", "")))
    elif kind == "TOOL_CALL_START":
        fold.tool_calls.append(str(raw.get("toolCallName", "")))
    elif kind == "STATE_SNAPSHOT":
        snapshot = raw.get("snapshot")
        if isinstance(snapshot, dict):
            fold.state = snapshot
    elif kind == "RUN_ERROR":
        fold.run_error = (raw.get("code"), str(raw.get("message", "run error")))


def _classify_error(error: BaseException, *, turn_index: int) -> EvalInfraError:
    if isinstance(error, ServedModelMismatchError):
        return EvalInfraError("served_model_mismatch", str(error), turn_index)
    if isinstance(error, TimeoutError):
        return EvalInfraError("timeout", str(error) or "timed out", turn_index)
    message = f"{type(error).__name__}: {error}"
    lowered = message.lower()
    if "timeout" in lowered:
        failure: FailureClass = "timeout"
    elif "ratelimit" in lowered or "429" in lowered:
        failure = "rate_limited"
    elif any(token in lowered for token in ("connection", "503", "unavailable")):
        failure = "unavailable"
    else:
        failure = "api_error"
    return EvalInfraError(failure, message, turn_index)


__all__ = [
    "DEFAULT_BASE_TIME",
    "DEFAULT_TURN_STEP",
    "DEFAULT_TURN_TIMEOUT",
    "BaseEvent",
    "EvalInfraError",
    "ModelRecorder",
    "RunErrorEvent",
    "ServedModelMismatchError",
    "StateSnapshotEvent",
    "ToolCallStartEvent",
    "play_case",
]
