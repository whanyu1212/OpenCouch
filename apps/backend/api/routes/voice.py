"""Voice endpoints for OpenAI Realtime speech-to-speech sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException

from agent.voice import realtime
from agent.voice import tools as voice_tools
from agent.models import CrisisAssessment
from agent.observability.decorators import trace_event
from agent.observability.events import (
    VOICE_CONCURRENT_SAFETY_ASSESSED,
    VOICE_CONCURRENT_SAFETY_TURN_OBSERVED,
    VOICE_SAFETY_INTERRUPTION_DECIDED,
    VOICE_SAFETY_RESOURCES_RESOLVED,
)
from agent.voice.safety_proof import (
    InvalidVoiceSafetyInterruptionProof,
    VoiceSafetyInterruptionProofService,
)
from api.dependencies import (
    get_llm_client,
    get_runtime_selection,
)
from api.session_end import end_runtime_session
from api.models import (
    VoiceRealtimeSessionRequest,
    VoiceRealtimeSessionResponse,
    VoiceConcurrentSafetyRequest,
    VoiceConcurrentSafetyResponse,
    VoiceSafetyResourcesResponse,
    VoiceToolCallRequest,
    VoiceToolCallResponse,
    VoiceTurnRecordRequest,
    VoiceTurnRecordResponse,
    VoiceEndSessionRequest,
    VoiceEndSessionResponse,
)
from llm.base import BaseLLMClient

router = APIRouter(prefix="/voice", tags=["voice"])

_VOICE_SESSION_FAILURE_HTTP_STATUS = 500
_VOICE_TOOL_FAILURE_HTTP_STATUS = 500
_VOICE_TURN_FAILURE_HTTP_STATUS = 500
_VOICE_END_FAILURE_HTTP_STATUS = 500
_VOICE_SAFETY_PROOFS = VoiceSafetyInterruptionProofService()
_VOICE_SESSION_WATERMARK_TIMEOUT_SECONDS = 0.5
_VOICE_TOOL_TIMEOUT_SECONDS = 25.0

_CRISIS_VOICE_TOOL_NAMES = {
    "get_crisis_support_template",
    "lookup_crisis_resources",
}


@router.post("/realtime/session", response_model=VoiceRealtimeSessionResponse)
async def create_voice_realtime_session(
    body: VoiceRealtimeSessionRequest,
) -> VoiceRealtimeSessionResponse:
    """Create an ephemeral OpenAI Realtime client secret for browser voice."""

    selection = get_runtime_selection(body.memory_mode)
    memory_context = await selection.runtime.voice.voice_session_memory_context(
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=selection.memory_mode,
    )
    try:
        message_count = await asyncio.wait_for(
            selection.runtime.voice.voice_session_message_count(
                thread_id=body.thread_id
            ),
            timeout=_VOICE_SESSION_WATERMARK_TIMEOUT_SECONDS,
        )
    except Exception:
        message_count = 0
    session_config = realtime.build_realtime_session_config(
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=selection.memory_mode,
        memory_context=memory_context,
        assistant_voice=body.assistant_voice,
    )
    try:
        client_secret = await realtime.create_realtime_client_secret(
            session_config=session_config,
            safety_identifier=_voice_safety_identifier(
                thread_id=body.thread_id,
                user_id=body.user_id,
            ),
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=_VOICE_SESSION_FAILURE_HTTP_STATUS,
            detail={
                "code": "voice_realtime_session_failed",
                "message": message,
            },
        ) from exc

    return VoiceRealtimeSessionResponse(
        client_secret=client_secret,
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=selection.memory_mode,
        message_count=message_count,
        session_config=session_config,
    )


@router.post("/realtime/tools", response_model=VoiceToolCallResponse)
async def execute_voice_realtime_tool(
    body: VoiceToolCallRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceToolCallResponse:
    """Execute one app-owned OpenAI Realtime function tool call."""

    selection = get_runtime_selection(body.memory_mode)
    try:
        async with asyncio.timeout(_VOICE_TOOL_TIMEOUT_SECONDS):
            output = await voice_tools.execute_voice_tool_call(
                runtime=selection.runtime,
                tool_name=body.tool_name,
                arguments=body.arguments,
                thread_id=body.thread_id,
                user_id=body.user_id,
                client_turn_id=body.client_turn_id,
                current_user_message=body.current_user_message,
                transcript=body.transcript,
                memory_mode=selection.memory_mode,
                llm_client=llm_client,
            )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=_VOICE_TOOL_FAILURE_HTTP_STATUS,
            detail={
                "code": "voice_realtime_tool_failed",
                "message": message,
            },
        ) from exc

    return VoiceToolCallResponse(output=output)


@router.post("/realtime/turn", response_model=VoiceTurnRecordResponse)
async def record_voice_realtime_turn(
    body: VoiceTurnRecordRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceTurnRecordResponse:
    """Record a finalized voice user/assistant turn in app-owned history."""

    selection = get_runtime_selection(body.memory_mode)
    correlation_hash = (
        _voice_turn_correlation_hash(
            thread_id=body.thread_id,
            client_turn_id=body.client_turn_id,
        )
        if body.client_turn_id is not None
        else None
    )
    request_hash = _voice_turn_request_hash(body)
    try:
        receipt = (
            await selection.runtime.voice.recorded_voice_turn_receipt(
                thread_id=body.thread_id,
                correlation_hash=correlation_hash,
                request_hash=request_hash,
            )
            if correlation_hash is not None
            else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "voice_realtime_turn_idempotency_conflict",
                "message": str(exc),
            },
        ) from exc
    if receipt is not None:
        return VoiceTurnRecordResponse(
            recorded=True,
            thread_id=body.thread_id,
            message_count=receipt.message_count,
            post_turn_safety=receipt.post_turn_safety,
        )

    state = None
    if state is None:
        safety_assessment = None
        if body.outcome == "safety_interrupted":
            try:
                allow_expired_proof = (
                    correlation_hash is not None
                    and await selection.runtime.voice.has_verified_pending_safety_interruption(
                        thread_id=body.thread_id,
                        correlation_hash=correlation_hash,
                        request_hash=request_hash,
                    )
                )
                proof = _VOICE_SAFETY_PROOFS.verify(
                    body.interruption_token or "",
                    thread_id=body.thread_id,
                    client_turn_id=body.client_turn_id or "",
                    user_text=body.user_text,
                    user_id=body.user_id,
                    memory_mode=str(selection.memory_mode),
                    allow_expired=allow_expired_proof,
                )
            except InvalidVoiceSafetyInterruptionProof as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "voice_safety_interruption_proof_invalid",
                        "message": (
                            "The interrupted turn was not authorized by a current "
                            "server safety decision."
                        ),
                    },
                ) from exc
            except ValueError as exc:
                if "client_turn_id was already used" in str(exc):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "voice_realtime_turn_idempotency_conflict",
                            "message": str(exc),
                        },
                    ) from exc
                raise
            safety_assessment = CrisisAssessment(
                level=proof.risk_level,
                confidence="high",
                reason="voice_concurrent_safety_interruption",
                needs_crisis_response=True,
            )
        try:
            state = await selection.runtime.voice.record_voice_turn(
                thread_id=body.thread_id,
                user_id=body.user_id,
                user_text=body.user_text,
                assistant_text=body.assistant_text,
                outcome=body.outcome,
                route=body.route,
                response_style=body.response_style,
                tool_calls=[call.model_dump(mode="json") for call in body.tool_calls],
                llm_client=llm_client,
                correlation_hash=correlation_hash,
                request_hash=request_hash,
                safety_assessment=safety_assessment,
            )
        except ValueError as exc:
            if "client_turn_id was already used" in str(exc):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "voice_realtime_turn_idempotency_conflict",
                        "message": str(exc),
                    },
                ) from exc
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise HTTPException(
                status_code=_VOICE_TURN_FAILURE_HTTP_STATUS,
                detail={
                    "code": "voice_realtime_turn_record_failed",
                    "message": message,
                },
            ) from exc

    diagnostics = state.get("diagnostics", {})
    post_turn_safety = None
    if isinstance(diagnostics, Mapping):
        candidate = diagnostics.get("voice_post_turn_safety")
        if isinstance(candidate, Mapping):
            post_turn_safety = dict(candidate)

    if body.client_turn_id is not None and body.outcome == "completed":
        completed_tool_names = [
            call.tool_name for call in body.tool_calls if call.status == "completed"
        ]
        failed_tool_names = [
            call.tool_name for call in body.tool_calls if call.status == "failed"
        ]
        observed_tool_names = {call.tool_name for call in body.tool_calls}
        trace_event(
            VOICE_CONCURRENT_SAFETY_TURN_OBSERVED,
            {
                "voice_runtime": "openai_realtime",
                "correlation_hash": _voice_turn_correlation_hash(
                    thread_id=body.thread_id,
                    client_turn_id=body.client_turn_id,
                ),
                "route": str(state.get("route") or ""),
                "completed_tool_names": completed_tool_names,
                "completed_tool_count": len(completed_tool_names),
                "failed_tool_names": failed_tool_names,
                "failed_tool_count": len(failed_tool_names),
                "crisis_tool_observed": bool(
                    observed_tool_names & _CRISIS_VOICE_TOOL_NAMES
                ),
            },
        )

    return VoiceTurnRecordResponse(
        recorded=True,
        thread_id=body.thread_id,
        message_count=len(state.get("transcript", []) or []),
        post_turn_safety=post_turn_safety,
    )


@router.post(
    "/realtime/safety/check",
    response_model=VoiceConcurrentSafetyResponse,
)
async def check_voice_realtime_safety(
    body: VoiceConcurrentSafetyRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceConcurrentSafetyResponse:
    """Assess one current voice turn and return the server playback policy."""

    selection = get_runtime_selection(body.memory_mode)
    result = await selection.runtime.voice.assess_voice_turn_safety(
        thread_id=body.thread_id,
        user_id=body.user_id,
        user_text=body.user_text,
        prior_message_count=body.prior_message_count,
        pending_prior_transcript=[
            entry.model_dump(mode="json") for entry in body.pending_prior_transcript
        ],
        llm_client=llm_client,
    )
    decision = selection.runtime.voice.decide_voice_safety(result)
    attributes: dict[str, object] = {
        "voice_runtime": "openai_realtime",
        "correlation_hash": _voice_turn_correlation_hash(
            thread_id=body.thread_id,
            client_turn_id=body.client_turn_id,
        ),
        "mode": "observe",
        "status": result.status,
        "reason": result.reason,
        "duration_ms": result.duration_ms,
        "memory_mode": selection.memory_mode,
    }
    if result.assessment is not None:
        attributes.update(
            {
                "level": result.assessment.level,
                "confidence": result.assessment.confidence,
                "needs_crisis_response": result.assessment.needs_crisis_response,
                "needs_clarification": result.assessment.needs_clarification,
            }
        )
    trace_event(VOICE_CONCURRENT_SAFETY_ASSESSED, attributes)
    trace_event(
        VOICE_SAFETY_INTERRUPTION_DECIDED,
        {
            "voice_runtime": "openai_realtime",
            "correlation_hash": attributes["correlation_hash"],
            "action": decision.action,
            "risk_level": decision.risk_level,
            "status": result.status,
            "memory_mode": selection.memory_mode,
        },
    )
    support = (
        {
            "headline": decision.support.headline,
            "validation": decision.support.validation,
            "immediate_step": decision.support.immediate_step,
        }
        if decision.support is not None
        else None
    )
    interruption_token = (
        _VOICE_SAFETY_PROOFS.issue(
            thread_id=body.thread_id,
            client_turn_id=body.client_turn_id,
            user_text=body.user_text,
            user_id=body.user_id,
            memory_mode=str(selection.memory_mode),
            risk_level=decision.risk_level,
        )
        if decision.action == "interrupt" and decision.risk_level is not None
        else None
    )
    return VoiceConcurrentSafetyResponse(
        client_turn_id=body.client_turn_id,
        status=result.status,
        reason=result.reason,
        action=decision.action,
        risk_level=decision.risk_level,
        support=support,
        interruption_token=interruption_token,
    )


@router.post(
    "/realtime/safety/resources",
    response_model=VoiceSafetyResourcesResponse,
)
async def resolve_voice_realtime_safety_resources(
    body: VoiceConcurrentSafetyRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceSafetyResourcesResponse:
    """Resolve verified crisis resources without mutating runtime state."""

    selection = get_runtime_selection(body.memory_mode)
    result = await selection.runtime.voice.resolve_voice_safety_resources(
        thread_id=body.thread_id,
        user_text=body.user_text,
        prior_message_count=body.prior_message_count,
        pending_prior_transcript=[
            entry.model_dump(mode="json") for entry in body.pending_prior_transcript
        ],
        llm_client=llm_client,
    )
    trace_event(
        VOICE_SAFETY_RESOURCES_RESOLVED,
        {
            "voice_runtime": "openai_realtime",
            "correlation_hash": _voice_turn_correlation_hash(
                thread_id=body.thread_id,
                client_turn_id=body.client_turn_id,
            ),
            "status": result.status,
            "resource_count": len(result.resources),
            "memory_mode": selection.memory_mode,
        },
    )
    return VoiceSafetyResourcesResponse(
        client_turn_id=body.client_turn_id,
        status=result.status,
        inferred_location="",
        resources=result.resources,
        message=result.message,
    )


@router.post("/realtime/end", response_model=VoiceEndSessionResponse)
async def end_voice_realtime_session(
    body: VoiceEndSessionRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceEndSessionResponse:
    """Finalize a voice session using the runtime session finalizer."""

    selection = get_runtime_selection(body.memory_mode)
    try:
        result = await end_runtime_session(
            runtime=selection.runtime,
            thread_id=body.thread_id,
            feedback=body.feedback,
            llm_client=llm_client,
            memory_mode=selection.memory_mode,
            modality="voice",
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=_VOICE_END_FAILURE_HTTP_STATUS,
            detail={
                "code": "voice_realtime_end_failed",
                "message": message,
            },
        ) from exc

    if not result.finalized:
        return VoiceEndSessionResponse(
            finalized=False,
            summary=None,
            detail=result.detail,
        )

    assert result.summary is not None
    return VoiceEndSessionResponse(
        finalized=True,
        summary=result.summary,
        detail=result.detail,
        themes=result.themes,
        mood_opened=result.mood_opened,
        mood_closed=result.mood_closed,
        turn_count=result.turn_count,
        open_loops=result.open_loops,
        resolved_threads=result.resolved_threads,
    )


def _voice_safety_identifier(*, thread_id: str, user_id: str | None) -> str:
    """Return a stable privacy-preserving safety identifier."""

    stable_id = (user_id or thread_id).strip()
    digest = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    return digest


def _voice_turn_correlation_hash(*, thread_id: str, client_turn_id: str) -> str:
    """Return a deterministic hash that correlates safety and final-turn events."""

    value = f"{thread_id}\0{client_turn_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _voice_turn_request_hash(body: VoiceTurnRecordRequest) -> str:
    payload = body.model_dump(mode="json", exclude={"interruption_token"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
