"""Voice endpoints for OpenAI Realtime speech-to-speech sessions."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException

from agent.voice import realtime
from agent.voice import tools as voice_tools
from agent.observability.decorators import trace_event
from agent.observability.events import (
    VOICE_CONCURRENT_SAFETY_ASSESSED,
    VOICE_CONCURRENT_SAFETY_TURN_OBSERVED,
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
_VOICE_SESSION_WATERMARK_TIMEOUT_SECONDS = 0.5

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
        output = await voice_tools.execute_voice_tool_call(
            runtime=selection.runtime,
            tool_name=body.tool_name,
            arguments=body.arguments,
            thread_id=body.thread_id,
            user_id=body.user_id,
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
    try:
        state = await selection.runtime.voice.record_voice_turn(
            thread_id=body.thread_id,
            user_id=body.user_id,
            user_text=body.user_text,
            assistant_text=body.assistant_text,
            route=body.route,
            response_style=body.response_style,
            tool_calls=[call.model_dump(mode="json") for call in body.tool_calls],
            llm_client=llm_client,
        )
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

    if body.client_turn_id is not None:
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
    """Observe one current voice turn without affecting response playback."""

    selection = get_runtime_selection(body.memory_mode)
    result = await selection.runtime.voice.assess_voice_turn_safety(
        thread_id=body.thread_id,
        user_id=body.user_id,
        user_text=body.user_text,
        prior_message_count=body.prior_message_count,
        pending_prior_transcript=body.pending_prior_transcript,
        llm_client=llm_client,
    )
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
    return VoiceConcurrentSafetyResponse(
        client_turn_id=body.client_turn_id,
        status=result.status,
        reason=result.reason,
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
