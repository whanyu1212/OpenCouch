"""Voice endpoints for OpenAI Realtime speech-to-speech sessions."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException

from agent.voice import realtime
from agent.voice import tools as voice_tools
from api.dependencies import (
    get_llm_client,
    get_runtime_selection,
)
from api.session_end import end_runtime_session
from api.models import (
    VoiceRealtimeSessionRequest,
    VoiceRealtimeSessionResponse,
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


@router.post("/realtime/session", response_model=VoiceRealtimeSessionResponse)
async def create_voice_realtime_session(
    body: VoiceRealtimeSessionRequest,
) -> VoiceRealtimeSessionResponse:
    """Create an ephemeral OpenAI Realtime client secret for browser voice."""

    selection = get_runtime_selection(body.memory_mode)
    memory_context = await selection.runtime.voice_session_memory_context(
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=selection.memory_mode,
    )
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
        state = await selection.runtime.record_voice_turn(
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

    return VoiceTurnRecordResponse(
        recorded=True,
        thread_id=body.thread_id,
        message_count=len(state.get("transcript", []) or []),
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
