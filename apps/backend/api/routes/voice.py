"""Voice endpoints for OpenAI Realtime speech-to-speech sessions."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException

from agent.memory.modes import resolve_effective_memory_mode
from agent.voice import realtime
from agent.voice import tools as voice_tools
from api.dependencies import get_llm_client, get_runtime
from api.models import (
    VoiceRealtimeSessionRequest,
    VoiceRealtimeSessionResponse,
    VoiceToolCallRequest,
    VoiceToolCallResponse,
    VoiceTurnPolicyRequest,
    VoiceTurnPolicyResponse,
    VoiceTurnRecordRequest,
    VoiceTurnRecordResponse,
    VoiceEndSessionRequest,
    VoiceEndSessionResponse,
)
from llm.base import BaseLLMClient

router = APIRouter(prefix="/voice", tags=["voice"])

_VOICE_SESSION_FAILURE_HTTP_STATUS = 500
_VOICE_TOOL_FAILURE_HTTP_STATUS = 500
_VOICE_POLICY_FAILURE_HTTP_STATUS = 500
_VOICE_TURN_FAILURE_HTTP_STATUS = 500
_VOICE_END_FAILURE_HTTP_STATUS = 500


@router.post("/realtime/session", response_model=VoiceRealtimeSessionResponse)
async def create_voice_realtime_session(
    body: VoiceRealtimeSessionRequest,
    runtime=Depends(get_runtime),
) -> VoiceRealtimeSessionResponse:
    """Create an ephemeral OpenAI Realtime client secret for browser voice."""

    effective_mode = resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), body.memory_mode
    )
    memory_context = await runtime.voice_session_memory_context(
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=effective_mode,
    )
    session_config = realtime.build_realtime_session_config(
        thread_id=body.thread_id,
        user_id=body.user_id,
        memory_mode=effective_mode,
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
        memory_mode=effective_mode,
        session_config=session_config,
    )


@router.post("/realtime/tools", response_model=VoiceToolCallResponse)
async def execute_voice_realtime_tool(
    body: VoiceToolCallRequest,
    runtime=Depends(get_runtime),
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceToolCallResponse:
    """Execute one app-owned OpenAI Realtime function tool call."""

    effective_mode = resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), body.memory_mode
    )
    try:
        output = await voice_tools.execute_voice_tool_call(
            runtime=runtime,
            tool_name=body.tool_name,
            arguments=body.arguments,
            thread_id=body.thread_id,
            user_id=body.user_id,
            current_user_message=body.current_user_message,
            transcript=body.transcript,
            memory_mode=effective_mode,
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


@router.post("/realtime/turn-policy", response_model=VoiceTurnPolicyResponse)
async def prepare_voice_realtime_turn_policy(
    body: VoiceTurnPolicyRequest,
    runtime=Depends(get_runtime),
) -> VoiceTurnPolicyResponse:
    """Return observe-only app policy for a finalized voice user transcript."""

    effective_mode = resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), body.memory_mode
    )
    try:
        policy = await runtime.prepare_voice_turn_policy(
            thread_id=body.thread_id,
            user_id=body.user_id,
            user_text=body.user_text,
            memory_mode=effective_mode,
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=_VOICE_POLICY_FAILURE_HTTP_STATUS,
            detail={
                "code": "voice_realtime_turn_policy_failed",
                "message": message,
            },
        ) from exc

    return VoiceTurnPolicyResponse(**policy.model_dump(mode="json"))


@router.post("/realtime/turn", response_model=VoiceTurnRecordResponse)
async def record_voice_realtime_turn(
    body: VoiceTurnRecordRequest,
    runtime=Depends(get_runtime),
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceTurnRecordResponse:
    """Record a finalized voice user/assistant turn in app-owned history."""

    effective_mode = resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), body.memory_mode
    )
    if effective_mode == "incognito":
        return VoiceTurnRecordResponse(
            recorded=False,
            thread_id=body.thread_id,
            message_count=0,
        )

    try:
        state = await runtime.record_voice_turn(
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
    runtime=Depends(get_runtime),
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> VoiceEndSessionResponse:
    """Finalize a voice session using the runtime session finalizer."""

    effective_mode = resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), body.memory_mode
    )
    if effective_mode == "incognito":
        return VoiceEndSessionResponse(
            finalized=False,
            summary=None,
            detail="Incognito voice session ended without durable finalization.",
        )

    try:
        arc = await runtime.end_session(body.thread_id, llm_client=llm_client)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=_VOICE_END_FAILURE_HTTP_STATUS,
            detail={
                "code": "voice_realtime_end_failed",
                "message": message,
            },
        ) from exc

    if arc is None:
        return VoiceEndSessionResponse(
            finalized=False,
            summary=None,
            detail="No summary produced (session too short, no LLM, or incognito mode).",
        )

    return VoiceEndSessionResponse(
        finalized=True,
        summary=arc.summary,
        detail="Session summary produced.",
    )


def _voice_safety_identifier(*, thread_id: str, user_id: str | None) -> str:
    """Return a stable privacy-preserving safety identifier."""

    stable_id = (user_id or thread_id).strip()
    digest = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    return digest
