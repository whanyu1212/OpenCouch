"""FastAPI routes for LiveKit voice sessions.

Provides a token endpoint that the frontend calls before connecting
to a LiveKit room.  The token is a signed JWT that grants the
browser participant permission to join a specific room and
publish/subscribe audio.

The agent is dispatched via ``room_config`` embedded in the token.
When the room is first created, LiveKit reads the config and
dispatches the ``opencouch-voice`` agent automatically.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from fastapi import APIRouter, HTTPException
from livekit.api import (
    AccessToken,
    RoomAgentDispatch,
    RoomConfiguration,
    VideoGrants,
)
from pydantic import BaseModel

from config import load_runtime_env
from agent.voice.config import normalize_assistant_voice
from agent.voice.finalization_status import get_voice_finalization_status
from agent.voice.session_data import parse_voice_memory_mode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice/livekit", tags=["voice-livekit"])


class TokenRequest(BaseModel):
    """Request body for the token endpoint."""

    user_id: str | None = None
    thread_id: str | None = None
    transcription_language: str | None = None
    assistant_voice: str | None = None
    memory_mode: str | None = None
    room_name: str | None = None
    dispatch_agent: bool = True


class TokenResponse(BaseModel):
    """Standardized LiveKit token endpoint response.

    Field names follow the LiveKit convention so client SDKs
    can consume this directly via an endpoint TokenSource.
    """

    server_url: str
    participant_token: str
    room_name: str
    identity: str
    memory_mode: str
    assistant_voice: str


class VoiceFinalizationStatusResponse(BaseModel):
    """Current disconnect-time memory finalization state for one thread."""

    status: str
    detail: str | None = None
    updated_at: str


@router.get(
    "/finalization-status/{thread_id}",
    response_model=VoiceFinalizationStatusResponse,
)
async def get_livekit_finalization_status(
    thread_id: str,
) -> VoiceFinalizationStatusResponse:
    """Return the current voice finalization status for one thread."""

    status = await get_voice_finalization_status(thread_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No LiveKit finalization status for thread {thread_id}",
        )

    return VoiceFinalizationStatusResponse(
        status=status.status,
        detail=status.detail,
        updated_at=status.updated_at,
    )


@router.post("/token", response_model=TokenResponse, status_code=201)
async def create_voice_token(body: TokenRequest) -> TokenResponse:
    """Generate a LiveKit room token for a browser voice session.

    Creates a room name from the thread_id (or generates one),
    signs a JWT granting the participant permission to join with
    microphone-only publishing, embeds agent dispatch config, and
    returns the token + LiveKit server URL.
    """
    load_runtime_env()

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    livekit_url = os.getenv("LIVEKIT_URL")

    if not api_key or not api_secret or not livekit_url:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET.",
        )

    user_id = body.user_id or "voice-user"
    thread_id = body.thread_id or f"voice-{uuid.uuid4().hex[:12]}"
    transcription_language = body.transcription_language
    if transcription_language is not None:
        transcription_language = transcription_language.strip()
    memory_mode = parse_voice_memory_mode(
        body.memory_mode,
        default=parse_voice_memory_mode(os.getenv("OPENCOUCH_MEMORY_MODE")),
    )
    assistant_voice = normalize_assistant_voice(
        body.assistant_voice,
        default="marin",
    )

    # Room name ties the browser participant and agent together.
    room_name = body.room_name or f"opencouch-{thread_id}"

    # Participant identity must be unique within the room.
    identity = f"user-{user_id}-{uuid.uuid4().hex[:8]}"

    # Metadata passed to the agent entrypoint via participant.metadata.
    metadata = json.dumps(
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "transcription_language": transcription_language,
            "assistant_voice": assistant_voice,
            "memory_mode": memory_mode.value,
        }
    )

    token = (
        AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity(identity)
        .with_name(user_id)
        .with_metadata(metadata)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                can_publish_sources=["microphone"],
            )
        )
    )

    if body.dispatch_agent:
        token = token.with_room_config(
            RoomConfiguration(
                agents=[
                    RoomAgentDispatch(
                        agent_name="opencouch-voice",
                        metadata=metadata,
                    )
                ],
            )
        )

    jwt = token.to_jwt()

    logger.info(
        "livekit token: room=%s identity=%s user=%s thread=%s transcription_language=%s assistant_voice=%s memory_mode=%s",
        room_name,
        identity,
        user_id,
        thread_id,
        transcription_language if transcription_language is not None else "default",
        assistant_voice,
        memory_mode.value,
    )

    return TokenResponse(
        server_url=livekit_url,
        participant_token=jwt,
        room_name=room_name,
        identity=identity,
        memory_mode=memory_mode.value,
        assistant_voice=assistant_voice,
    )
