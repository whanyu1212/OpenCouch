"""FastAPI routes for the voice websocket test harness.

Browser protocol (browser ↔ backend):

Browser sends:
    {"type": "start", "user_id": "...", "thread_id": "...", "voice": "cedar", "transcription_language": "en"}
    {"type": "audio", "data": "<base64 PCM16 24kHz mono>"}
    {"type": "truncate", "item_id": "...", "content_index": 0, "audio_end_ms": 1234}

Backend sends:
    {"type": "ready"}
    {"type": "audio", "data": "<base64 PCM16 24kHz mono>", "item_id": "...", "content_index": 0}
    {"type": "caption", "role": "user", "text": "...", "item_id": "...", "status": "partial"}
    {"type": "caption", "role": "assistant", "text": "...", "item_id": "...", "status": "cleared"}
    {"type": "transcript", "role": "user", "text": "...", "item_id": "..."}
    {"type": "transcript", "role": "assistant", "text": "...", "item_id": "..."}
    {"type": "interrupted"}
    {"type": "truncated", "item_id": "...", "content_index": 0, "audio_end_ms": 1234}
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import os
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from voice.realtime import (
    DEFAULT_ASSISTANT_VOICE,
    DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
    RealtimeVoiceSession,
    SUPPORTED_REALTIME_VOICES,
    SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

TEST_PAGE_PATH = Path(__file__).parent / "test_page.html"


@router.get("/voice/test")
async def voice_test_page() -> HTMLResponse:
    """Serve the standalone websocket voice test page."""

    return HTMLResponse(TEST_PAGE_PATH.read_text())


@router.websocket("/voice/session")
async def voice_session(websocket: WebSocket) -> None:
    """Bridge the browser websocket to one OpenAI Realtime session."""

    await websocket.accept()

    session: RealtimeVoiceSession | None = None

    try:
        start_data = await websocket.receive_json()
        if start_data.get("type") != "start":
            await websocket.send_json(
                {"type": "error", "message": "First message must be type 'start'"}
            )
            await websocket.close()
            return

        user_id = start_data.get("user_id", "voice-user")
        thread_id = start_data.get("thread_id", "voice-default")
        voice = start_data.get("voice", DEFAULT_ASSISTANT_VOICE)
        raw_transcription_language = start_data.get(
            "transcription_language",
            DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
        )
        if voice not in SUPPORTED_REALTIME_VOICES:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Unsupported voice. Supported voices are: "
                        + ", ".join(SUPPORTED_REALTIME_VOICES)
                    ),
                }
            )
            await websocket.close()
            return

        transcription_language: str | None
        if raw_transcription_language is None:
            transcription_language = DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE
        else:
            normalized_language = str(raw_transcription_language).strip().lower()
            if normalized_language in {"", "auto"}:
                transcription_language = None
            elif normalized_language in SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES:
                transcription_language = normalized_language
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Unsupported transcription language. Supported values are: "
                            "auto, "
                            + ", ".join(SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES)
                        ),
                    }
                )
                await websocket.close()
                return

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            await websocket.send_json(
                {"type": "error", "message": "OPENAI_API_KEY not configured"}
            )
            await websocket.close()
            return

        from api.dependencies import _runtime

        if _runtime is None:
            await websocket.send_json(
                {"type": "error", "message": "Runtime not initialized"}
            )
            await websocket.close()
            return

        memory_store = _runtime.memory_store
        memory_mode = _runtime.memory_mode

        async def on_audio_delta(
            audio_bytes: bytes,
            item_id: str,
            content_index: int,
        ) -> None:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_json(
                {
                    "type": "audio",
                    "data": encoded,
                    "item_id": item_id,
                    "content_index": content_index,
                }
            )

        async def on_user_transcript(text: str, item_id: str) -> None:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "role": "user",
                    "text": text,
                    "item_id": item_id,
                }
            )

        async def on_agent_transcript(text: str, item_id: str) -> None:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "role": "assistant",
                    "text": text,
                    "item_id": item_id,
                }
            )

        async def on_caption(
            role: str,
            text: str,
            item_id: str,
            status: str,
        ) -> None:
            await websocket.send_json(
                {
                    "type": "caption",
                    "role": role,
                    "text": text,
                    "item_id": item_id,
                    "status": status,
                }
            )

        async def on_interrupted() -> None:
            await websocket.send_json({"type": "interrupted"})

        async def on_truncated(
            item_id: str,
            content_index: int,
            audio_end_ms: int,
        ) -> None:
            await websocket.send_json(
                {
                    "type": "truncated",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )

        async def on_error(message: str) -> None:
            await websocket.send_json({"type": "error", "message": message})

        session = RealtimeVoiceSession(
            openai_api_key=openai_api_key,
            memory_store=memory_store,
            memory_mode=memory_mode,
            user_id=user_id,
            thread_id=thread_id,
            voice=voice,
            transcription_language=transcription_language,
            on_audio_delta=on_audio_delta,
            on_transcript=on_user_transcript,
            on_agent_transcript=on_agent_transcript,
            on_caption=on_caption,
            on_interrupted=on_interrupted,
            on_truncated=on_truncated,
            on_error=on_error,
        )
        await session.start()
        await websocket.send_json({"type": "ready"})

        logger.info(
            "voice session: started for user=%s thread=%s voice=%s transcription_language=%s",
            user_id,
            thread_id,
            voice,
            transcription_language or "auto",
        )

        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "audio":
                try:
                    audio_bytes = base64.b64decode(message.get("data", ""))
                except (TypeError, ValueError, binascii.Error):
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid base64 audio payload"}
                    )
                    continue

                await session.send_audio(audio_bytes)

            elif message_type == "truncate":
                await session.truncate_assistant_audio(
                    item_id=message.get("item_id", ""),
                    content_index=message.get("content_index", 0),
                    audio_end_ms=message.get("audio_end_ms", 0),
                )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unsupported client message type: {message_type!r}",
                    }
                )

    except WebSocketDisconnect:
        logger.info("voice session: client disconnected")
    except Exception:
        logger.exception("voice session: error")
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "message": "Internal server error"}
            )
    finally:
        if session is not None:
            await session.end_session()
            await session.close()
