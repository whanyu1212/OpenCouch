"""FastAPI WebSocket endpoint for voice chat (Option B).

Bridges the browser's microphone/speaker to an OpenAI Realtime API
session. The browser sends raw PCM16 audio chunks over WebSocket;
this endpoint forwards them to Realtime and streams Realtime's audio
response back.

Protocol (browser ↔ server):

    Browser sends:
        {"type": "audio", "data": "<base64 PCM16 24kHz>"}
        {"type": "start", "user_id": "...", "thread_id": "..."}

    Server sends:
        {"type": "audio", "data": "<base64 PCM16 24kHz>"}
        {"type": "transcript", "role": "user", "text": "..."}
        {"type": "transcript", "role": "assistant", "text": "..."}
        {"type": "error", "message": "..."}

The ``start`` message must be sent first to initialize the Realtime
session with the user's memory context. Subsequent ``audio`` messages
stream microphone data to Realtime.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from voice.realtime import RealtimeVoiceSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

TEST_PAGE_PATH = Path(__file__).parent / "test_page.html"


@router.get("/voice/test")
async def voice_test_page() -> HTMLResponse:
    """Serve the minimal voice test page."""
    return HTMLResponse(TEST_PAGE_PATH.read_text())


@router.websocket("/voice/session")
async def voice_session(websocket: WebSocket) -> None:
    """WebSocket endpoint for a voice conversation.

    The client connects, sends a ``start`` message with user/thread
    identifiers, then streams audio. The server streams audio back
    from the Realtime API and sends transcript events for the UI.
    """

    await websocket.accept()

    session: RealtimeVoiceSession | None = None

    try:
        # Wait for the start message
        start_data = await websocket.receive_json()
        if start_data.get("type") != "start":
            await websocket.send_json(
                {"type": "error", "message": "First message must be type 'start'"}
            )
            await websocket.close()
            return

        user_id = start_data.get("user_id", "voice-user")
        thread_id = start_data.get("thread_id", "voice-default")
        voice = start_data.get("voice", "sage")

        # Resolve API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            await websocket.send_json(
                {"type": "error", "message": "OPENAI_API_KEY not configured"}
            )
            await websocket.close()
            return

        # Get the runtime's memory store
        from api.dependencies import _runtime

        if _runtime is None:
            await websocket.send_json(
                {"type": "error", "message": "Runtime not initialized"}
            )
            await websocket.close()
            return

        memory_store = _runtime.memory_store
        memory_mode = _runtime.memory_mode

        # Callbacks that forward events to the browser WebSocket
        async def on_audio_delta(audio_bytes: bytes) -> None:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_json({"type": "audio", "data": encoded})

        async def on_transcript(text: str) -> None:
            await websocket.send_json(
                {"type": "transcript", "role": "user", "text": text}
            )

        async def on_agent_transcript(text: str) -> None:
            await websocket.send_json(
                {"type": "transcript", "role": "assistant", "text": text}
            )

        # Get LLM client and embedding provider for extractors
        from api.dependencies import _llm_client

        llm_client = _llm_client
        embedding_provider = getattr(_runtime, "_embedding_provider", None)

        # Create and start the Realtime session
        session = RealtimeVoiceSession(
            openai_api_key=openai_api_key,
            memory_store=memory_store,
            memory_mode=memory_mode,
            user_id=user_id,
            thread_id=thread_id,
            voice=voice,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
            on_audio_delta=on_audio_delta,
            on_transcript=on_transcript,
            on_agent_transcript=on_agent_transcript,
        )
        await session.start()

        logger.info(
            "voice session: started for user=%s thread=%s",
            user_id,
            thread_id,
        )

        # Stream audio from browser to Realtime
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                audio_bytes = base64.b64decode(data["data"])
                await session.send_audio(audio_bytes)

    except WebSocketDisconnect:
        logger.info("voice session: client disconnected")
    except Exception:
        logger.exception("voice session: error")
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal server error"}
            )
        except Exception:
            pass
    finally:
        if session is not None:
            await session.end_session()  # summarize transcript → episodic arc
            await session.close()
