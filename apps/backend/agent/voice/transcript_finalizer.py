"""Idempotent transcript finalization for LiveKit voice sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from livekit.agents import AgentSession, ChatContext

from agent.memory.modes import MemoryMode
from agent.persistence import PersistentAgentRuntime
from agent.voice.finalization_status import set_voice_finalization_status
from agent.voice.session_data import SessionData
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

VoiceFinalizationState = Literal["in_progress", "completed", "failed"]


def serialize_session_history(chat_ctx: ChatContext) -> list[dict[str, str]]:
    """Convert LiveKit session history into the persisted transcript format."""

    transcript: list[dict[str, str]] = []
    for item in chat_ctx.items:
        if item.type != "message":
            continue

        role = getattr(item.role, "value", item.role)
        if role not in {"user", "assistant"}:
            continue

        text = (item.text_content or "").strip()
        if text:
            transcript.append({"role": role, "content": text})

    return transcript


class VoiceFinalizationService:
    """Finalize one voice transcript at most once."""

    def __init__(
        self,
        *,
        runtime: PersistentAgentRuntime,
        llm_client: BaseLLMClient,
        enabled: bool,
    ) -> None:
        self._runtime = runtime
        self._llm_client = llm_client
        self._enabled = enabled
        self._lock = asyncio.Lock()
        self._started = False

    async def _set_status(
        self,
        thread_id: str,
        *,
        status: VoiceFinalizationState,
        detail: str | None,
    ) -> None:
        try:
            await set_voice_finalization_status(
                thread_id,
                status=status,
                detail=detail,
            )
        except Exception:
            logger.warning(
                "livekit session: failed to persist disconnect finalization status",
                extra={"thread_id": thread_id, "status": status},
                exc_info=True,
            )

    async def finalize(
        self,
        session: AgentSession[SessionData],
        *,
        trigger: str,
    ) -> None:
        """Persist transcript-derived memory for a completed voice session."""

        async with self._lock:
            if self._started:
                return
            self._started = True

        if not self._enabled:
            logger.info("livekit session: skipping transcript finalization")
            return

        userdata = session.userdata
        await self._set_status(
            userdata.thread_id,
            status="in_progress",
            detail="Saving session memory.",
        )

        if userdata.memory_mode == MemoryMode.INCOGNITO:
            await self._set_status(
                userdata.thread_id,
                status="completed",
                detail="Incognito session; no memory saved.",
            )
            return

        try:
            transcript = serialize_session_history(session.history)
            logger.info(
                "livekit session: finalization trigger=%s user=%s thread=%s transcript_turns=%d",
                trigger,
                userdata.user_id,
                userdata.thread_id,
                len(transcript),
            )
            if not transcript:
                await self._set_status(
                    userdata.thread_id,
                    status="completed",
                    detail="No transcript memory to save.",
                )
                return

            await self._runtime.end_transcript_session(
                thread_id=userdata.thread_id,
                user_id=userdata.user_id,
                transcript=transcript,
                llm_client=self._llm_client,
                started_at=userdata.started_at,
                crisis_level_max=userdata.max_crisis_level,
            )
            await self._set_status(
                userdata.thread_id,
                status="completed",
                detail="Session memory saved.",
            )
            logger.info(
                "livekit session: transcript saved thread=%s turns=%d",
                userdata.thread_id,
                len(transcript),
            )
        except Exception:
            await self._set_status(
                userdata.thread_id,
                status="failed",
                detail="Session memory save failed.",
            )
            logger.warning("livekit session: failed to save transcript", exc_info=True)
