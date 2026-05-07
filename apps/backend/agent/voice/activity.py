"""Sanitized frontend activity events for LiveKit voice sessions."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from agent.memory.hashing import iso_now

logger = logging.getLogger(__name__)

VOICE_ACTIVITY_TOPIC = "opencouch.voice_activity"

VoiceActivityName = Literal[
    "memory_saved",
    "memory_recall_updated",
    "memory_delete_pending",
    "memory_deleted",
    "factual_lookup",
    "crisis_resources_lookup",
    "exercise",
]
VoiceActivityStatus = Literal[
    "started",
    "completed",
    "failed",
    "pending",
    "cancelled",
]


async def emit_voice_activity(
    context: Any,
    *,
    activity: VoiceActivityName,
    status: VoiceActivityStatus,
    label: str,
    detail: str = "",
) -> None:
    """Publish a sanitized activity event to the browser voice client.

    Args:
        context: LiveKit run context or object with a ``session`` attribute.
        activity: Stable activity category for frontend rendering.
        status: Lifecycle status for the activity.
        label: Short user-facing label.
        detail: Optional sanitized user-facing detail.

    Returns:
        None: Failures are logged and do not affect the conversation.
    """

    session = getattr(context, "session", None)
    if session is None:
        return

    try:
        room = session.room_io.room
    except RuntimeError:
        return
    except AttributeError:
        room = getattr(session, "room", None)

    participant = getattr(room, "local_participant", None)
    if participant is None:
        return

    payload = {
        "type": "voice_activity",
        "activity": activity,
        "status": status,
        "label": label,
        "detail": detail,
        "timestamp": iso_now(),
    }

    try:
        await participant.publish_data(
            json.dumps(payload),
            reliable=True,
            topic=VOICE_ACTIVITY_TOPIC,
        )
    except Exception:
        logger.warning(
            "Failed to publish voice activity event activity=%s status=%s",
            activity,
            status,
            exc_info=True,
        )
