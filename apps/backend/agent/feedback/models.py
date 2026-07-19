"""User feedback record models.

These models describe explicit user quality signals captured by trusted
session-ending surfaces. They are not therapeutic memory and should not be
loaded into prompt context by ordinary memory recall paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

FeedbackLabel = Literal["positive", "negative", "skip"]
FeedbackSource = Literal["cli_end", "cli_exit", "api_end"]
FeedbackModality = Literal["text", "voice"]


class SessionFeedbackRecord(BaseModel):
    """One end-of-session feedback record."""

    id: str
    session_id_opaque: str
    user_id_or_null: str | None = None
    recorded_at: str
    label: FeedbackLabel
    turn_count_at_end: int
    source: FeedbackSource
    modality: FeedbackModality = "text"
    schema_version: int = 1


__all__ = [
    "FeedbackLabel",
    "FeedbackSource",
    "FeedbackModality",
    "SessionFeedbackRecord",
]
