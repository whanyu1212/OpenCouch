from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# Normalizes the surface the message came from so routing can stay provider-agnostic.
class Channel(str, Enum):
    TEST = "test"
    WEB = "web"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    VOICE = "voice"


# Distinguishes conversation turns when building history for the model.
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Keeps the public agent result explicit about whether the reply is normal or safety-driven.
class ResponseKind(str, Enum):
    THERAPEUTIC = "therapeutic"
    CRISIS = "crisis"


# Represents one prior conversation turn in a validated, serializable form.
class Message(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


# Carries the safety decision separately from the reply so crisis handling is inspectable.
class CrisisAssessment(BaseModel):
    level: Literal[0, 1, 2, 3] = 0
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""
    needs_crisis_response: bool = False
    needs_clarification: bool = False


# Defines the minimal external contract needed to run the MVP agent kernel.
class AgentInput(BaseModel):
    message: str = Field(min_length=1)
    channel: Channel = Channel.TEST
    user_id: str | None = None
    session_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    installed_skills: list[str] = Field(default_factory=list)


# Defines the normalized result so API, CLI, and tests can all consume the same shape.
class AgentOutput(BaseModel):
    response_text: str
    response_kind: ResponseKind
    crisis: CrisisAssessment
    should_persist_memory: bool = False
