"""Synthetic-only screening evidence and server risk snapshots for offline replay."""

import asyncio
import json
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage
from typesafe_sdk import Noul

from opencouch_agent.safety.jev import FailureReason, JevClient, failure_reason
from opencouch_agent.safety.risk import (
    DEFAULT_RISK_CONFIG,
    CheckInAnswer,
    RiskConfig,
    RiskState,
)


class ScreeningCall(BaseModel):
    """Exact synthetic input and question definitions for one Jev call."""

    source: Literal["scripted", "live", "fake"]
    state: str
    questions: str


class ScreeningSuccess(ScreeningCall):
    """Raw probabilities, including invalid responses rejected by screening."""

    outcome: Literal["success"] = "success"
    probabilities: dict[str, float]


class ScreeningError(ScreeningCall):
    """Safe error classification; provider payloads are never persisted."""

    outcome: Literal["error"] = "error"
    failure: FailureReason


ScreeningRecord = Annotated[
    ScreeningSuccess | ScreeningError, Field(discriminator="outcome")
]


class RiskTrace(BaseModel):
    """Inputs and persisted outcome of the real gate, including missed-turn rechecks."""

    at: datetime
    before: RiskState
    after: RiskState
    history: list[ModelMessage]
    submitted_check_in: CheckInAnswer | None
    calls: list[ScreeningRecord]
    config: RiskConfig = DEFAULT_RISK_CONFIG


def question_definitions(questions: Mapping[str, Noul]) -> str:
    """Canonical question definitions to detect changes during replay."""
    return json.dumps(
        {
            name: question.model_dump(mode="json")
            for name, question in questions.items()
        },
        sort_keys=True,
    )


class RecordingJev:
    """Time screening and retain every call without changing production decisions."""

    def __init__(self, inner: JevClient | None) -> None:
        self.inner = inner
        self.override: Mapping[str, float] | None = None
        self.last_ms = 0
        self.calls: list[ScreeningRecord] = []

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Record readings without suppressing failures or cancellation."""
        source: Literal["scripted", "live", "fake"] = (
            "scripted"
            if self.override is not None
            else "live"
            if self.inner
            else "fake"
        )
        call = ScreeningCall(
            source=source, state=state, questions=question_definitions(questions)
        )
        start = time.perf_counter()
        try:
            if self.override is not None:
                probabilities = {
                    name: self.override.get(name, 0.0) for name in questions
                }
            elif self.inner is not None:
                probabilities = await self.inner.ask(state, questions)
            else:
                probabilities = dict.fromkeys(questions, 0.0)
        except asyncio.CancelledError:
            # A completed turn can retain this only when screening's deadline fired.
            # Cancelling the whole eval still propagates and produces no result row.
            self.calls.append(
                ScreeningError(**call.model_dump(), failure=FailureReason.TIMEOUT)
            )
            raise
        except Exception as error:
            self.calls.append(
                ScreeningError(**call.model_dump(), failure=failure_reason(error))
            )
            raise
        else:
            self.calls.append(
                ScreeningSuccess(**call.model_dump(), probabilities=probabilities)
            )
            return probabilities
        finally:
            self.last_ms = round((time.perf_counter() - start) * 1000)
