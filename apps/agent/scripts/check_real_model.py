"""Opt-in check of the agent against the real model (HY-202).

Calls the real OpenAI API, so it costs money (well under $1 per run) and is not
part of the default test run. Run from `apps/agent` with the key in the
environment:

    uv run --env-file ../../.env python scripts/check_real_model.py

It checks five things and prints a report:

1. A multi-turn chat through the real AG-UI endpoint, with server-owned
   history: time to first token per turn, and a tool call making the full
   round trip through AG-UI events.
2. Handing that history, tool calls included, to a second agent with no tools
   (the planned crisis responder, HY-222 / HY-230).
3. `UsageLimits` stopping a run that needs more model calls than allowed.
4. Reasoning summaries on the OpenAI Responses model.
5. Strict structured output on the OpenAI Responses model.
"""

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import httpx
import uvicorn
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import ModelMessage, ThinkingPart
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from opencouch_agent.agent import CompanionDeps
from opencouch_agent.agent.companion import INSTRUCTIONS
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.sessions import InMemorySessionStore

MODEL = "openai:gpt-6.1-sol"
PORT = 8765
THREAD_ID = "hy-202-check"

TOOL_INSTRUCTIONS = "If the user asks what time it is, call `get_local_time`."

TURNS = [
    "Work has been really stressful this week and I can't switch off at night.",
    "Mostly I replay conversations with my manager and think about what I "
    "should have said.",
    "What time is it right now? I've lost track.",
    "Yeah, it's late. I always end up scrolling instead of sleeping.",
    "I think part of it is that I don't feel good enough at this job.",
]

RESPONDER_INSTRUCTIONS = """\
You are a calm, supportive responder. The user may be going through something
difficult. Acknowledge what they said and gently ask how they are doing now.
"""


def build_check_agent() -> Agent[CompanionDeps, str]:
    """Build the companion agent with one trivial tool, on the real model."""
    agent = Agent(
        MODEL,
        deps_type=CompanionDeps,
        instructions=[INSTRUCTIONS, TOOL_INSTRUCTIONS],
        name="companion",
    )

    @agent.tool_plain
    def get_local_time() -> str:
        """Return the user's local time."""
        return datetime.now().strftime("%H:%M")

    return agent


@dataclass
class TurnResult:
    """What one AG-UI turn produced."""

    message: str
    status: int
    first_token_seconds: float | None = None
    total_seconds: float = 0.0
    reply: str = ""
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    event_counts: Counter[str] = field(default_factory=Counter)


async def send_turn(client: httpx.AsyncClient, index: int, message: str) -> TurnResult:
    """Send one user message over AG-UI and time the streamed reply."""
    body = {
        "threadId": THREAD_ID,
        "runId": f"run-{index}",
        "state": {},
        # The server owns history, so only the newest message is sent.
        "messages": [{"id": f"msg-{index}", "role": "user", "content": message}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    start = time.perf_counter()
    async with client.stream(
        "POST", "/api/agent", json=body, headers={"accept": "text/event-stream"}
    ) as response:
        result = TurnResult(message=message, status=response.status_code)
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                record_event(result, json.loads(line.removeprefix("data: ")), start)
    result.total_seconds = time.perf_counter() - start
    return result


def record_event(result: TurnResult, event: dict[str, Any], start: float) -> None:
    """Fold one AG-UI event into the turn result."""
    kind = event["type"]
    result.event_counts[kind] += 1
    if kind == "TEXT_MESSAGE_CONTENT":
        if result.first_token_seconds is None:
            result.first_token_seconds = time.perf_counter() - start
        result.reply += event["delta"]
    elif kind == "TOOL_CALL_START":
        result.tool_calls.append(event["toolCallName"])
    elif kind == "RUN_ERROR":
        result.errors.append(event.get("message", ""))


async def run_conversation(store: InMemorySessionStore) -> list[TurnResult]:
    """Serve the app on a local port and play the scripted turns through it."""
    app = create_app(settings=Settings(), agent=build_check_agent(), store=store)
    server = uvicorn.Server(uvicorn.Config(app, port=PORT, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    # uvicorn exposes no startup event, only this flag.
    while not server.started:
        if serving.done():
            # Startup failed (e.g. the port is taken); surface why.
            serving.result()
            raise RuntimeError(f"Server failed to start on port {PORT}")
        await asyncio.sleep(0.05)
    try:
        base_url = f"http://127.0.0.1:{PORT}"
        async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
            return [
                await send_turn(client, index, message)
                for index, message in enumerate(TURNS)
            ]
    finally:
        server.should_exit = True
        await serving


def describe_history(history: list[ModelMessage]) -> dict[str, int]:
    """Count message part kinds in a stored history."""
    return dict(Counter(part.part_kind for m in history for part in m.parts))


async def check_toolless_history(history: list[ModelMessage]) -> dict[str, Any]:
    """Run an agent with no tools on history that contains tool calls."""
    responder = Agent(MODEL, instructions=RESPONDER_INSTRUCTIONS, name="responder")
    try:
        result = await responder.run(
            "Thanks. Honestly I've been feeling really low lately.",
            message_history=history,
        )
    # Any failure is the finding here, so report it instead of raising.
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"ok": True, "reply": result.output, "requests": result.usage.requests}


async def check_usage_limit() -> dict[str, Any]:
    """Ask for a tool call with a one-request cap; it needs two requests."""
    try:
        result = await build_check_agent().run(
            "What time is it?",
            deps=CompanionDeps(),
            usage_limits=UsageLimits(request_limit=1),
        )
    except UsageLimitExceeded as error:
        return {"stopped": True, "error": str(error)}
    return {"stopped": False, "output": result.output}


async def check_reasoning_summary() -> dict[str, Any]:
    """Request reasoning summaries and see whether any come back."""
    agent = Agent(MODEL, instructions=INSTRUCTIONS)
    settings = OpenAIResponsesModelSettings(openai_reasoning_summary="auto")
    result = await agent.run(TURNS[4], model_settings=settings)
    summaries = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ThinkingPart) and part.content
    ]
    first = summaries[0][:200] if summaries else None
    return {"summaries": len(summaries), "first": first}


class MoodCheck(BaseModel):
    """Structured output used to test strict schemas."""

    mood: Literal["calm", "stressed", "low", "angry", "unsure"]
    intensity: int = Field(ge=1, le=5)
    needs_follow_up: bool


async def check_structured_output() -> dict[str, Any]:
    """Ask for strict structured output and validate it."""
    agent = Agent(MODEL, output_type=NativeOutput(MoodCheck))
    try:
        result = await agent.run(f"Classify this message: {TURNS[4]!r}")
    # Any failure is the finding here, so report it instead of raising.
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"ok": True, "output": result.output.model_dump()}


def print_conversation(turns: list[TurnResult]) -> None:
    """Print the per-turn timings and outcomes."""
    print("\n== 1. Conversation over AG-UI ==")
    for index, turn in enumerate(turns):
        first = f"{turn.first_token_seconds:.2f}s" if turn.first_token_seconds else "-"
        print(
            f"turn {index}: HTTP {turn.status}, first token {first}, "
            f"total {turn.total_seconds:.2f}s, tools {turn.tool_calls or '-'}, "
            f"errors {turn.errors or '-'}"
        )
        print(f"   reply: {turn.reply[:160]!r}")
    print("events:", dict(sum((t.event_counts for t in turns), Counter())))


async def main() -> None:
    """Run every check and print a report."""
    store = InMemorySessionStore()
    turns = await run_conversation(store)
    print_conversation(turns)
    history = (await store.load(THREAD_ID)).history
    print("stored history parts:", describe_history(history))

    print("\n== 2. Tool-less agent on history with tool calls ==")
    print(await check_toolless_history(history))
    print("\n== 3. UsageLimits ==")
    print(await check_usage_limit())
    print("\n== 4. Reasoning summaries ==")
    print(await check_reasoning_summary())
    print("\n== 5. Strict structured output ==")
    print(await check_structured_output())


if __name__ == "__main__":
    asyncio.run(main())
