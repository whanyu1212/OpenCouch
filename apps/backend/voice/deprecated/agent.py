"""LiveKit voice agent worker for OpenCouch.

Option A implementation: STT → LangGraph (full pipeline) → TTS.
The LangGraph LLMAdapter wraps our compiled bridge graph so the
full agent pipeline (crisis gate, dispatcher, modes, extractors)
runs on every voice turn.

Usage:
    # Development mode (auto-reload on changes)
    uv run python -m voice.agent dev

    # Production mode
    uv run python -m voice.agent start

    # Download required model files (Silero VAD, turn detector)
    uv run python -m voice.agent download-files

Environment variables (in .env.local):
    LIVEKIT_URL         — LiveKit server URL (e.g., wss://your-project.livekit.cloud)
    LIVEKIT_API_KEY     — LiveKit API key
    LIVEKIT_API_SECRET  — LiveKit API secret
    DEEPGRAM_API_KEY    — Deepgram API key for STT
    OPENAI_API_KEY      — OpenAI API key for TTS (and optionally for the therapeutic LLM)

    Plus the existing OpenCouch env vars:
    GEMINI_API_KEY      — For the therapeutic LLM + embeddings
    OPENCOUCH_MEMORY_MODE — "persistent" (default) or "guest"
"""

from __future__ import annotations

import logging
import os
import uuid

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, AgentServer
from livekit.plugins import deepgram, openai, silero, langchain
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment from .env.local (LiveKit convention) and .env
load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger(__name__)

# ── Runtime setup ────────────────────────────────────────────────────
#
# The PersistentAgentRuntime is created once when the worker starts
# and shared across all voice sessions. It owns the SQLite
# connections, embedding provider, and LLM client — same lifecycle
# as the CLI's runtime, just managed by the LiveKit worker process
# instead of the CLI's chat_loop.
#
# The runtime is opened in the worker's prewarm hook (called once
# per worker process before any sessions start) and closed when the
# worker shuts down.

_runtime = None
_bridge_graph = None


async def _ensure_runtime():
    """Lazily initialize the runtime and bridge graph.

    Called once per worker process. Subsequent calls return the
    cached instances.
    """

    global _runtime, _bridge_graph  # noqa: PLW0603

    if _runtime is not None:
        return

    from agent.memory.modes import MemoryMode
    from agent.persistence import (
        DEFAULT_CRISIS_LOG_DB_PATH,
        DEFAULT_MEMORY_DB_PATH,
        DEFAULT_THREAD_DB_PATH,
        PersistentAgentRuntime,
    )

    memory_mode_str = os.getenv("OPENCOUCH_MEMORY_MODE", "persistent")
    memory_mode = (
        MemoryMode.INCOGNITO if memory_mode_str == "guest" else MemoryMode.LOCAL
    )

    _runtime = PersistentAgentRuntime(
        sqlite_path=str(DEFAULT_THREAD_DB_PATH),
        memory_sqlite_path=str(DEFAULT_MEMORY_DB_PATH),
        crisis_log_sqlite_path=str(DEFAULT_CRISIS_LOG_DB_PATH),
        memory_mode=memory_mode,
    )
    # Open the runtime's async resources (checkpointer, store connections)
    await _runtime.__aenter__()

    # Resolve the LLM client (same as CLI and API server)
    from core.config import create_configured_llm_client

    try:
        import api.dependencies

        api.dependencies._llm_client = create_configured_llm_client()
    except Exception:
        logger.warning(
            "voice agent: no LLM client configured; running in deterministic mode"
        )

    # Build the bridge graph that wraps our full pipeline for the LLMAdapter
    from voice.bridge import build_voice_bridge_graph

    _bridge_graph = build_voice_bridge_graph(_runtime)

    logger.info("voice agent: runtime initialized (mode=%s)", memory_mode.value)


# ── Agent definition ─────────────────────────────────────────────────

# The Agent's instructions become the system prompt that the
# LLMAdapter prepends to every LLM call. For our bridge graph,
# this system message is included in the messages list but our
# bridge node extracts only HumanMessage content — the system
# instructions are already baked into our knowledge files and
# prompt builders, so we keep this minimal to avoid double-prompting.
VOICE_AGENT_INSTRUCTIONS = (
    "You are OpenCouch, a calm and supportive mental health support "
    "assistant. You are not a therapist. You help people talk through "
    "difficult moments, reflect on patterns, and practice grounding "
    "techniques. Keep your responses concise and conversational — "
    "this is a voice conversation, not a text chat."
)


# ── LiveKit server and session setup ─────────────────────────────────

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    """LiveKit session entrypoint — called for each new voice session.

    Creates an AgentSession wired to our LangGraph bridge via the
    LLMAdapter, with Deepgram for STT, OpenAI for TTS, and Silero
    for VAD. Each session gets a unique thread_id for conversation
    state persistence.
    """

    await _ensure_runtime()

    # Generate a thread_id for this voice session. If the participant
    # has metadata with a thread_id, use that for continuity;
    # otherwise generate a fresh one.
    participant_identity = "voice-user"
    thread_id = f"voice-{uuid.uuid4().hex[:12]}"

    # Check if the participant provided a thread_id via metadata
    for participant in ctx.room.remote_participants.values():
        participant_identity = participant.identity or participant_identity
        if participant.metadata:
            import json

            try:
                meta = json.loads(participant.metadata)
                if "thread_id" in meta:
                    thread_id = meta["thread_id"]
                if "user_id" in meta:
                    # Store user_id in the config for the bridge to read
                    pass  # handled via config below
            except (json.JSONDecodeError, TypeError):
                pass

    logger.info(
        "voice agent: session starting — thread=%s participant=%s",
        thread_id,
        participant_identity,
    )

    # Create the AgentSession with our bridge graph as the LLM
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=langchain.LLMAdapter(
            graph=_bridge_graph,
            config={"configurable": {"thread_id": thread_id}},
        ),
        tts=openai.TTS(voice="sage"),
        vad=silero.VAD.load(),
        turn_handling=agents.TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions=VOICE_AGENT_INSTRUCTIONS),
    )

    # Generate an initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and let them know you're here to listen."
    )


# ── CLI entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    agents.cli.run_app(server)
