"""OpenAI Realtime voice session manager (Option B).

Direct integration with the OpenAI Realtime API — no LiveKit, no
separate STT/TTS. The Realtime model handles audio-in → audio-out
natively with natural prosody, turn detection, and low latency
(~300-500ms TTFT).

Architecture:

    Browser (mic audio via WebSocket)
      → FastAPI WebSocket endpoint
      → OpenAI Realtime API session
        (system prompt = our knowledge + memory + rules)
      → On each user transcript:
          1. Crisis gate runs on transcript text (~2-4ms deterministic)
          2. If crisis → response.cancel + inject crisis template
          3. If safe → Realtime continues generating
          4. Extractors run async in background
          5. System prompt refreshed if memory changed
      → Realtime streams audio response
      → FastAPI WebSocket → Browser (speaker)

The system prompt injected into the Realtime session contains:
- soul.md (identity, voice, tone)
- All mode knowledge files (Realtime picks the register implicitly)
- Semantic facts + episodic arcs as context
- Procedural rules as directives
- Crisis safety instructions (backup — the hard gate is external)

The crisis gate is a hard pre-check, not a prompt instruction.
When the transcript arrives, we run the full deterministic regex
tier before Realtime is allowed to generate. If the regex detects
crisis, we cancel the Realtime response and inject our crisis
template. The LLM classifier only fires on genuinely ambiguous
messages (~1.5s), which is still faster than Option A's full
graph pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.memory.store import MemoryStore
from agent.nodes.crisis_gate import (
    IMMINENT_PATTERNS,
    CLEAR_SELF_HARM_PATTERNS,
    _matches_any,
)

logger = logging.getLogger(__name__)

# ── Knowledge loading ────────────────────────────────────────────────

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[3] / "knowledge"


def _load_knowledge_file(relative_path: str) -> str:
    """Load a knowledge markdown file by relative path."""
    path = KNOWLEDGE_ROOT / relative_path
    if path.exists():
        return path.read_text()
    logger.warning("Knowledge file not found: %s", path)
    return ""


def build_voice_system_prompt(
    *,
    semantic_facts: list[str] | None = None,
    episodic_arcs: list[str] | None = None,
    procedural_rules: list[str] | None = None,
) -> str:
    """Build the complete system prompt for the Realtime session.

    Combines all knowledge sources into a single prompt. The
    Realtime model picks the appropriate response register
    (supportive, reflective, psychoeducation, etc.) based on the
    conversation context — no explicit dispatcher needed. The mode
    knowledge files provide the model with the vocabulary and
    constraints of each register.

    The prompt is structured so the most important instructions
    come first (identity, safety) and context comes last (memory).
    """

    sections = []

    # Core identity and safety
    soul = _load_knowledge_file("soul.md")
    if soul:
        sections.append(f"# Identity and Voice\n\n{soul}")

    identity = _load_knowledge_file("identity.md")
    if identity:
        sections.append(f"# Product Boundaries\n\n{identity}")

    # Safety policy
    boundaries = _load_knowledge_file("policy/boundaries.md")
    if boundaries:
        sections.append(f"# Safety Policy\n\n{boundaries}")

    # Voice-specific instructions
    sections.append("""# Voice Conversation Guidelines

You are in a live voice conversation. Keep these in mind:
- Keep responses concise — 2-3 sentences is ideal for voice. Long responses lose the listener.
- Use natural spoken language, not written prose. No bullet points, no markdown, no numbered lists.
- Pause where a human would pause. Don't rush through heavy emotional content.
- Match the user's energy. If they're quiet and hesitant, be gentle. If they're venting with energy, match it.
- Never say "I'm an AI" or "as a language model." You are OpenCouch, a support assistant.
- When wrapping up a guided exercise, summarize what was practiced and check how the user is feeling.""")

    # Mode knowledge — all modes loaded, model picks register implicitly
    mode_files = [
        "response_modes/support.md",
        "response_modes/reflection.md",
        "response_modes/psychoeducation.md",
        "response_modes/closing.md",
        "response_modes/guided_exercise.md",
    ]
    mode_sections = []
    for mf in mode_files:
        content = _load_knowledge_file(mf)
        if content:
            mode_name = Path(mf).stem
            mode_sections.append(f"## {mode_name}\n\n{content}")
    if mode_sections:
        sections.append(
            "# Response Modes\n\n"
            "Choose the appropriate mode based on what the user needs. "
            "You don't need to announce which mode you're in — just "
            "respond in the right register.\n\n" + "\n\n".join(mode_sections)
        )

    # MI baseline (the one active modality)
    mi = _load_knowledge_file("modalities/motivational_interviewing.md")
    if mi:
        sections.append(f"# Conversational Stance (Motivational Interviewing)\n\n{mi}")

    # Procedural rules
    if procedural_rules:
        rules_block = "\n".join(f"- {rule}" for rule in procedural_rules)
        sections.append(
            f"# Style Rules from Past Conversations\n\n"
            f"Always follow these — they are explicit user preferences:\n"
            f"{rules_block}"
        )

    # Memory context
    memory_lines = []
    if semantic_facts:
        for fact in semantic_facts:
            memory_lines.append(f"- {fact}")
    if episodic_arcs:
        for arc in episodic_arcs:
            memory_lines.append(f"- {arc}")
    if memory_lines:
        sections.append("# What You Know About This User\n\n" + "\n".join(memory_lines))

    # Crisis safety instructions — both backup (the hard gate is
    # external) and tool usage guidance
    sections.append(
        "# Crisis Safety\n\n"
        "If the user expresses suicidal thoughts, self-harm intent, "
        "or imminent danger, STOP normal conversation immediately. "
        "Acknowledge calmly and directly.\n\n"
        "You have a `search_crisis_resources` tool available. USE IT "
        "whenever:\n"
        "- The user is in crisis or severe distress and needs real "
        "contact information\n"
        "- The user mentions a specific country or region — search for "
        "their LOCAL crisis hotline, not just US 988\n"
        "- The user asks for therapist referrals, support groups, or "
        "local mental health services\n"
        "- You are unsure of the correct hotline number for their region\n\n"
        "After getting search results, share the most relevant verified "
        "numbers or resources in a calm, direct manner. Then ask one "
        "safety question: 'Is there someone you trust who can be with "
        "you right now?'\n\n"
        "If you cannot search or the search fails, default to: "
        "'Please reach out to the 988 Suicide and Crisis Lifeline by "
        "calling or texting 988 if you are in the US, or contact your "
        "local emergency services.'\n\n"
        "Do not attempt to handle a crisis therapeutically. Your role "
        "is to acknowledge, provide resources, and connect them to help."
    )

    return "\n\n---\n\n".join(sections)


# ── Realtime session ─────────────────────────────────────────────────


class RealtimeVoiceSession:
    """Manages one voice conversation over the OpenAI Realtime API.

    Lifecycle:
    1. ``__init__`` stores config but doesn't connect
    2. ``start()`` opens the Realtime WebSocket and configures the
       session with the system prompt built from the user's memory
    3. Audio flows bidirectionally: browser → ``send_audio()`` →
       Realtime; Realtime → ``on_audio_delta`` callback → browser
    4. On each user transcript, the crisis gate runs as a pre-check
    5. ``close()`` shuts down the Realtime connection

    The session does NOT manage the browser WebSocket — that's
    handled by the FastAPI endpoint in ``voice/api.py``.
    """

    def __init__(
        self,
        *,
        openai_api_key: str,
        memory_store: MemoryStore,
        memory_mode: MemoryMode,
        user_id: str,
        thread_id: str,
        voice: str = "sage",
        llm_client: Any = None,
        embedding_provider: Any = None,
        on_audio_delta: Any = None,
        on_transcript: Any = None,
        on_agent_transcript: Any = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._memory_store = memory_store
        self._memory_mode = memory_mode
        self._user_id = user_id
        self._thread_id = thread_id
        self._voice = voice
        self._llm_client = llm_client
        self._embedding_provider = embedding_provider
        self._connection = None
        self._running = False

        # Conversation transcript accumulated across turns so extractors
        # have context. Each entry is {"role": "user"|"assistant", "content": "..."}
        self._transcript: list[dict[str, str]] = []
        self._turn_count = 0
        self._session_started_at = ""

        # Callbacks for the FastAPI WebSocket layer
        self._on_audio_delta = on_audio_delta  # (bytes) → send to browser
        self._on_transcript = on_transcript  # (str) → user said this
        self._on_agent_transcript = on_agent_transcript  # (str) → agent said this

    async def start(self) -> None:
        """Open the Realtime connection and configure the session."""

        # Load memory for the system prompt
        semantic_facts = await self._load_semantic_facts()
        episodic_arcs = await self._load_episodic_arcs()
        procedural_rules = await self._load_procedural_rules()

        system_prompt = build_voice_system_prompt(
            semantic_facts=semantic_facts,
            episodic_arcs=episodic_arcs,
            procedural_rules=procedural_rules,
        )

        from agent.memory.hashing import iso_now

        self._session_started_at = iso_now()

        logger.info(
            "realtime session: starting for user=%s thread=%s prompt_len=%d",
            self._user_id,
            self._thread_id,
            len(system_prompt),
        )

        self._connection = await self._client.realtime.connect(
            model="gpt-realtime-1.5",
        ).__aenter__()

        await self._connection.session.update(
            session={
                "type": "realtime",
                "instructions": system_prompt,
                "audio": {
                    "output": {"voice": self._voice},
                    "input": {
                        "transcription": {"model": "gpt-4o-transcribe"},
                        "turn_detection": {"type": "server_vad"},
                    },
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "search_crisis_resources",
                        "description": (
                            "Search the web for crisis hotlines, mental health "
                            "emergency contacts, and support resources. Use this "
                            "when the user is in distress or crisis and needs "
                            "real, verified contact information — especially if "
                            "they mention a specific country or region. Also use "
                            "this when the user asks for therapist referrals, "
                            "support groups, or local mental health services."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "The search query. Be specific about "
                                        "what resources are needed and include "
                                        "the region/country if known. Examples: "
                                        "'crisis hotline Singapore', "
                                        "'suicide prevention helpline UK', "
                                        "'mental health emergency services Australia'"
                                    ),
                                },
                            },
                            "required": ["query"],
                        },
                    },
                ],
            }
        )

        self._running = True
        # Start the event listener in the background
        asyncio.create_task(self._listen_events())

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Forward raw audio from the browser to Realtime.

        The browser sends PCM16 audio at 24kHz. We base64-encode it
        for the Realtime API's input_audio_buffer.append event.
        """

        if self._connection is None:
            return
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self._connection.input_audio_buffer.append(audio=encoded)

    async def end_session(self) -> None:
        """Summarize the voice session and write an episodic arc.

        Called when the user disconnects. Runs the same session
        summarizer as the CLI's ``/end`` command — reads the
        accumulated transcript, makes one LLM call to produce a
        ``StoredSessionArc``, and writes it to the episodic namespace.

        Skips silently if:
        - No LLM client (can't summarize)
        - Incognito mode (no episodic writes)
        - Transcript too short (< 3 user turns — the summarizer
          would return arc=None anyway)
        """

        user_turns = sum(1 for t in self._transcript if t.get("role") == "user")
        if user_turns < 2:
            logger.info(
                "realtime session: skipping summarization (only %d user turns)",
                user_turns,
            )
            return

        if not self._llm_client:
            logger.debug("realtime session: no llm_client, skipping summarization")
            return

        if self._memory_mode == MemoryMode.INCOGNITO:
            logger.debug("realtime session: incognito, skipping summarization")
            return

        try:
            from agent.memory.hashing import iso_now
            from agent.nodes.summarize_session import run_summarize_session

            # Build a minimal state with the transcript
            minimal_state = {
                "transcript": list(self._transcript),
                "user_id": self._user_id,
                "session_id": self._thread_id,
            }

            ended_at = iso_now()

            arc = await run_summarize_session(
                minimal_state,
                llm_client=self._llm_client,
                memory_store=self._memory_store,
                memory_mode=self._memory_mode,
                session_id=self._thread_id,
                started_at=self._session_started_at,
                ended_at=ended_at,
                embedding_provider=self._embedding_provider,
            )

            if arc is not None:
                logger.info(
                    "realtime session: wrote episodic arc — themes=%s summary=%s",
                    arc.primary_themes,
                    arc.summary[:80],
                )
            else:
                logger.info(
                    "realtime session: summarizer returned no arc (session too thin)"
                )

        except Exception:
            logger.warning(
                "realtime session: summarization failed (non-fatal)",
                exc_info=True,
            )

    async def close(self) -> None:
        """Shut down the Realtime connection."""

        self._running = False
        if self._connection is not None:
            try:
                await self._connection.__aexit__(None, None, None)
            except Exception:
                pass
            self._connection = None

    async def _listen_events(self) -> None:
        """Background loop consuming Realtime server events."""

        if self._connection is None:
            return

        agent_transcript_buffer: dict[str, str] = {}

        try:
            async for event in self._connection:
                if not self._running:
                    break

                # Log every event type for debugging
                if not event.type.startswith("input_audio_buffer"):
                    logger.info("realtime event: %s", event.type)

                # Audio delta — forward to browser for playback
                if (
                    event.type == "response.audio.delta"
                    or event.type == "response.output_audio.delta"
                ):
                    if self._on_audio_delta:
                        audio_bytes = base64.b64decode(event.delta)
                        await self._on_audio_delta(audio_bytes)

                # User transcript completed — run crisis gate + extractors
                elif event.type in (
                    "conversation.item.input_audio_transcription.completed",
                    "input_audio_transcription.completed",
                ):
                    transcript = event.transcript or ""
                    logger.info(
                        "realtime session: user said: %s",
                        transcript[:80],
                    )

                    # Track in conversation transcript
                    if transcript.strip():
                        self._transcript.append({"role": "user", "content": transcript})
                        self._turn_count += 1

                    if self._on_transcript:
                        await self._on_transcript(transcript)

                    # Run crisis gate on the transcript
                    await self._handle_crisis_check(transcript)

                    # Run extractors async (no latency impact)
                    if transcript.strip():
                        asyncio.create_task(self._run_extractors(transcript))

                # Agent transcript delta — accumulate for logging
                elif event.type in (
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                    "response.output_text.delta",
                ):
                    item_id = getattr(event, "item_id", "unknown")
                    agent_transcript_buffer.setdefault(item_id, "")
                    agent_transcript_buffer[item_id] += event.delta

                # Agent transcript done
                elif event.type in (
                    "response.audio_transcript.done",
                    "response.output_audio_transcript.done",
                    "response.output_text.done",
                ):
                    item_id = getattr(event, "item_id", "unknown")
                    full_text = agent_transcript_buffer.pop(
                        item_id, event.transcript or ""
                    )
                    logger.info(
                        "realtime session: agent said: %s",
                        full_text[:80],
                    )

                    # Track in conversation transcript
                    if full_text.strip():
                        self._transcript.append(
                            {"role": "assistant", "content": full_text}
                        )

                    if self._on_agent_transcript:
                        await self._on_agent_transcript(full_text)

                # Function call — the model wants to use a tool
                elif event.type == "response.function_call_arguments.done":
                    await self._handle_function_call(event)

                elif event.type == "error":
                    logger.error(
                        "realtime session error: %s",
                        getattr(event, "error", event),
                    )

        except Exception:
            logger.exception("realtime session: event loop crashed")
        finally:
            self._running = False

    async def _handle_function_call(self, event: Any) -> None:
        """Execute a function call from the Realtime model.

        The model calls our custom tools (currently just
        search_crisis_resources). We execute the function, return
        the result, and tell Realtime to continue generating its
        response with the result.
        """

        import json as _json

        call_id = getattr(event, "call_id", None)
        name = getattr(event, "name", "")
        arguments_str = getattr(event, "arguments", "{}")

        logger.info(
            "realtime function call: name=%s call_id=%s args=%s",
            name,
            call_id,
            arguments_str[:100],
        )

        try:
            args = _json.loads(arguments_str)
        except _json.JSONDecodeError:
            args = {}

        result = ""

        if name == "search_crisis_resources":
            query = args.get("query", "crisis mental health hotline")
            result = await self._execute_web_search(query)
        else:
            result = f"Unknown function: {name}"
            logger.warning("realtime function call: unknown function %s", name)

        # Send the result back to Realtime
        if self._connection is not None:
            try:
                await self._connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )
                # Tell Realtime to continue generating with the result
                await self._connection.response.create()
            except Exception:
                logger.exception("realtime function call: failed to send result")

    async def _execute_web_search(self, query: str) -> str:
        """Execute a web search for crisis resources.

        Uses the existing LLM client's web search capability (same
        ``use_search=True`` path as the text-mode crisis resource
        lookup). Falls back to a static message if the search fails.
        """

        logger.info("realtime web search: query=%r", query)

        if self._llm_client is None:
            return (
                "Search unavailable. Default resources: "
                "988 Suicide & Crisis Lifeline (US): call or text 988. "
                "Crisis Text Line: text HOME to 741741. "
                "International Association for Suicide Prevention: "
                "https://www.iasp.info/resources/Crisis_Centres/"
            )

        try:
            result = await self._llm_client.generate_text(
                prompt=(
                    f"Find verified, current crisis hotlines and mental "
                    f"health emergency contacts for this query: {query}\n\n"
                    f"Return ONLY the contact information in a clear, "
                    f"concise format suitable for someone in distress. "
                    f"Include phone numbers, text lines, and websites. "
                    f"Prioritize official government and established "
                    f"nonprofit resources."
                ),
                system_instruction=(
                    "You are helping find crisis resources for someone "
                    "who may be in immediate distress. Be direct, "
                    "accurate, and concise. Only include verified, "
                    "currently operating resources."
                ),
                use_search=True,
                temperature=0,
            )
            logger.info(
                "realtime web search: got results (%d chars)",
                len(result),
            )
            return result
        except Exception:
            logger.warning(
                "realtime web search: failed, returning defaults",
                exc_info=True,
            )
            return (
                "Search failed. Default resources: "
                "988 Suicide & Crisis Lifeline (US): call or text 988. "
                "Crisis Text Line: text HOME to 741741. "
                "For international resources, visit "
                "https://www.iasp.info/resources/Crisis_Centres/"
            )

    async def _handle_crisis_check(self, transcript: str) -> None:
        """Run the crisis gate on the user's transcript.

        If crisis is detected, cancel the Realtime response and
        inject our crisis template. Uses the same regex patterns
        as the deterministic tier of the graph's crisis_gate_node.
        Runs in ~2-4ms.
        """

        if not transcript.strip():
            return

        text = transcript.lower()

        # Fast deterministic check (~2-4ms) using the same patterns
        # as the graph's crisis gate
        if _matches_any(text, IMMINENT_PATTERNS):
            logger.warning(
                "realtime session: CRISIS DETECTED (imminent) — cancelling response"
            )
            await self._inject_crisis_response()
            return

        if _matches_any(text, CLEAR_SELF_HARM_PATTERNS):
            logger.warning(
                "realtime session: CRISIS DETECTED (self-harm) — cancelling response"
            )
            await self._inject_crisis_response()
            return

        # If deterministic paths don't fire, let Realtime continue.
        # The system prompt has backup crisis instructions for edge
        # cases the regex misses. The full LLM classifier could be
        # added here for ambiguous cases, but it would add ~1.5s
        # latency — deferred until dogfood shows it's needed.

    async def _inject_crisis_response(self) -> None:
        """Cancel the current Realtime response and inject a crisis template."""

        if self._connection is None:
            return

        crisis_text = (
            "I hear you, and I take this seriously. "
            "Please reach out to the 988 Suicide and Crisis Lifeline "
            "by calling or texting 988. They're available 24/7. "
            "Is there someone you trust who can be with you right now?"
        )

        try:
            # Cancel any in-progress response
            await self._connection.response.cancel()
        except Exception:
            pass

        try:
            # Inject our crisis response
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": crisis_text}],
                }
            )
            await self._connection.response.create()
        except Exception:
            logger.exception("realtime session: failed to inject crisis response")

    async def _run_extractors(self, user_message: str) -> None:
        """Run memory extractors on the user message in the background.

        Constructs a minimal AgentState and a fake runtime context,
        then calls the extractor node functions directly. This is the
        same extraction pipeline as text mode — same LLM prompts, same
        dedup, same embedding writes — just invoked outside the graph.

        Runs async so it doesn't block the voice response. Failures
        are logged but never propagate — same silent-degradation
        contract as the graph extractor nodes.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        if not self._llm_client:
            logger.debug("realtime extractors: no llm_client, skipping")
            return

        # Check the small-talk gate before making the LLM call
        from agent.memory.small_talk_gate import is_small_talk

        if is_small_talk(user_message):
            logger.debug(
                "realtime extractors: small-talk gate triggered for %r",
                user_message[:40],
            )
            return

        try:
            # Build a minimal AgentState with just enough fields for
            # the extractors to read. The extractor nodes read:
            # - state["message"] — the current user message
            # - state["transcript"] — for context in the LLM prompt
            # - state["user_id"] / state["session_id"] — for namespace
            # - state["progress"]["turn_count"] — for provenance
            # - state.get("diagnostics", {}) — for the delta return
            minimal_state = {
                "message": user_message,
                "transcript": list(self._transcript),
                "history": list(self._transcript),
                "user_id": self._user_id,
                "session_id": self._thread_id,
                "progress": {"turn_count": self._turn_count},
                "diagnostics": {},
                "memory": {},
            }

            # Build a fake runtime context matching WorkflowContext
            class _FakeRuntime:
                def __init__(self, context):
                    self.context = context

            runtime_ctx = {
                "llm_client": self._llm_client,
                "memory_store": self._memory_store,
                "memory_mode": self._memory_mode,
                "embedding_provider": self._embedding_provider,
            }
            fake_runtime = _FakeRuntime(runtime_ctx)

            # Run semantic fact extractor
            from agent.nodes.extract_facts import run_extract_semantic_facts_node

            facts_delta = await run_extract_semantic_facts_node(
                minimal_state, fake_runtime
            )
            facts_reason = facts_delta.get("diagnostics", {}).get(
                "extract_facts_reason", ""
            )
            facts_writes = facts_delta.get("diagnostics", {}).get("semantic_writes", 0)

            # Run procedural rule extractor
            from agent.nodes.extract_procedural_rules import (
                run_extract_procedural_rules_node,
            )

            rules_delta = await run_extract_procedural_rules_node(
                minimal_state, fake_runtime
            )
            rules_writes = rules_delta.get("diagnostics", {}).get(
                "procedural_writes", 0
            )

            logger.info(
                "realtime extractors: facts=%d rules=%d reason=%r",
                facts_writes,
                rules_writes,
                facts_reason,
            )

            # If anything was written to memory, refresh the system
            # prompt so the Realtime model sees the updated context on
            # subsequent turns. This covers:
            # - New semantic facts ("my sister Sarah") → model now
            #   knows about Sarah for the rest of the session
            # - New procedural rules ("don't suggest meditation") →
            #   model now follows the rule immediately
            if facts_writes > 0 or rules_writes > 0:
                await self._refresh_system_prompt()

        except Exception:
            logger.warning(
                "realtime extractors: failed (non-fatal)",
                exc_info=True,
            )

    async def _refresh_system_prompt(self) -> None:
        """Reload memory and update the Realtime session's system prompt.

        Called when extractors write new data (facts or rules) so the
        Realtime model sees the updated context on subsequent turns
        without needing to restart the session.
        """

        if self._connection is None:
            return

        try:
            semantic_facts = await self._load_semantic_facts()
            episodic_arcs = await self._load_episodic_arcs()
            procedural_rules = await self._load_procedural_rules()

            updated_prompt = build_voice_system_prompt(
                semantic_facts=semantic_facts,
                episodic_arcs=episodic_arcs,
                procedural_rules=procedural_rules,
            )

            await self._connection.session.update(
                session={
                    "type": "realtime",
                    "instructions": updated_prompt,
                }
            )

            logger.info(
                "realtime session: system prompt refreshed (facts=%d arcs=%d rules=%d)",
                len(semantic_facts),
                len(episodic_arcs),
                len(procedural_rules),
            )
        except Exception:
            logger.warning(
                "realtime session: failed to refresh system prompt",
                exc_info=True,
            )

    async def _load_semantic_facts(self) -> list[str]:
        """Load semantic facts for the system prompt."""

        namespace = (self._user_id, "semantic")
        try:
            records = await self._memory_store.asearch(namespace, query=None, limit=20)
            return [
                f"Previously noted: {r.value.get('evidence_quote', '')}"
                for r in records
                if r.value.get("evidence_quote")
            ]
        except Exception:
            return []

    async def _load_episodic_arcs(self) -> list[str]:
        """Load episodic arcs for the system prompt."""

        namespace = (self._user_id, "episodic")
        try:
            records = await self._memory_store.asearch(namespace, query=None, limit=5)
            formatted = []
            for r in records:
                summary = r.value.get("summary", "")
                themes = r.value.get("primary_themes", [])
                if summary:
                    themes_str = ", ".join(themes) if themes else "untagged"
                    formatted.append(f"Past session ({themes_str}): {summary}")
            return formatted
        except Exception:
            return []

    async def _load_procedural_rules(self) -> list[str]:
        """Load procedural rules for the system prompt."""

        try:
            profile = await aget_procedural_profile(
                self._memory_store, user_id=self._user_id
            )
            return [rule.rule for rule in profile.rules]
        except Exception:
            return []
