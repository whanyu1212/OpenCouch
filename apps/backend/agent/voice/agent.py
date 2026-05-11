"""LiveKit voice worker entrypoint for OpenCouch."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    ChatContext,
    RunResult,
    StopResponse,
    room_io,
)
from livekit.plugins import silero

from agent.memory.hashing import iso_now
from agent.voice.agents import build_therapeutic_agent
from agent.voice.config import build_voice_system_prompt
from agent.voice.memory_context import VoiceMemoryContextService
from agent.voice.session_bootstrap import (
    build_realtime_model,
    build_turn_handling,
    close_runtime,
    ensure_runtime,
    get_control_llm_client,
    resolve_livekit_session_metadata,
    should_finalize_transcript_on_shutdown,
)
from agent.voice.session_data import SessionData
from agent.voice.transcript_finalizer import VoiceFinalizationService

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger(__name__)

_VOICE_OUTPUT_WARMUP_TOPIC = "opencouch.voice_output_warmup"


def _prewarm_process(proc: agents.JobProcess) -> None:
    """Preload blocking voice assets for a LiveKit worker process."""

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("livekit agent: prewarmed Silero VAD")
    asyncio.run(ensure_runtime())
    logger.info("livekit agent: prewarmed runtime + control LLM client")


async def _generate_text_reply_with_policy(
    session: AgentSession[SessionData],
    text: str,
    *,
    interrupt_existing: bool,
) -> None:
    """Route typed turns through the same pre-turn policy path as voice turns."""

    text = text.strip()
    if not text:
        return

    current_agent = session.current_agent
    session.userdata.last_input_modality = "text"
    if interrupt_existing:
        try:
            await session.interrupt(force=True)
        except RuntimeError:
            pass

    turn_ctx = current_agent.chat_ctx.copy()
    new_message = ChatContext().add_message(role="user", content=text)
    try:
        logger.info(
            "voice text input: running pre-turn hook agent=%s",
            type(current_agent).__name__,
        )
        await current_agent.on_user_turn_completed(turn_ctx, new_message)
    except StopResponse:
        return
    except Exception:
        logger.exception(
            "voice text input: error occurred during on_user_turn_completed"
        )
        return

    logger.info(
        "voice text input: pre-turn hook complete turn=%s consent_turn=%s exercise_type=%s crisis_level=%s",
        session.userdata.turn_index,
        session.userdata.exercise_consent_turn_index,
        session.userdata.recommended_exercise_type,
        session.userdata.crisis_level,
    )
    if session.current_agent is not current_agent:
        return

    session.generate_reply(
        user_input=new_message,
        chat_ctx=turn_ctx,
        input_modality="text",
    )


async def _handle_text_input(
    session: AgentSession[SessionData],
    event: room_io.TextInputEvent,
) -> None:
    """Handle room text input with the same policy path as spoken input."""

    await _generate_text_reply_with_policy(
        session,
        event.text or "",
        interrupt_existing=True,
    )


class OpenCouchAgentSession(AgentSession[SessionData]):
    """AgentSession variant that keeps local console text on OpenCouch policy."""

    def run(
        self,
        *,
        user_input: str,
        input_modality: Literal["text", "audio"] = "text",
        output_type: type[Any] | None = None,
    ) -> RunResult[Any]:
        """Run a local console turn.

        Args:
            user_input (str): User text passed by the LiveKit console.
            input_modality (Literal["text", "audio"]): Input modality.
            output_type (type[Any] | None): Optional final output type.

        Returns:
            RunResult[Any]: LiveKit run result for the turn.
        """

        if input_modality != "text":
            return super().run(
                user_input=user_input,
                input_modality=input_modality,
                output_type=output_type,
            )

        if self._global_run_state is not None and not self._global_run_state.done():
            raise RuntimeError("nested runs are not supported")

        run_state = RunResult(user_input=user_input, output_type=output_type)
        self._global_run_state = run_state
        task = asyncio.create_task(
            _generate_text_reply_with_policy(
                self,
                user_input,
                interrupt_existing=False,
            ),
            name="opencouch_voice_text_run",
        )
        run_state._watch_handle(task)
        return run_state


server = AgentServer(
    shutdown_process_timeout=45.0,
    initialize_process_timeout=45.0,
    setup_fnc=_prewarm_process,
)


@server.rtc_session(agent_name="opencouch-voice")
async def opencouch_voice(ctx: agents.JobContext) -> None:
    """Start one LiveKit voice session."""

    runtime = await ensure_runtime()
    control_llm = get_control_llm_client()

    await ctx.connect()
    participant = None
    participant_metadata = None
    if not ctx.is_fake_job():
        participant = await ctx.wait_for_participant()
        participant_metadata = participant.metadata

    metadata = resolve_livekit_session_metadata(
        job_metadata=ctx.job.metadata,
        participant_metadata=participant_metadata,
    )
    ctx.log_context_fields = {
        "opencouch_user_id": metadata.user_id,
        "opencouch_thread_id": metadata.thread_id,
        "opencouch_room": ctx.room.name,
    }

    logger.info(
        "livekit session: starting user=%s thread=%s participant=%s transcription_language=%s assistant_voice=%s memory_mode=%s",
        metadata.user_id,
        metadata.thread_id,
        participant.identity if participant is not None else "console",
        metadata.transcription_language or "auto",
        metadata.assistant_voice,
        metadata.memory_mode.value,
    )

    memory_context_service = VoiceMemoryContextService()
    startup_memory = await memory_context_service.load_startup_context(
        runtime.memory_store,
        user_id=metadata.user_id,
        mode=metadata.memory_mode,
    )
    instructions = build_voice_system_prompt(
        semantic_facts=startup_memory.semantic_facts,
        episodic_arcs=startup_memory.episodic_arcs,
        procedural_rules=startup_memory.procedural_rules,
        proactive_recall_enabled=startup_memory.proactive_recall_enabled,
    )
    logger.info(
        "livekit session: memory loaded facts=%d arcs=%d rules=%d prompt_chars=%d",
        len(startup_memory.semantic_facts),
        len(startup_memory.episodic_arcs),
        len(startup_memory.procedural_rules),
        len(instructions),
    )

    userdata = SessionData(
        user_id=metadata.user_id,
        thread_id=metadata.thread_id,
        memory_store=runtime.memory_store,
        memory_mode=metadata.memory_mode,
        llm_client=control_llm,
        proactive_recall_enabled=startup_memory.proactive_recall_enabled,
        started_at=iso_now(),
        therapeutic_instructions=instructions,
    )

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    session = OpenCouchAgentSession(
        llm=build_realtime_model(
            transcription_language=metadata.transcription_language,
            assistant_voice=metadata.assistant_voice,
        ),
        vad=vad,
        turn_handling=build_turn_handling(),
        userdata=userdata,
    )

    def _mark_voice_input(_event: agents.UserInputTranscribedEvent) -> None:
        session.userdata.last_input_modality = "voice"

    session.on("user_input_transcribed", _mark_voice_input)

    finalizer = VoiceFinalizationService(
        runtime=runtime,
        llm_client=control_llm,
        enabled=should_finalize_transcript_on_shutdown(
            is_fake_job=ctx.is_fake_job(),
        ),
    )
    finalization_task: asyncio.Task[None] | None = None

    def _schedule_finalization(trigger: str) -> None:
        nonlocal finalization_task
        if finalization_task is not None:
            return
        finalization_task = asyncio.create_task(
            finalizer.finalize(session, trigger=trigger),
            name="livekit_voice_finalize_transcript",
        )

    def _on_session_close(event) -> None:
        _schedule_finalization(
            f"session_close:{getattr(event.reason, 'value', event.reason)}"
        )

    session.on("close", _on_session_close)

    output_warmup_requested = False
    output_warmup_tasks: set[asyncio.Task[None]] = set()
    output_warmup_registered = False

    async def _handle_output_warmup(reader, participant_identity: str) -> None:
        nonlocal output_warmup_requested

        try:
            await reader.read_all()
        except Exception:
            logger.warning("livekit session: failed to read output warmup request")
            return

        if participant is not None and participant_identity != participant.identity:
            return
        if output_warmup_requested:
            return

        output_warmup_requested = True
        logger.info("livekit session: warming first voice output")
        session.generate_reply(
            instructions=(
                "Say exactly: \"I'm here whenever you're ready to start.\" "
                "Keep it brief and do not ask a question."
            ),
            tools=[],
            allow_interruptions=True,
        )

    def _on_output_warmup(reader, participant_identity: str) -> None:
        task = asyncio.create_task(
            _handle_output_warmup(reader, participant_identity),
            name="livekit_voice_output_warmup",
        )
        output_warmup_tasks.add(task)
        task.add_done_callback(output_warmup_tasks.discard)

    try:
        ctx.room.register_text_stream_handler(
            _VOICE_OUTPUT_WARMUP_TOPIC,
            _on_output_warmup,
        )
        output_warmup_registered = True
    except ValueError:
        logger.warning(
            "livekit session: output warmup text stream handler already registered"
        )

    async def _on_shutdown() -> None:
        if output_warmup_registered:
            try:
                ctx.room.unregister_text_stream_handler(_VOICE_OUTPUT_WARMUP_TOPIC)
            except ValueError:
                pass

        if output_warmup_tasks:
            for task in output_warmup_tasks:
                task.cancel()
            await asyncio.gather(*output_warmup_tasks, return_exceptions=True)

        if finalization_task is not None:
            await asyncio.shield(finalization_task)
        else:
            await finalizer.finalize(session, trigger="job_shutdown")

        if ctx.is_fake_job():
            await close_runtime()

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=build_therapeutic_agent(
            instructions=instructions,
            greet_on_enter=ctx.is_fake_job(),
            greet_delay_seconds=0.0,
        ),
        room_options=room_io.RoomOptions(
            text_input=room_io.TextInputOptions(text_input_cb=_handle_text_input),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
