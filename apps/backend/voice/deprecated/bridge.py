"""Bridge graph for LiveKit LLMAdapter ↔ OpenCouch AgentState.

The LiveKit ``livekit-plugins-langchain`` ``LLMAdapter`` expects a
compiled LangGraph that follows the LangChain message convention:
state contains a ``messages`` list of ``HumanMessage``/``AIMessage``
objects, and the graph streams response tokens via the ``messages``
stream mode.

Our agent graph uses a custom ``AgentState`` TypedDict with fields
like ``message``, ``transcript``, ``history``, ``working_memory``,
``routing``, ``response``, etc. The LLMAdapter can't talk to it
directly.

This module provides ``build_voice_bridge_graph()`` which compiles
a thin wrapper graph that:

1. Receives LangChain-style messages from the LLMAdapter
2. Extracts the latest user message and conversation history
3. Invokes our full agent pipeline (crisis gate, dispatcher,
   modes, extractors — everything)
4. Yields the response text back as an ``AIMessage`` for the
   LLMAdapter to stream to TTS

The bridge is ~zero-latency overhead — it's just dict manipulation,
no LLM or network calls. All the real work happens inside our
compiled agent workflow.

Thread persistence: the bridge passes a ``thread_id`` via
``RunnableConfig["configurable"]["thread_id"]`` so the LangGraph
checkpointer maintains conversation state across turns. The LiveKit
worker sets this from the room/participant metadata.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.models import Channel, Message, MessageRole
from agent.persistence import PersistentAgentRuntime

logger = logging.getLogger(__name__)


class BridgeState(TypedDict):
    """State schema for the bridge graph.

    The ``messages`` field follows the LangChain convention that
    ``LLMAdapter`` expects. ``add_messages`` is the standard reducer
    that appends new messages rather than replacing the list.
    """

    messages: Annotated[list[BaseMessage], add_messages]


def _extract_history_from_messages(
    messages: list[BaseMessage],
) -> list[Message]:
    """Convert LangChain messages to our Message format for history.

    The LLMAdapter sends the full conversation history as a list of
    BaseMessage objects. We convert them to our internal Message
    format so ``build_initial_state`` can populate the transcript.

    We skip the last message (which is the current user turn — it
    goes into ``message``, not ``history``) and the first message
    if it's a SystemMessage (those are the Agent instructions, not
    conversation history).
    """

    history: list[Message] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(Message(role=MessageRole.USER, content=msg.content))
        elif isinstance(msg, AIMessage) and msg.content:
            history.append(Message(role=MessageRole.ASSISTANT, content=msg.content))
    return history


def build_voice_bridge_graph(
    runtime: PersistentAgentRuntime,
) -> StateGraph:
    """Build and compile the bridge graph for the LLMAdapter.

    The bridge graph has a single node that:
    1. Reads the latest HumanMessage from BridgeState.messages
    2. Converts the conversation history to our Message format
    3. Calls runtime.run_turn() with the extracted message
    4. Returns an AIMessage with the response text

    Args:
        runtime: The active PersistentAgentRuntime instance. The
            bridge calls ``run_turn()`` on this runtime, which
            handles the full graph pipeline including checkpointing.

    Returns:
        A compiled StateGraph ready to pass to ``LLMAdapter(graph=...)``.
    """

    async def bridge_node(
        state: BridgeState,
        config: dict[str, Any] | None = None,
    ) -> dict[str, list[BaseMessage]]:
        """The single node in the bridge graph.

        Extracts the user message, runs the full agent pipeline,
        and returns the response as an AIMessage.
        """

        messages = state["messages"]

        # Find the latest HumanMessage — that's the current user turn
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return {
                "messages": [
                    AIMessage(content="I didn't catch that. Could you say that again?")
                ]
            }

        # Extract thread_id from config (set by the LiveKit worker)
        thread_id = "voice-default"
        user_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id", thread_id)
            user_id = config["configurable"].get("user_id")

        # Note: we don't need to pass history explicitly — run_turn()
        # loads it from the LangGraph checkpoint. The LLMAdapter sends
        # prior messages for context, but the runtime already has the
        # authoritative transcript.

        # Get the LLM client from the runtime's context
        # (resolved at startup, same as the CLI path)
        from api.dependencies import _llm_client

        try:
            result = await runtime.run_turn(
                thread_id=thread_id,
                message=user_message,
                channel=Channel.VOICE,
                user_id=user_id,
                llm_client=_llm_client,
            )
            response_text = result.output.response_text

            logger.info(
                "voice bridge: thread=%s mode=%s crisis=%d response_len=%d",
                thread_id,
                result.output.mode,
                result.output.crisis.level,
                len(response_text),
            )

            return {"messages": [AIMessage(content=response_text)]}

        except Exception:
            logger.exception("voice bridge: run_turn failed")
            return {
                "messages": [
                    AIMessage(
                        content="I'm having trouble right now. Can you try again?"
                    )
                ]
            }

    # Build and compile the single-node bridge graph
    graph = StateGraph(BridgeState)
    graph.add_node("bridge", bridge_node)
    graph.add_edge(START, "bridge")
    graph.add_edge("bridge", END)

    return graph.compile()
