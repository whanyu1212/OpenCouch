"""Chat endpoints for conversation turns.

POST /api/chat: synchronous turn that waits for the full response.
WS   /api/chat/stream: streaming turn with status events and response text.

Both endpoints wrap ``PersistentAgentRuntime.run_turn`` and
``run_turn_stream`` respectively, converting between the HTTP/WS
contract and the internal ``AgentInput``/``AgentOutput`` types.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from agent.models import (
    AgentOutput,
    Channel,
    ChunkEvent,
    DoneEvent,
    StatusEvent,
    friendly_stage,
)
from agent.runtime.types import (
    ActiveSessionExists,
    SessionInterrupted,
    SessionLeaseExpired,
)
from api.dependencies import (
    get_llm_client,
    get_response_llm_clients,
    get_runtime_for_memory_mode,
)
from api.models import (
    ChatRequest,
    ChatResponse,
    CrisisInfo,
    StreamChunkMessage,
    StreamDoneMessage,
    StreamErrorMessage,
    StreamStatusMessage,
)
from config import ResponseModelTier
from llm.base import BaseLLMClient

router = APIRouter(tags=["chat"])

_TURN_FAILURE_HTTP_STATUS = 500
_SESSION_CONFLICT_HTTP_STATUS = 409
_TURN_FAILURE_WS_CLOSE_CODE = 1011
_SESSION_CONFLICT_WS_CLOSE_CODE = 1008
_ERROR_REASON_LIMIT = 120


def _agent_error_code(exc: Exception) -> str:
    """Return the public error code for an agent turn exception.

    Args:
        exc (Exception): Exception raised by the runtime.

    Returns:
        str: Stable public error code.
    """

    if isinstance(exc, SessionInterrupted):
        return "session_interrupted"
    if isinstance(exc, ActiveSessionExists):
        return "active_session_exists"
    if isinstance(exc, SessionLeaseExpired):
        return "session_lease_expired"
    if isinstance(exc, ValidationError):
        return "invalid_request"
    return "agent_turn_failed"


def _agent_error_status(exc: Exception) -> int:
    """Return the HTTP status for an agent turn exception.

    Args:
        exc (Exception): Exception raised by the runtime.

    Returns:
        int: Public HTTP status code.
    """

    if isinstance(
        exc,
        (
            ActiveSessionExists,
            SessionInterrupted,
            SessionLeaseExpired,
        ),
    ):
        return _SESSION_CONFLICT_HTTP_STATUS
    return _TURN_FAILURE_HTTP_STATUS


def _agent_ws_close_code(exc: Exception) -> int:
    """Return the WebSocket close code for an agent turn exception.

    Args:
        exc (Exception): Exception raised by the runtime.

    Returns:
        int: WebSocket close code.
    """

    if isinstance(
        exc,
        (
            ActiveSessionExists,
            SessionInterrupted,
            SessionLeaseExpired,
            ValidationError,
        ),
    ):
        return _SESSION_CONFLICT_WS_CLOSE_CODE
    return _TURN_FAILURE_WS_CLOSE_CODE


def _agent_error_message(exc: Exception) -> str:
    """Return a compact user-safe error message.

    Args:
        exc (Exception): Exception raised by the runtime.

    Returns:
        str: Compact error text suitable for API responses.
    """

    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _output_to_chat_response(output: AgentOutput) -> ChatResponse:
    """Map an internal AgentOutput to the API response schema.

    Args:
        output: Internal agent output returned by the runtime.

    Returns:
        Public API chat response.
    """

    return ChatResponse(
        response_text=output.response_text,
        response_type=output.response_type.value,
        response_style=output.response_style,
        therapeutic_approach=output.therapeutic_approach,
        session_action=output.session_action,
        crisis=CrisisInfo(
            level=output.crisis.level,
            confidence=output.crisis.confidence,
            reason=output.crisis.reason,
            needs_crisis_response=output.crisis.needs_crisis_response,
            needs_clarification=output.crisis.needs_clarification,
        ),
        diagnostics=output.diagnostics,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
    response_llm_clients: dict[ResponseModelTier, BaseLLMClient | None] = Depends(
        get_response_llm_clients
    ),
) -> ChatResponse:
    """Run one conversation turn and return the full response.

    This is the synchronous endpoint: it blocks until the entire
    graph pipeline completes and returns the response in one shot.
    For real-time progress updates, use the WebSocket endpoint at
    ``/api/chat/stream`` instead.

    The ``thread_id`` in the request body determines conversation
    continuity. Reuse the same thread_id to continue a conversation;
    use a new one to start fresh. The optional ``user_id`` enables
    cross-thread memory sharing (same as the CLI's ``--user-id``
    flag).

    Args:
        body: Validated chat request body.
        llm_client: Optional control-plane LLM client.
        response_llm_clients: Response-tier clients keyed by tier.

    Returns:
        The completed chat response.
    """

    response_tier: ResponseModelTier = body.response_model_tier or "fast"
    runtime = get_runtime_for_memory_mode(body.memory_mode)
    try:
        result = await runtime.run_turn(
            thread_id=body.thread_id,
            message=body.message,
            channel=Channel.WEB,
            user_id=body.user_id,
            llm_client=llm_client,
            response_llm_client=response_llm_clients.get(response_tier),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=_agent_error_status(exc),
            detail={
                "code": _agent_error_code(exc),
                "message": _agent_error_message(exc),
            },
        ) from exc
    return _output_to_chat_response(result.output)


@router.websocket("/chat/stream")
async def chat_stream(
    websocket: WebSocket,
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
    response_llm_clients: dict[ResponseModelTier, BaseLLMClient | None] = Depends(
        get_response_llm_clients
    ),
) -> None:
    """Stream a conversation turn over WebSocket.

    Protocol:

    1. Client connects and sends a JSON message matching the
       ``ChatRequest`` schema (message, thread_id, optional user_id).
    2. Server streams back JSON messages:
       - ``{"type": "status", "stage": "...", "detail": ""}`` for
         each graph node that completes
       - ``{"type": "done", "response": {...}}`` as the terminal
         message carrying the full ChatResponse

    After the ``done`` message, the server closes the WebSocket.
    The client can reconnect for the next turn.

    Error handling: if the graph raises during execution, the server
    sends a terminal ``{"type": "error", ...}`` message before closing.
    Session liveness conflicts close with code 1008. Unexpected turn
    failures close with code 1011.

    Args:
        websocket: Accepted WebSocket connection.
        llm_client: Optional control-plane LLM client.
        response_llm_clients: Response-tier clients keyed by tier.

    Returns:
        None.
    """

    await websocket.accept()

    try:
        # Read the single request message from the client.
        data = await websocket.receive_json()
        request = ChatRequest.model_validate(data)
        response_tier: ResponseModelTier = request.response_model_tier or "fast"
        runtime = get_runtime_for_memory_mode(request.memory_mode)

        async for event in runtime.run_turn_stream(
            thread_id=request.thread_id,
            message=request.message,
            channel=Channel.WEB,
            user_id=request.user_id,
            llm_client=llm_client,
            response_llm_client=response_llm_clients.get(response_tier),
        ):
            if isinstance(event, StatusEvent):
                status_msg = StreamStatusMessage(
                    stage=friendly_stage(event.stage),
                    detail=event.detail,
                )
                await websocket.send_json(status_msg.model_dump())

            elif isinstance(event, ChunkEvent):
                chunk_msg = StreamChunkMessage(text=event.text)
                await websocket.send_json(chunk_msg.model_dump())

            elif isinstance(event, DoneEvent):
                chat_response = _output_to_chat_response(event.output)
                done_msg = StreamDoneMessage(response=chat_response)
                await websocket.send_json(done_msg.model_dump())

    except WebSocketDisconnect:
        # Client disconnected before the turn finished. The runtime
        # will complete the turn regardless, but we stop sending messages.
        pass
    except Exception as exc:
        message = _agent_error_message(exc)
        try:
            error_msg = StreamErrorMessage(
                code=_agent_error_code(exc),
                message=message,
            )
            await websocket.send_json(error_msg.model_dump())
            await websocket.close(
                code=_agent_ws_close_code(exc),
                reason=message[:_ERROR_REASON_LIMIT],
            )
        except Exception:
            pass
