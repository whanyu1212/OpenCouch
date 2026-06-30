"""Shared post-turn finalization helpers for persistent runtime paths.

The runtime owns the lifecycle ordering around a successfully finalized turn. In
particular, once app state is durably saved, safety-event capture must get a
bounded chance to run before best-effort SDK bookkeeping or public completion
signals can move the turn forward.
"""

from __future__ import annotations

from typing import Protocol

from agent.audit.capture import SafetyEventCaptureResult, capture_crisis_outcome
from agent.runtime.session.manager import ActiveSessionManager
from agent.runtime.state_store import RuntimeStateStore
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState


class EnsureSdkTurnRecorded(Protocol):
    """Callable that reconciles a finalized app turn into SDK history."""

    async def __call__(
        self,
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        """Record a finalized user/assistant turn for model-visible history."""


async def capture_post_save_safety_event(
    final_state: AgentState,
    workflow_context: WorkflowContext,
) -> SafetyEventCaptureResult:
    """Run the post-save safety capture invariant for a finalized turn.

    Callers must invoke this only after the final app state has been saved. The
    underlying capture is bounded/best-effort and returns an observable status,
    so non-crisis turns, failures, and timeouts remain explicit without leaking
    crisis-ledger storage details into runtime paths.
    """

    return await capture_crisis_outcome(final_state, workflow_context)


async def finalize_successful_turn(
    *,
    thread_id: str,
    user_message: str,
    final_state: AgentState,
    workflow_context: WorkflowContext,
    state_store: RuntimeStateStore,
    active_session_manager: ActiveSessionManager,
    mutation_token: str,
    ensure_sdk_turn_recorded: EnsureSdkTurnRecorded,
    capture_safety_event: bool = True,
) -> SafetyEventCaptureResult:
    """Persist and finalize a successful turn while preserving ordering.

    The caller must already own the per-thread lock and be inside
    ``active_session_mutation`` for ``mutation_token``. The ordering is the
    contract shared by text, streaming, and voice paths:

    1. save final app state;
    2. run the post-save safety capture invariant;
    3. reconcile SDK history;
    4. clear the active-session mutation marker.

    ``capture_safety_event`` should stay ``True`` for text turns, whose crisis
    state is recomputed every turn. Voice callers may set it ``False`` for a
    current non-crisis Realtime turn because voice state can intentionally carry
    prior crisis fields for other lifecycle bookkeeping.

    The returned safety-capture result is available for tests and future
    diagnostics; current callers rely on the capture module's emitted events.
    """

    await state_store.save_state(thread_id, final_state)
    if capture_safety_event:
        capture_result = await capture_post_save_safety_event(
            final_state,
            workflow_context,
        )
    else:
        capture_result = SafetyEventCaptureResult(
            kind="crisis_response",
            status="skipped",
            reason="safety_capture_not_required",
        )
    await ensure_sdk_turn_recorded(
        thread_id,
        user_message=user_message,
        final_state=final_state,
    )
    await active_session_manager.clear_active_session_mutation(
        thread_id,
        mutation_token,
    )
    return capture_result


__all__ = [
    "EnsureSdkTurnRecorded",
    "capture_post_save_safety_event",
    "finalize_successful_turn",
]
