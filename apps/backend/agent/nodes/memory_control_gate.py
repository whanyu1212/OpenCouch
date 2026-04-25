"""Memory-control routing gate for explicit user memory commands."""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.runtime_context import WorkflowContext
from agent.state import AgentState

_YES_RE = re.compile(
    r"^\s*(yes|yep|yeah|please do|do it|confirm|delete it)(?:,\s*delete it)?\s*[.!]?\s*$",
    re.I,
)
_NO_RE = re.compile(
    r"^\s*(no|nope|cancel|never mind|don't|do not|stop)\s*[.!]?\s*$", re.I
)
_INDEX_RE = re.compile(r"(?:#|number\s+)?(?P<index>\d+)")


def _extract_after_marker(text: str, markers: tuple[str, ...]) -> str:
    """Return text after the first matching marker.

    Args:
        text: Original user message.
        markers: Lowercase marker strings to search for.

    Returns:
        The suffix after the marker, or an empty string when no marker matches.
    """

    lowered = text.lower()
    for marker in markers:
        index = lowered.find(marker)
        if index >= 0:
            return text[index + len(marker) :].strip(" .:;\"'")
    return ""


def _detect_indexed_forget(lowered: str) -> dict[str, Any] | None:
    """Detect explicit ``forget <kind> #N`` memory commands.

    Args:
        lowered: Normalized lowercase user message.

    Returns:
        Serializable memory-control action, or ``None`` when no indexed
        deletion command is present.
    """

    if not any(verb in lowered for verb in ("forget", "delete", "remove")):
        return None
    match = _INDEX_RE.search(lowered)
    if match is None:
        return None
    kind = "fact"
    if "session" in lowered or "episode" in lowered:
        kind = "session"
    elif "rule" in lowered or "preference" in lowered:
        kind = "rule"
    return {
        "type": "forget_by_index",
        "target_kind": kind,
        "target_index": int(match.group("index")),
    }


def _detect_memory_control_action(message: str) -> dict[str, Any] | None:
    """Detect explicit memory-control intent from one user message.

    Args:
        message: Current user message.

    Returns:
        A serializable action dict, or ``None`` when the message should proceed
        through the normal therapeutic path.
    """

    stripped = message.strip()
    lowered = " ".join(stripped.lower().split())

    if not lowered:
        return None

    if any(
        phrase in lowered
        for phrase in (
            "what do you remember about me",
            "what have you saved about me",
            "show my saved memories",
            "show me my saved memories",
            "list my saved memories",
            "list my memories",
            "show my memory",
            "show memory",
        )
    ):
        return {"type": "list"}

    if any(
        phrase in lowered
        for phrase in (
            "memory status",
            "what is my memory status",
            "is proactive recall on",
            "is recall on",
        )
    ):
        return {"type": "status"}

    if any(
        phrase in lowered
        for phrase in (
            "turn proactive recall off",
            "turn recall off",
            "disable proactive recall",
            "disable recall",
            "don't bring up past sessions",
            "do not bring up past sessions",
            "don't mention past sessions",
            "do not mention past sessions",
            "don't bring up old memories",
            "do not bring up old memories",
        )
    ):
        return {"type": "set_recall", "enabled": False}

    if any(
        phrase in lowered
        for phrase in (
            "turn proactive recall on",
            "turn recall on",
            "enable proactive recall",
            "enable recall",
            "you can bring up past sessions",
            "you may bring up past sessions",
            "bring up past sessions if relevant",
        )
    ):
        return {"type": "set_recall", "enabled": True}

    indexed = _detect_indexed_forget(lowered)
    if indexed is not None:
        return indexed

    if any(verb in lowered for verb in ("forget", "delete", "remove")) and any(
        noun in lowered
        for noun in ("memory", "remember", "saved", "fact", "session", "rule")
    ):
        query = _extract_after_marker(
            stripped,
            (
                "what you remember about",
                "the memory about",
                "memory about",
                "remember about",
                "saved about",
                "fact about",
                "session about",
                "rule about",
                "forget",
                "delete",
                "remove",
            ),
        )
        if query:
            return {"type": "forget_by_query", "query": query}

    if any(
        lowered.startswith(prefix)
        for prefix in (
            "remember that i prefer",
            "please remember that i prefer",
            "can you remember that i prefer",
            "remember i prefer",
        )
    ):
        preference = _extract_after_marker(
            stripped,
            (
                "please remember that i prefer",
                "can you remember that i prefer",
                "remember that i prefer",
                "remember i prefer",
            ),
        )
        if preference:
            return {
                "type": "save_preference",
                "rule_text": f"You prefer {preference.rstrip('.')}.",
            }

    return None


async def run_memory_control_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],  # noqa: ARG001 - required node shape
) -> Command[Literal["memory_control_node", "grounded_lookup_gate_node"]]:
    """Route explicit memory-control turns before normal memory loading.

    Args:
        state: Current graph state after crisis classification.
        runtime: LangGraph runtime carrying the workflow context.

    Returns:
        State update plus the next node to run.
    """

    start = time.monotonic()
    message = state.get("message", "")
    pending_action = (state.get("memory_control", {}) or {}).get("pending_action")

    if pending_action:
        if _YES_RE.match(message):
            return Command(
                update={
                    "route": "memory_control",
                    "memory_control_action": {"type": "confirm_pending"},
                    "diagnostics": {
                        "memory_control_gate_ms": round(
                            (time.monotonic() - start) * 1000, 2
                        )
                    },
                },
                goto="memory_control_node",
            )
        if _NO_RE.match(message):
            return Command(
                update={
                    "route": "memory_control",
                    "memory_control_action": {"type": "cancel_pending"},
                    "diagnostics": {
                        "memory_control_gate_ms": round(
                            (time.monotonic() - start) * 1000, 2
                        )
                    },
                },
                goto="memory_control_node",
            )
        return Command(
            update={
                "memory_control": {"pending_action": None},
                "memory_control_action": {},
                "diagnostics": {
                    "memory_control_gate_ms": round(
                        (time.monotonic() - start) * 1000, 2
                    )
                },
            },
            goto="grounded_lookup_gate_node",
        )

    action = _detect_memory_control_action(message)
    if action is None:
        return Command(
            update={
                "memory_control_action": {},
                "diagnostics": {
                    "memory_control_gate_ms": round(
                        (time.monotonic() - start) * 1000, 2
                    )
                },
            },
            goto="grounded_lookup_gate_node",
        )

    return Command(
        update={
            "route": "memory_control",
            "memory_control_action": action,
            "diagnostics": {
                "memory_control_gate_ms": round((time.monotonic() - start) * 1000, 2)
            },
        },
        goto="memory_control_node",
    )
