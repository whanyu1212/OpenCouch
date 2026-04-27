"""Memory-control routing gate for explicit user memory commands."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class MemoryControlDecision(BaseModel):
    """Structured output for ambiguous memory-control routing."""

    action_type: Literal[
        "none",
        "list",
        "status",
        "set_recall",
        "forget_by_query",
        "save_preference",
    ] = Field(description="The memory-control action to take, or none.")
    enabled: bool | None = Field(
        default=None,
        description="Desired proactive-recall state for set_recall actions.",
    )
    query: str | None = Field(
        default=None,
        description="Concrete saved-memory target for forget_by_query actions.",
    )
    rule_text: str | None = Field(
        default=None,
        description="Second-person procedural preference rule to save.",
    )
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


_YES_RE = re.compile(
    r"^\s*(yes|yep|yeah|please do|do it|confirm|delete it)(?:,\s*delete it)?\s*[.!]?\s*$",
    re.I,
)
_NO_RE = re.compile(
    r"^\s*(no|nope|cancel|never mind|don't|do not|stop)\s*[.!]?\s*$", re.I
)
_INDEX_RE = re.compile(r"(?:#|number\s+)?(?P<index>\d+)")
_AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE = re.compile(
    r"("
    r"\bcan you (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"\bcould you (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"\bplease (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"^\s*keep in mind that\b|^\s*remember (?:this|that)\b|"
    r"^\s*save (?:this|that)\b|"
    r"\bdon't (?:remember|save|store|bring up|bring (?:this|that|it) up|"
    r"mention|use)\b|"
    r"\bdo not (?:remember|save|store|bring up|bring (?:this|that|it) up|"
    r"mention|use)\b|"
    r"\bstop (?:remembering|saving|storing|bringing up|"
    r"bringing (?:this|that|it) up|mentioning|using)\b|"
    r"\bforget (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bdelete (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bremove (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bwhat (?:do|have) you (?:remember|know|saved)\b"
    r")",
    re.I,
)
_PREFERENCE_RULE_RE = re.compile(
    r"\b(prefer|preference|respond|repl(?:y|ies)|answer|ask|remind|bring up|"
    r"mention|use|avoid|tone|style|brief|short(?:er)?|concise|gentle|direct|"
    r"format|language|call me|address me)\b",
    re.I,
)


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


def _build_memory_control_prompt(state: AgentState) -> str:
    """Build the LLM prompt for ambiguous memory-control routing.

    Args:
        state: Current graph state.

    Returns:
        Prompt asking for a structured memory-control decision.
    """

    history_lines = []
    for turn in (state.get("history", []) or [])[-6:]:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if content:
            history_lines.append(f"{role}: {content}")
    recent_history = "\n".join(history_lines) or "(none)"
    return (
        "Decide whether the user's message is an explicit request to manage "
        "OpenCouch's saved memory before ordinary therapeutic routing.\n\n"
        "Route to memory control only for requests to list or inspect saved "
        "memories, check memory status, enable or disable proactive recall, "
        "delete a concrete saved memory, or save a preference about how the "
        "assistant should respond or use memory.\n\n"
        "Do not route ordinary autobiographical facts, requests for help with "
        "human memory, or reflective statements such as 'I remember...', "
        "'I keep forgetting...', or 'help me remember to...'. Do not route "
        "new facts like names, pets, places, or life details; normal memory "
        "extraction handles those later. Use action_type='none' when uncertain.\n\n"
        "For forget_by_query, provide a concrete saved-memory target from the "
        "message or recent conversation. Do not confirm deletion; the memory "
        "control node will ask the user before deleting anything.\n\n"
        "For save_preference, only save response or memory-use preferences. "
        "Return rule_text as a concise second-person rule, for example "
        "'You prefer concise replies.'\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def _build_memory_control_system_prompt() -> str:
    """Build the system prompt for the memory-control classifier.

    Returns:
        System instruction for structured memory-control routing.
    """

    return (
        "You are a strict routing classifier. Return only the structured "
        "decision. You do not answer the user."
    )


def _needs_memory_control_classifier(message: str) -> bool:
    """Return whether a message is ambiguous enough to ask the classifier.

    Args:
        message: Current user message.

    Returns:
        ``True`` when the message contains memory-control-shaped signals that
        are not decisive enough for a hard deterministic route.
    """

    stripped = message.strip()
    if not stripped:
        return False
    return bool(_AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE.search(stripped))


def _normalize_preference_rule(rule_text: str) -> str:
    """Normalize an LLM-proposed preference rule for persistence.

    Args:
        rule_text: Raw model-proposed procedural rule.

    Returns:
        A trimmed, sentence-like rule.
    """

    normalized = " ".join(rule_text.strip().split()).strip("\"'")
    if not normalized:
        return ""
    if not normalized.endswith((".", "!", "?")):
        normalized = f"{normalized}."
    if normalized.lower().startswith(("you ", "your ", "when ", "if ")):
        return normalized
    return f"You prefer {normalized[0].lower()}{normalized[1:]}"


def _decision_to_action(decision: MemoryControlDecision) -> dict[str, Any] | None:
    """Convert a structured classifier decision into a safe action.

    Args:
        decision: LLM-produced memory-control routing decision.

    Returns:
        A serializable memory-control action, or ``None`` when the decision is
        too vague or unsupported.
    """

    if decision.confidence == "low" or decision.action_type == "none":
        return None
    if decision.action_type == "list":
        return {"type": "list"}
    if decision.action_type == "status":
        return {"type": "status"}
    if decision.action_type == "set_recall":
        if decision.enabled is None:
            return None
        return {"type": "set_recall", "enabled": decision.enabled}
    if decision.action_type == "forget_by_query":
        query = " ".join((decision.query or "").strip().split())
        if not query or query.lower() in {"this", "that", "it"}:
            return None
        return {"type": "forget_by_query", "query": query}
    if decision.action_type == "save_preference":
        raw_rule = " ".join((decision.rule_text or "").strip().split())
        if not raw_rule or not _PREFERENCE_RULE_RE.search(raw_rule):
            return None
        rule_text = _normalize_preference_rule(raw_rule)
        if not rule_text:
            return None
        return {"type": "save_preference", "rule_text": rule_text}
    return None


async def _classify_memory_control_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> dict[str, Any] | None:
    """Classify an ambiguous message as memory control or ordinary routing.

    Args:
        state: Current graph state.
        llm_client: Configured control-plane LLM client.

    Returns:
        Memory-control action, or ``None`` when ordinary routing should handle
        the turn.
    """

    decision: MemoryControlDecision = await llm_client.generate_structured(
        prompt=_build_memory_control_prompt(state),
        response_schema=MemoryControlDecision,
        system_instruction=_build_memory_control_system_prompt(),
    )
    return _decision_to_action(decision)


async def _resolve_memory_control_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Resolve memory-control routing using hard rules plus LLM middle path.

    Args:
        state: Current graph state.
        llm_client: Optional control-plane LLM client.

    Returns:
        Tuple of action, classifier path, and whether an LLM failure occurred.
    """

    message = state.get("message", "")
    hard_action = _detect_memory_control_action(message)
    if hard_action is not None:
        return hard_action, "deterministic", False

    if not _needs_memory_control_classifier(message):
        return None, "not_attempted", False

    if llm_client is None:
        return None, "deterministic", False

    try:
        action = await _classify_memory_control_action(state, llm_client=llm_client)
    except Exception:
        logger.warning(
            "Memory control LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return None, "deterministic", True

    return action, "llm_primary", False


async def run_memory_control_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
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

    (
        action,
        classifier_path,
        llm_failure_occurred,
    ) = await _resolve_memory_control_action(
        state,
        llm_client=runtime.context.llm_client,
    )
    diagnostics = {
        "memory_control_gate_ms": round((time.monotonic() - start) * 1000, 2),
        "memory_control_classifier_path": classifier_path,
        "memory_control_llm_failure_occurred": llm_failure_occurred,
    }

    if action is None:
        return Command(
            update={
                "memory_control_action": {},
                "diagnostics": diagnostics,
            },
            goto="grounded_lookup_gate_node",
        )

    return Command(
        update={
            "route": "memory_control",
            "memory_control_action": action,
            "diagnostics": diagnostics,
        },
        goto="memory_control_node",
    )
