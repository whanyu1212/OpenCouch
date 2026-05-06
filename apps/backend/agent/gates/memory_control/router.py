"""Routing policy for explicit user memory-management requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from agent.gates.memory_control.patterns import (
    AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE as _AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE,
    INDEX_RE as _INDEX_RE,
    NO_RE as _NO_RE,
    PREFERENCE_RULE_RE as _PREFERENCE_RULE_RE,
    YES_RE as _YES_RE,
)
from agent.memory.prompts.control import (
    build_memory_control_prompt as _build_memory_control_prompt,
    build_memory_control_system_prompt as _build_memory_control_system_prompt,
)
from agent.state import AgentState
from services.base import BaseLLMClient

logger = logging.getLogger(__name__)


# ── Typed memory-control actions ─────────────────────────────────────────
#
# The wire format on graph state remains a plain dict (so LangGraph state
# serialization is unchanged). These typed models give the service layer a
# pydantic discriminated union it can validate inbound dicts against and
# dispatch on with ``match`` instead of ``dict.get`` defensiveness.


class _ActionBase(BaseModel):
    """Base for typed memory-control actions (frozen, allow extras for fwd-compat)."""

    model_config = {"frozen": True, "extra": "ignore"}


class ListAction(_ActionBase):
    type: Literal["list"] = "list"


class StatusAction(_ActionBase):
    type: Literal["status"] = "status"


class SetRecallAction(_ActionBase):
    type: Literal["set_recall"] = "set_recall"
    enabled: bool


class SavePreferenceAction(_ActionBase):
    type: Literal["save_preference"] = "save_preference"
    rule_text: str = Field(min_length=1)


class ForgetByIndexAction(_ActionBase):
    type: Literal["forget_by_index"] = "forget_by_index"
    target_kind: Literal["fact", "session", "rule"]
    target_index: int = Field(ge=1)


class ForgetByQueryAction(_ActionBase):
    type: Literal["forget_by_query"] = "forget_by_query"
    query: str = Field(min_length=1)


class ConfirmPendingAction(_ActionBase):
    type: Literal["confirm_pending"] = "confirm_pending"


class CancelPendingAction(_ActionBase):
    type: Literal["cancel_pending"] = "cancel_pending"


TypedMemoryAction = Annotated[
    Union[
        ListAction,
        StatusAction,
        SetRecallAction,
        SavePreferenceAction,
        ForgetByIndexAction,
        ForgetByQueryAction,
        ConfirmPendingAction,
        CancelPendingAction,
    ],
    Field(discriminator="type"),
]

_TYPED_ACTION_ADAPTER: TypeAdapter[TypedMemoryAction] = TypeAdapter(TypedMemoryAction)


def parse_memory_control_action(payload: dict[str, Any]) -> TypedMemoryAction | None:
    """Parse a graph-state action dict into a typed memory-control action.

    Args:
        payload: Action dict as carried on graph state (``{"type": ..., ...}``).

    Returns:
        Parsed typed action, or ``None`` when the dict is missing a known
        ``type`` discriminator or fails per-type validation.
    """

    if not payload or "type" not in payload:
        return None
    try:
        return _TYPED_ACTION_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


@dataclass(frozen=True)
class MemoryControlAction:
    """Resolved memory-management action.

    Wraps a serializable dict payload to keep LangGraph state interop
    unchanged. Use :meth:`parsed` to obtain a typed model for dispatch.
    """

    payload: dict[str, Any]

    def to_state_action(self) -> dict[str, Any]:
        """Return a serializable action for graph state updates.

        Returns:
            dict[str, Any]: Serializable memory-management action payload.
        """

        return dict(self.payload)

    def parsed(self) -> TypedMemoryAction | None:
        """Return the typed action model, or ``None`` when payload is invalid.

        Returns:
            Typed memory-control action, or ``None`` when the payload's
            ``type`` is unknown or required fields are missing/invalid.
        """

        return parse_memory_control_action(self.payload)


@dataclass(frozen=True)
class MemoryControlRoute:
    """Resolved memory-management route decision."""

    action: MemoryControlAction | None
    classifier_path: str
    llm_failure_occurred: bool


class MemoryControlDecision(BaseModel):
    """Structured output for ambiguous memory-management routing."""

    action_type: Literal[
        "none",
        "list",
        "status",
        "set_recall",
        "forget_by_query",
        "save_preference",
    ] = Field(description="The memory-management action to take, or none.")
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


def is_pending_confirmation(message: str) -> bool:
    """Return whether a message confirms a pending memory-management action.

    Args:
        message (str): Current user message.

    Returns:
        bool: ``True`` when the message confirms the pending action.
    """

    return bool(_YES_RE.match(message))


def is_pending_cancellation(message: str) -> bool:
    """Return whether a message cancels a pending memory-management action.

    Args:
        message (str): Current user message.

    Returns:
        bool: ``True`` when the message cancels the pending action.
    """

    return bool(_NO_RE.match(message))


def _extract_after_marker(text: str, markers: tuple[str, ...]) -> str:
    """Return text after the first matching marker.

    Args:
        text (str): Original user message.
        markers (tuple[str, ...]): Lowercase marker strings to search for.

    Returns:
        str: The suffix after the marker, or an empty string when no marker
            matches.
    """

    lowered = text.lower()
    for marker in markers:
        index = lowered.find(marker)
        if index >= 0:
            return text[index + len(marker) :].strip(" .:;\"'")
    return ""


def _detect_indexed_forget(lowered: str) -> MemoryControlAction | None:
    """Detect explicit ``forget <kind> #N`` memory commands.

    Args:
        lowered (str): Normalized lowercase user message.

    Returns:
        MemoryControlAction | None: Resolved memory-management action, or ``None``
            when no indexed deletion command is present.
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
    return MemoryControlAction(
        {
            "type": "forget_by_index",
            "target_kind": kind,
            "target_index": int(match.group("index")),
        }
    )


def detect_memory_control_action(message: str) -> MemoryControlAction | None:
    """Detect explicit memory-management intent from one user message.

    Args:
        message (str): Current user message.

    Returns:
        MemoryControlAction | None: Resolved memory-management action, or ``None``
            when the message should proceed through the normal therapeutic path.
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
        return MemoryControlAction({"type": "list"})

    if any(
        phrase in lowered
        for phrase in (
            "memory status",
            "what is my memory status",
            "is proactive recall on",
            "is recall on",
        )
    ):
        return MemoryControlAction({"type": "status"})

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
        return MemoryControlAction({"type": "set_recall", "enabled": False})

    if "unless i ask" in lowered and any(
        phrase in lowered
        for phrase in (
            "stop bringing",
            "don't bring",
            "do not bring",
            "stop mentioning",
            "don't mention",
            "do not mention",
            "stop using",
            "don't use",
            "do not use",
        )
    ):
        return MemoryControlAction({"type": "set_recall", "enabled": False})

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
        return MemoryControlAction({"type": "set_recall", "enabled": True})

    indexed = _detect_indexed_forget(lowered)
    if indexed is not None:
        return indexed

    if any(
        lowered.startswith(prefix)
        for prefix in (
            "please don't remember",
            "please do not remember",
            "don't remember",
            "do not remember",
            "can you not remember",
            "could you not remember",
        )
    ):
        query = _extract_after_marker(
            stripped,
            (
                "please don't remember",
                "please do not remember",
                "don't remember",
                "do not remember",
                "can you not remember",
                "could you not remember",
            ),
        )
        if query:
            return MemoryControlAction({"type": "forget_by_query", "query": query})

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
            return MemoryControlAction({"type": "forget_by_query", "query": query})

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
            return MemoryControlAction(
                {
                    "type": "save_preference",
                    "rule_text": f"You prefer {preference.rstrip('.')}.",
                }
            )

    return None


def _needs_memory_control_classifier(message: str) -> bool:
    """Return whether a message is ambiguous enough to ask the classifier.

    Args:
        message (str): Current user message.

    Returns:
        bool: ``True`` when the message contains memory-management-shaped signals
            that are not decisive enough for a hard deterministic route.
    """

    stripped = message.strip()
    if not stripped:
        return False
    return bool(_AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE.search(stripped))


def _normalize_preference_rule(rule_text: str) -> str:
    """Normalize an LLM-proposed preference rule for persistence.

    Args:
        rule_text (str): Raw model-proposed procedural rule.

    Returns:
        str: A trimmed, sentence-like rule.
    """

    normalized = " ".join(rule_text.strip().split()).strip("\"'")
    if not normalized:
        return ""
    if not normalized.endswith((".", "!", "?")):
        normalized = f"{normalized}."
    if normalized.lower().startswith(("you ", "your ", "when ", "if ")):
        return normalized
    return f"You prefer {normalized[0].lower()}{normalized[1:]}"


def _decision_to_action(decision: MemoryControlDecision) -> MemoryControlAction | None:
    """Convert a structured classifier decision into a safe action.

    Args:
        decision (MemoryControlDecision): LLM-produced memory-management routing
            decision.

    Returns:
        MemoryControlAction | None: Resolved memory-management action, or ``None``
            when the decision is too vague or unsupported.
    """

    if decision.confidence == "low" or decision.action_type == "none":
        return None
    if decision.action_type == "list":
        return MemoryControlAction({"type": "list"})
    if decision.action_type == "status":
        return MemoryControlAction({"type": "status"})
    if decision.action_type == "set_recall":
        if decision.enabled is None:
            return None
        return MemoryControlAction({"type": "set_recall", "enabled": decision.enabled})
    if decision.action_type == "forget_by_query":
        query = " ".join((decision.query or "").strip().split())
        if not query or query.lower() in {"this", "that", "it"}:
            return None
        return MemoryControlAction({"type": "forget_by_query", "query": query})
    if decision.action_type == "save_preference":
        raw_rule = " ".join((decision.rule_text or "").strip().split())
        if not raw_rule or not _PREFERENCE_RULE_RE.search(raw_rule):
            return None
        rule_text = _normalize_preference_rule(raw_rule)
        if not rule_text:
            return None
        return MemoryControlAction({"type": "save_preference", "rule_text": rule_text})
    return None


async def _classify_memory_control_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> MemoryControlAction | None:
    """Classify an ambiguous message as memory management or ordinary routing.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient): Configured control-plane LLM client.

    Returns:
        MemoryControlAction | None: Resolved memory-management action, or ``None``
            when ordinary routing should handle the turn.
    """

    decision: MemoryControlDecision = await llm_client.generate_structured(
        prompt=_build_memory_control_prompt(state),
        response_schema=MemoryControlDecision,
        system_instruction=_build_memory_control_system_prompt(),
    )
    return _decision_to_action(decision)


async def resolve_memory_control_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
) -> MemoryControlRoute:
    """Resolve memory-management routing using hard rules plus LLM middle path.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient | None): Optional control-plane LLM client.

    Returns:
        MemoryControlRoute: Resolved route decision with optional memory-management
            action, classifier path, and LLM-failure flag.
    """

    message = state.get("message", "")
    hard_action = detect_memory_control_action(message)
    if hard_action is not None:
        return MemoryControlRoute(
            action=hard_action,
            classifier_path="deterministic",
            llm_failure_occurred=False,
        )

    if not _needs_memory_control_classifier(message):
        return MemoryControlRoute(
            action=None,
            classifier_path="not_attempted",
            llm_failure_occurred=False,
        )

    if llm_client is None:
        return MemoryControlRoute(
            action=None,
            classifier_path="deterministic",
            llm_failure_occurred=False,
        )

    try:
        action = await _classify_memory_control_action(state, llm_client=llm_client)
    except Exception:
        logger.warning(
            "Memory management LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return MemoryControlRoute(
            action=None,
            classifier_path="deterministic",
            llm_failure_occurred=True,
        )

    return MemoryControlRoute(
        action=action,
        classifier_path="llm_primary",
        llm_failure_occurred=False,
    )
