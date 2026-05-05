"""Standalone shared tools for LiveKit voice agents.

These are passed via ``tools=[...]`` on agent constructors so they
can be shared across agents without duplication.  Agent-specific
tools (e.g. ``CrisisAgent.de_escalate``) live as methods on the
agent class instead.

All tools receive a ``RunContext[SessionData]`` which gives access
to ``context.userdata`` for session state.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import cast

from livekit.agents import RunContext, function_tool

from agent.memory.user_controls import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    set_memory_recall,
)
from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import aget_procedural_profile
from agent.safety.crisis_rules import (
    AMBIGUOUS_PATTERNS,
    CLEAR_SELF_HARM_PATTERNS,
    IMMINENT_PATTERNS,
)
from agent.state import AgentState
from agent.tools.grounded_lookup import answer_grounded_lookup
from agent.tools.web_search import ResourceLookupStatus, find_local_crisis_resources
from voice.livekit.activity import emit_voice_activity
from voice.livekit.session_data import SessionData

logger = logging.getLogger(__name__)


# ── Keyword patterns for the safety net ─────────────────────────────
# Subset of the deterministic patterns used for fast pre-screening
# in on_user_turn_completed.  These are the high-confidence patterns
# that should trigger an immediate agent swap without waiting for
# the LLM to decide.
SAFETY_NET_PATTERNS: tuple[str, ...] = IMMINENT_PATTERNS + CLEAR_SELF_HARM_PATTERNS


def matches_crisis_keywords(text: str) -> bool:
    """Return True if text matches any high-confidence crisis pattern.

    Used by the ``on_user_turn_completed`` safety net to force an
    immediate handoff to CrisisAgent without waiting for the LLM.
    """
    lowered = text.lower()
    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE)
        for pattern in SAFETY_NET_PATTERNS
    )


def matches_ambiguous_distress(text: str) -> bool:
    """Return True if text matches ambiguous distress patterns (level 1)."""
    lowered = text.lower()
    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE)
        for pattern in AMBIGUOUS_PATTERNS
    )


# ── Shared function tools ───────────────────────────────────────────


def _memory_access_error(userdata: SessionData) -> str | None:
    """Return a user-facing memory access error when memory is unavailable.

    Args:
        userdata: Current LiveKit session state.

    Returns:
        Error reply, or ``None`` when persistent memory is available.
    """

    if userdata.memory_mode == MemoryMode.INCOGNITO:
        return (
            "You're in guest mode, so I don't have persistent memory to show or edit "
            "for this session."
        )
    if userdata.memory_store is None:
        return "Memory is not available in this session."
    return None


def _format_memory_overview(previews: dict[str, list[str]]) -> str:
    """Render saved memory previews for spoken voice responses.

    Args:
        previews: Memory previews grouped by facts, sessions, and rules.

    Returns:
        Voice-friendly memory overview.
    """

    lines: list[str] = []
    labels = {
        "facts": "Saved facts",
        "sessions": "Session summaries",
        "rules": "Style preferences",
    }
    for key in ("facts", "sessions", "rules"):
        items = previews.get(key, [])
        if not items:
            continue
        lines.append(f"{labels[key]}:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))

    if not lines:
        return (
            "I don't have any saved facts, session summaries, or style preferences "
            "for you right now."
        )
    return "Here's what I currently have saved:\n" + "\n".join(lines)


def _pending_delete_reply(target: MemoryControlTarget) -> str:
    """Build the confirmation prompt for a pending memory deletion.

    Args:
        target: Selected saved memory target.

    Returns:
        User-facing confirmation prompt.
    """

    return (
        f'I found this saved {target["kind"]}: "{target["preview"]}". '
        "If you want me to delete it, say yes or tell me to delete it."
    )


def _multiple_delete_matches_reply(targets: list[MemoryControlTarget]) -> str:
    """Build a disambiguation prompt for multiple deletion matches.

    Args:
        targets: Candidate saved memories matching the user's request.

    Returns:
        User-facing disambiguation prompt.
    """

    lines = [
        "I found more than one saved memory that might match. "
        "Which number should I delete?"
    ]
    lines.extend(
        f"{index}. {target['kind']}: {target['preview']}"
        for index, target in enumerate(targets, start=1)
    )
    return "\n".join(lines)


def _lookup_state_from_context(
    context: RunContext[SessionData],
    *,
    message: str,
) -> AgentState:
    """Build the state slice needed by shared lookup helpers.

    Args:
        context: LiveKit run context.
        message: Current lookup request or location statement.

    Returns:
        Minimal graph-state-shaped dict containing message, history, and
        owner identity.
    """

    history: list[dict[str, str]] = []
    session = getattr(context, "session", None)
    current_agent = getattr(session, "current_agent", None)
    chat_ctx = getattr(current_agent, "chat_ctx", None)
    for item in getattr(chat_ctx, "items", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        role = getattr(item, "role", "")
        role = getattr(role, "value", role)
        if role not in {"user", "assistant"}:
            continue
        content = (getattr(item, "text_content", None) or "").strip()
        if content:
            history.append({"role": str(role), "content": content})

    userdata = context.userdata
    return cast(
        AgentState,
        {
            "message": message,
            "history": history[-6:],
            "user_id": userdata.user_id,
            "session_id": userdata.thread_id,
        },
    )


def _format_crisis_resources(
    *,
    location: str,
    resources: list[dict[str, str]],
) -> str:
    """Render verified crisis resources for voice.

    Args:
        location: User-stated location used for lookup.
        resources: Parsed verified resource rows.

    Returns:
        Voice-friendly resource list.
    """

    lines = [f"I found these verified crisis resources for {location}:"]
    for resource in resources:
        contact = resource.get("phone", "").strip()
        url = resource.get("url", "").strip()
        detail = contact
        if url:
            detail = f"{detail} ({url})" if detail else url
        lines.append(f"- {resource.get('name', 'Crisis resource')}: {detail}")
    lines.append("If there is immediate danger, call local emergency services now.")
    return "\n".join(lines)


def _crisis_resource_fallback(status: ResourceLookupStatus) -> str:
    """Return safe fallback crisis-resource text.

    Args:
        status: Lookup status from the shared crisis resource helper.

    Returns:
        User-facing fallback text that avoids unverified local details.
    """

    if status == "no_location":
        return (
            "I can look for local crisis resources if you share your country "
            "or city. If you're in immediate danger, call local emergency "
            "services now. If you're in the US or Canada, call or text 988."
        )
    return (
        "I couldn't verify local crisis resources right now, so I won't guess. "
        "If you're in immediate danger, call local emergency services now. "
        "If you're in the US or Canada, call or text 988. You can also use "
        "findahelpline.com to find support by country."
    )


@function_tool()
async def save_insight(
    context: RunContext[SessionData],
    fact: str,
) -> str:
    """Save an important fact about the user to memory for future sessions.

    Call this when the user shares something meaningful about themselves,
    their situation, relationships, or preferences that would be useful
    to remember in future conversations.

    Args:
        fact: A concise factual statement about the user.
    """
    userdata = context.userdata
    store = userdata.memory_store

    if store is None:
        return "Memory is not available in this session."

    if userdata.memory_mode == MemoryMode.INCOGNITO:
        return "Memory is disabled in guest mode."

    # This mutates durable memory, so do not allow a barge-in to hide a write.
    context.disallow_interruptions()

    namespace = (userdata.user_id, "semantic")
    key = f"voice-insight-{uuid.uuid4().hex[:12]}"

    await store.aput(
        namespace,
        key,
        {
            "evidence_quote": fact,
            "created_at": iso_now(),
            "source": "voice_tool",
            "thread_id": userdata.thread_id,
        },
    )

    logger.info(
        "save_insight: user=%s key=%s fact=%s",
        userdata.user_id,
        key,
        fact[:80],
    )
    await emit_voice_activity(
        context,
        activity="memory_saved",
        status="completed",
        label="Memory saved",
        detail="Saved for future sessions.",
    )

    return (
        "Saved for future conversations. Do not narrate the save unless "
        "the user asked you to."
    )


@function_tool()
async def answer_grounded_factual_lookup(
    context: RunContext[SessionData],
    query: str,
) -> str:
    """Answer an explicit factual or current-information request with search.

    Call this only when the user clearly asks you to look up, verify, or check
    factual/current information. Do not call this for ordinary therapeutic
    support, coping suggestions, emotional reflection, or exercise requests.

    Args:
        query: The exact factual/current-information question to verify.

    Returns:
        Search-grounded answer text, or a safe no-guess fallback.
    """

    query = query.strip()
    if not query:
        return "What would you like me to verify?"

    llm_client = context.userdata.llm_client
    if llm_client is None:
        await emit_voice_activity(
            context,
            activity="factual_lookup",
            status="failed",
            label="Lookup unavailable",
            detail="Search is not available in this voice session.",
        )
        return (
            "I can't do a verified lookup in this voice session right now, "
            "so I won't guess."
        )

    await emit_voice_activity(
        context,
        activity="factual_lookup",
        status="started",
        label="Lookup started",
        detail="Checking current information.",
    )
    answer, status = await answer_grounded_lookup(
        _lookup_state_from_context(context, message=query),
        llm_client=llm_client,
        query=query,
    )
    if status == "answered":
        await emit_voice_activity(
            context,
            activity="factual_lookup",
            status="completed",
            label="Lookup used",
            detail="Answered with grounded search.",
        )
        return answer
    if answer:
        await emit_voice_activity(
            context,
            activity="factual_lookup",
            status="failed",
            label="Lookup incomplete",
            detail="Search could not verify a final answer.",
        )
        return answer
    await emit_voice_activity(
        context,
        activity="factual_lookup",
        status="failed",
        label="Lookup incomplete",
        detail="Search could not verify a final answer.",
    )
    return (
        "I couldn't verify that with search right now, so I won't guess. "
        "If you want, give me more detail and I can try again."
    )


@function_tool()
async def provide_crisis_resources(
    context: RunContext[SessionData],
    location_context: str = "",
) -> str:
    """Find verified crisis resources for the user's stated location.

    Call this in crisis mode when the user asks for numbers, hotlines, crisis
    resources, or local emergency mental-health support. Pass the user's own
    stated location or the latest location-bearing message. Do not infer a
    location from IP address, accent, timezone, or someone else's location.

    Args:
        location_context: User's location statement or latest relevant message.

    Returns:
        Verified local crisis resources when found, otherwise a safe fallback.
    """

    llm_client = context.userdata.llm_client
    if llm_client is None:
        await emit_voice_activity(
            context,
            activity="crisis_resources_lookup",
            status="failed",
            label="Resource lookup unavailable",
            detail="Local crisis-resource search is not available.",
        )
        return _crisis_resource_fallback("search_failed")

    await emit_voice_activity(
        context,
        activity="crisis_resources_lookup",
        status="started",
        label="Crisis resources search started",
        detail="Checking verified local resources.",
    )
    location, resources, status = await find_local_crisis_resources(
        _lookup_state_from_context(
            context,
            message=location_context.strip(),
        ),
        llm_client=llm_client,
    )
    if status == "found":
        await emit_voice_activity(
            context,
            activity="crisis_resources_lookup",
            status="completed",
            label="Crisis resources found",
            detail="Verified local resources were found.",
        )
        return _format_crisis_resources(location=location, resources=resources)
    await emit_voice_activity(
        context,
        activity="crisis_resources_lookup",
        status="pending" if status == "no_location" else "failed",
        label=(
            "Location needed"
            if status == "no_location"
            else "Resource lookup incomplete"
        ),
        detail=(
            "Share a country or city to search local resources."
            if status == "no_location"
            else "Local crisis resources could not be verified."
        ),
    )
    return _crisis_resource_fallback(status)


@function_tool()
async def show_saved_memory(context: RunContext[SessionData]) -> str:
    """Show the user's saved memory.

    Call this when the user asks what you remember, what you have saved
    about them, or asks to list saved memories. Keep the spoken response
    concise and operational.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    previews = await list_memory_for_owner(
        store,
        owner_id=userdata.user_id,
    )
    return _format_memory_overview(previews)


@function_tool()
async def show_memory_status(context: RunContext[SessionData]) -> str:
    """Show memory counts and proactive recall status.

    Call this when the user asks whether memory is on, asks for memory
    status, or asks whether proactive recall is enabled.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    profile = await aget_procedural_profile(
        store,
        user_id=userdata.user_id,
    )
    fact_count = await store.arecord_count((userdata.user_id, "semantic"))
    session_count = await store.arecord_count((userdata.user_id, "episodic"))
    userdata.proactive_recall_enabled = profile.proactive_recall_enabled
    return (
        "Memory status:\n"
        f"Saved facts: {fact_count}\n"
        f"Session summaries: {session_count}\n"
        f"Style preferences: {len(profile.rules)}\n"
        f"Proactive recall: {'on' if profile.proactive_recall_enabled else 'off'}"
    )


@function_tool()
async def set_proactive_memory_recall(
    context: RunContext[SessionData],
    enabled: bool,
) -> str:
    """Turn proactive memory recall on or off.

    Call this when the user says not to bring up past sessions, not to
    mention old memories, or says that you may bring up past sessions if
    relevant.

    Args:
        enabled: ``True`` to allow proactive recall; ``False`` to stop
            proactively mentioning past sessions or saved memories.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    context.disallow_interruptions()
    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    await set_memory_recall(
        store,
        owner_id=userdata.user_id,
        enabled=enabled,
    )
    userdata.proactive_recall_enabled = enabled
    state_text = "on" if enabled else "off"
    await emit_voice_activity(
        context,
        activity="memory_recall_updated",
        status="completed",
        label="Memory recall updated",
        detail=f"Proactive recall is now {state_text}.",
    )
    return (
        f"I turned proactive recall {state_text}. "
        "Your saved style preferences can still shape how I respond, but I "
        f"{'may' if enabled else 'will not'} proactively bring up past sessions."
    )


@function_tool()
async def prepare_memory_deletion(
    context: RunContext[SessionData],
    query: str,
) -> str:
    """Find a saved memory to delete, but do not delete it yet.

    Call this when the user asks you to forget, delete, or remove a saved
    memory using a natural-language description. After this tool selects
    a target, wait for the user to confirm before calling
    ``confirm_memory_deletion``.

    Args:
        query: The memory topic the user wants deleted.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    query = query.strip()
    if not query:
        return "What saved memory would you like me to look for?"

    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    targets = await find_memory_targets(
        store,
        owner_id=userdata.user_id,
        query=query,
    )
    userdata.pending_memory_delete = None
    userdata.pending_memory_delete_candidates = []

    if not targets:
        return "I couldn't find a saved memory matching that."
    if len(targets) > 1:
        userdata.pending_memory_delete_candidates = targets
        await emit_voice_activity(
            context,
            activity="memory_delete_pending",
            status="pending",
            label="Memory deletion needs selection",
            detail="Choose which saved memory to delete.",
        )
        return _multiple_delete_matches_reply(targets)

    userdata.pending_memory_delete = targets[0]
    await emit_voice_activity(
        context,
        activity="memory_delete_pending",
        status="pending",
        label="Memory deletion pending",
        detail="Waiting for confirmation before deleting.",
    )
    return _pending_delete_reply(targets[0])


@function_tool()
async def prepare_indexed_memory_deletion(
    context: RunContext[SessionData],
    kind: str,
    index: int,
) -> str:
    """Find a saved memory by displayed kind and number, but do not delete it yet.

    Call this when the user asks to forget a numbered item, such as
    "forget fact number 2" or "delete rule 1." After this tool selects
    a target, wait for confirmation before calling
    ``confirm_memory_deletion``.

    Args:
        kind: One of ``fact``, ``session``, or ``rule``.
        index: The 1-based item number the user named.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    normalized_kind = kind.strip().lower()
    if normalized_kind not in {"fact", "session", "rule"}:
        return "I can delete saved facts, session summaries, or style preferences."

    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    target = await find_memory_target_by_index(
        store,
        owner_id=userdata.user_id,
        kind=normalized_kind,  # type: ignore[arg-type]
        index_1based=index,
    )
    userdata.pending_memory_delete = target
    userdata.pending_memory_delete_candidates = []

    if target is None:
        return f"I couldn't find saved {normalized_kind} #{index}."
    await emit_voice_activity(
        context,
        activity="memory_delete_pending",
        status="pending",
        label="Memory deletion pending",
        detail="Waiting for confirmation before deleting.",
    )
    return _pending_delete_reply(target)


@function_tool()
async def select_memory_deletion_candidate(
    context: RunContext[SessionData],
    candidate_number: int,
) -> str:
    """Select one candidate from the previous deletion disambiguation list.

    Call this after ``prepare_memory_deletion`` returned multiple possible
    matches and the user chooses a number. Wait for confirmation before
    calling ``confirm_memory_deletion``.

    Args:
        candidate_number: The 1-based candidate number the user selected.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    candidates = userdata.pending_memory_delete_candidates
    if candidate_number < 1 or candidate_number > len(candidates):
        return (
            "I don't have that memory option pending. "
            "Which saved memory should I delete?"
        )

    target = candidates[candidate_number - 1]
    userdata.pending_memory_delete = target
    userdata.pending_memory_delete_candidates = []
    await emit_voice_activity(
        context,
        activity="memory_delete_pending",
        status="pending",
        label="Memory deletion pending",
        detail="Waiting for confirmation before deleting.",
    )
    return _pending_delete_reply(target)


@function_tool()
async def confirm_memory_deletion(context: RunContext[SessionData]) -> str:
    """Delete the currently pending saved memory after explicit confirmation.

    Call this only after a prior deletion tool selected a memory and the
    user clearly confirmed with yes, confirm, or delete it.
    """

    userdata = context.userdata
    error = _memory_access_error(userdata)
    if error is not None:
        return error

    target = userdata.pending_memory_delete
    if target is None:
        userdata.pending_memory_delete_candidates = []
        return "There isn't a pending memory deletion to confirm."

    context.disallow_interruptions()
    store = userdata.memory_store
    if store is None:
        return "Memory is not available in this session."
    deleted = await delete_memory_target(
        store,
        owner_id=userdata.user_id,
        target=target,
    )
    kind = target["kind"]
    userdata.pending_memory_delete = None
    userdata.pending_memory_delete_candidates = []
    await emit_voice_activity(
        context,
        activity="memory_deleted",
        status="completed" if deleted else "failed",
        label="Memory deleted" if deleted else "Memory deletion failed",
        detail=(
            "Removed the selected saved memory."
            if deleted
            else "The selected memory was already gone."
        ),
    )
    return (
        f"Deleted that saved {kind}."
        if deleted
        else "I couldn't delete that memory because it was already gone."
    )


@function_tool()
async def cancel_memory_deletion(context: RunContext[SessionData]) -> str:
    """Cancel the currently pending saved-memory deletion.

    Call this when the user says no, cancel, never mind, or otherwise
    declines a pending memory deletion.
    """

    context.userdata.pending_memory_delete = None
    context.userdata.pending_memory_delete_candidates = []
    await emit_voice_activity(
        context,
        activity="memory_delete_pending",
        status="cancelled",
        label="Memory deletion cancelled",
        detail="No saved memory was changed.",
    )
    return "Cancelled. I didn't change your memory."


@function_tool()
async def crisis_check(
    context: RunContext[SessionData],
    concern: str,
) -> str:
    """Assess whether the user may be in crisis and needs immediate support.

    Call this when the user expresses hopelessness, self-harm thoughts,
    suicidal ideation, or indicates they may be in danger.  Provide
    the specific statement or behaviour that triggered your concern.

    Args:
        concern: The user's statement or behaviour that raised concern.
    """
    userdata = context.userdata

    # Run deterministic check first (instant, no network).
    lowered = concern.lower()

    is_imminent = any(
        re.search(p, lowered, flags=re.IGNORECASE) for p in IMMINENT_PATTERNS
    )
    is_clear_self_harm = any(
        re.search(p, lowered, flags=re.IGNORECASE) for p in CLEAR_SELF_HARM_PATTERNS
    )
    is_ambiguous = any(
        re.search(p, lowered, flags=re.IGNORECASE) for p in AMBIGUOUS_PATTERNS
    )

    if is_imminent:
        userdata.crisis_level = 3
        userdata.max_crisis_level = max(userdata.max_crisis_level, 3)
        logger.warning(
            "crisis_check: IMMINENT RISK detected user=%s concern=%s",
            userdata.user_id,
            concern[:100],
        )
        from voice.livekit.agent import CrisisAgent, _copy_handoff_chat_ctx

        return (
            CrisisAgent(
                chat_ctx=_copy_handoff_chat_ctx(context.session.current_agent.chat_ctx)
            ),
            "Transferring to crisis support",
        )

    if is_clear_self_harm:
        userdata.crisis_level = 2
        userdata.max_crisis_level = max(userdata.max_crisis_level, 2)
        logger.warning(
            "crisis_check: clear self-harm detected user=%s concern=%s",
            userdata.user_id,
            concern[:100],
        )
        from voice.livekit.agent import CrisisAgent, _copy_handoff_chat_ctx

        return (
            CrisisAgent(
                chat_ctx=_copy_handoff_chat_ctx(context.session.current_agent.chat_ctx)
            ),
            "Transferring to crisis support",
        )

    if is_ambiguous:
        userdata.crisis_level = 1
        userdata.max_crisis_level = max(userdata.max_crisis_level, 1)
        logger.info(
            "crisis_check: ambiguous distress user=%s concern=%s",
            userdata.user_id,
            concern[:100],
        )
        return (
            "The user may be experiencing distress. "
            "Gently ask a clarifying question to understand whether "
            "they are having thoughts of self-harm, without being "
            "alarmist. If they confirm, call crisis_check again with "
            "their response."
        )

    # No crisis signal detected.
    userdata.crisis_level = 0
    return (
        "No immediate crisis signal detected. Continue the "
        "conversation with care and empathy."
    )
