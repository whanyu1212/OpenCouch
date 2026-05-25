"""Prompt policy for OpenCouch voice sessions."""

from __future__ import annotations


def build_voice_instructions(
    *,
    thread_id: str,
    user_id: str | None,
    memory_mode: str,
    memory_context: str | None = None,
) -> str:
    """Build compact instructions for one realtime voice session."""

    normalized_mode = memory_mode.strip().lower()
    identity = (
        "You are OpenCouch in a live speech-to-speech conversation. "
        "Keep responses concise, warm, grounded, and natural for spoken audio."
    )
    tool_policy = (
        "Use tools only when they are needed for app-owned state, memory, "
        "crisis resources, or guided exercises. For noticeable tool work, say "
        "a short preamble before waiting."
    )
    crisis_policy = (
        "Use lookup_crisis_resources as the only source for specific crisis "
        "resource names, phone numbers, URLs, or local availability. Do not "
        "invent crisis resources."
    )
    exercise_policy = (
        "For guided exercises, use runtime-selected exercise skill IDs and "
        "loaded skill context. Do not default to 5-4-3-2-1 grounding unless "
        "that exact runtime-selected skill is provided."
    )

    if normalized_mode == "incognito":
        memory_policy = (
            "This is incognito mode. Do not save durable memory. Do not claim "
            "to remember this conversation later. You may use transient session "
            "context only."
        )
        recall_policy = ""
    else:
        memory_policy = (
            "This is persistent mode. Durable memory may be read or changed "
            "only through the exposed memory tools. Save preferences only when "
            "the user explicitly asks you to remember a response or memory-use "
            "preference."
        )
        recall_policy = (
            "When the user mentions a topic that might have prior saved "
            "context (an ongoing concern, a relationship, a past exercise), "
            "call recall_saved_memory with a short topic query before "
            "responding. Do not call it every turn; only when a specific "
            "topic surfaces. The server refuses the call in incognito mode "
            "or when the user has proactive recall disabled — honor the "
            "refusal silently."
        )

    session_context = (
        f"Session metadata: thread_id={thread_id}; "
        f"user_scope={'persistent' if user_id else 'guest'}; "
        f"memory_mode={normalized_mode}."
    )
    blocks = [
        identity,
        session_context,
        memory_policy,
    ]
    if recall_policy:
        blocks.append(recall_policy)
    blocks.extend([tool_policy, crisis_policy, exercise_policy])
    if normalized_mode == "persistent" and memory_context:
        blocks.append(
            "Private saved-memory context for this session. Use it only when "
            "relevant and do not recite it verbatim.\n"
            f"{memory_context.strip()}"
        )
    return "\n\n".join(blocks)
