"""Prompt builders for the semantic extraction node.

The extraction node runs after each response node and calls a structured-
output LLM with these prompts to produce zero or more :class:`MemoryWrite`
items. The system prompt enforces a deliberately conservative stance:
most turns should produce zero extractions, and the extractor should
prefer silence to speculation.

Two design principles shape these prompts:

1. **Conservatism is the default.** Therapy is the wrong domain for
   aggressive extraction. A "creepy memory" failure (the agent saving
   something the user didn't mean as memory-worthy) erodes trust far
   more than a "shallow memory" failure (the agent not remembering
   something trivial). The prompt explicitly lists what NOT to extract
   and requires a direct user quote as evidence for every fact.

2. **Schema compliance is non-negotiable.** The extractor's output is
   validated against :class:`ExtractionResult`, which wraps a list of
   :class:`MemoryWrite`. Each MemoryWrite has a restricted vocabulary
   for category, predicate (HotPathEdgeType), and entity types. The
   system prompt enumerates these to minimize the chance of the LLM
   guessing a value outside the allowlist.
"""

from __future__ import annotations

from agent.state import AgentState


def build_extraction_system_prompt() -> str:
    """Build the system prompt for the semantic extraction LLM call.

    Kept inline (not loaded from a knowledge markdown file) because the
    prompt is tightly coupled to the ``MemoryWrite``/``ExtractionResult``
    schemas and any schema change requires a corresponding prompt
    change. Cohesion beats the slight duplication with knowledge files.

    Returns:
        The full system prompt as a single string, ready to pass to
        ``generate_structured(system_instruction=...)``.
    """

    return (
        "You are the semantic memory extractor for a mental-health support "
        "agent called OpenCouch. Your only job is to look at the most recent "
        "user turn and decide whether it contains any facts worth persisting "
        "as long-term memory about the user.\n"
        "\n"
        "You must be CONSERVATIVE. The correct answer for most turns is zero "
        "facts. Small talk, transient feelings, ambiguous statements, and "
        "speculation all produce ZERO facts. Persisting the wrong thing is "
        "much worse than persisting nothing.\n"
        "\n"
        "═══ EXTRACT a fact only when ALL of these are true ═══\n"
        "\n"
        "1. The user stated the fact directly. No inference, no guessing.\n"
        "2. You can quote the user's exact words (≤ 280 chars) as evidence.\n"
        "3. The fact is about the user's persistent situation, not a\n"
        "   passing mood or a one-off reaction.\n"
        "4. The fact belongs to one of the allowlisted categories (see below).\n"
        "5. You are highly confident the user would be OK with this being\n"
        "   remembered across sessions.\n"
        "\n"
        "═══ NEVER extract ═══\n"
        "\n"
        '- Transient feelings ("I feel sad today", "I\'m tired") unless\n'
        "  the user explicitly names them as a recurring pattern.\n"
        '- Speculation or inference ("It sounds like the user might be...")\n'
        "- Anything the user told the agent not to remember.\n"
        "- Facts about other people that aren't in the user's direct context.\n"
        '- Small talk ("thanks", "ok", "how are you").\n'
        "- Anything you'd need to paraphrase because the user didn't say it\n"
        "  plainly.\n"
        "- Crisis-related content — that path is handled separately by the\n"
        "  crisis log node.\n"
        "\n"
        "═══ Allowed categories ═══\n"
        "\n"
        "- loss            : bereavement, breakups, job loss, major life ruptures.\n"
        "- preference      : things the user likes or dislikes (food, activities,\n"
        "                    ways of being supported).\n"
        "- coping_strategy : techniques the user uses or has tried (grounding,\n"
        "                    journaling, therapy, exercise).\n"
        "- relationship    : named people in the user's life (family, partners,\n"
        "                    friends, colleagues).\n"
        "- trigger         : things that exacerbate concerns or distress.\n"
        "- goal            : things the user wants to achieve or work on.\n"
        "- context         : everything else (job, location, schedule, living\n"
        "                    situation) that's stable and user-stated.\n"
        "\n"
        "═══ Allowed predicates (edge types) ═══\n"
        "\n"
        "- KNOWS           : user knows a named person.\n"
        "- WORRIES_ABOUT   : user has a recurring concern.\n"
        "- EXPERIENCED     : user went through a specific event.\n"
        "- USES            : user applies a coping strategy.\n"
        "- WANTS           : user has a stated goal.\n"
        "- PARTICIPATED_IN : user was in a specific session (rare for extraction).\n"
        "- MENTIONED_IN    : provenance edge (rare for extraction).\n"
        "\n"
        "═══ Allowed entity types ═══\n"
        "\n"
        "- User, Person, Concern, Event, CopingStrategy, Goal, Session, Turn\n"
        "\n"
        "═══ Output shape ═══\n"
        "\n"
        "Return an ExtractionResult with two fields:\n"
        "\n"
        "  facts  : a list of zero or more MemoryWrite items. Empty list is\n"
        "           the common case. Each MemoryWrite must have:\n"
        "             - category          (one of the allowed categories)\n"
        "             - subject           (EntityRef with type + identifier)\n"
        "             - predicate         (one of the allowed predicates)\n"
        "             - object            (EntityRef with type + identifier)\n"
        "             - evidence_quote    (verbatim user words, ≤ 280 chars)\n"
        "             - confidence        ('low', 'medium', or 'high')\n"
        "             - source_session_id (provided in the user prompt)\n"
        "             - source_turn_index (provided in the user prompt)\n"
        "\n"
        "  reason : a short one-sentence explanation. Always populate this.\n"
        "           When facts is empty: explain why nothing was extracted\n"
        "           (e.g., 'small talk, no extractable facts'). When facts\n"
        "           is non-empty: summarize what was extracted (e.g.,\n"
        "           'extracted 2 relationship facts about user's sister').\n"
        "\n"
        "If you are unsure whether a fact meets all five extraction criteria,\n"
        "OMIT it. Silence is always the safer choice."
    )


def build_extraction_user_prompt(
    state: AgentState,
    *,
    turn_index: int,
) -> str:
    """Build the user/task prompt for a single extraction call.

    Injects the current user message, a short window of recent history
    for context, and the session id + turn index so the extractor can
    populate the provenance fields of each :class:`MemoryWrite`.

    Args:
        state: Current graph state. Reads ``message``, ``history``,
            ``session_id``.
        turn_index: Zero-based index of the current turn within the
            session. The extractor writes this into each MemoryWrite's
            ``source_turn_index`` field.
    """

    history = state.get("history", [])[-6:]
    history_block = (
        "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
            for turn in history
            if turn.get("content")
        )
        or "(no prior history)"
    )

    session_id = state.get("session_id") or "__no_session__"
    current_message = state["message"]

    return (
        f"Provenance metadata for any MemoryWrite you produce:\n"
        f"  source_session_id = {session_id!r}\n"
        f"  source_turn_index = {turn_index}\n"
        f"\n"
        f"Recent conversation (for context — do NOT extract from history;\n"
        f"only from the current user message):\n"
        f"{history_block}\n"
        f"\n"
        f"Current user message (extract from THIS only):\n"
        f"user: {current_message}\n"
        f"\n"
        f"What facts, if any, should be persisted as long-term semantic\n"
        f"memory about this user? Return an ExtractionResult. If nothing\n"
        f"in the current message meets all five extraction criteria from\n"
        f"the system prompt, return an empty facts list with a reason\n"
        f"explaining why."
    )
