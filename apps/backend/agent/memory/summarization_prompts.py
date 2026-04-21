"""Prompt builders for the session summarizer node.

The summarizer runs ONCE per session at session end (triggered by `/end`
or a `/exit` confirmation in the CLI). It reads the full transcript and
produces a single :class:`SessionArc` — a structured narrative summary
with concerns touched, open loops, resolved threads, mood arc, and a
short summary paragraph.

Three design principles shape these prompts:

1. **Narrative, not comprehensive.** The summarizer should capture the
   emotional arc and the main concerns, not transcribe every turn. A
   good summary is something the user would recognize if they read it
   days later — "yes, that's what we talked about" — not a minute-by-
   minute log. The system prompt tells the LLM explicitly to paraphrase
   rather than quote at length.

2. **Conservative on primary_themes.** The allowlist of themes is
   intentionally loose ("work stress", "grief", "sleep", "family") to
   match real therapy vocabulary, but the schema caps the list at 3
   entries. The prompt tells the LLM to pick the 1-2 dominant themes,
   leaving 3 for the rare multi-topic session.

3. **None is a valid outcome.** Very short sessions, sessions with only
   small talk, or sessions that didn't go anywhere emotionally should
   return ``arc=None`` with a reason. Summarizing an empty conversation
   produces false memories that erode trust — silence is safer.

The prompts are **tightly coupled to ``SessionArc`` and
``SummarizationResult``**. Any schema change requires a corresponding
prompt change; that cohesion is why the prompts live next to the models
in the memory package rather than in ``knowledge/``.
"""

from __future__ import annotations

from agent.state import AgentState


# ─── Per-modality field hints for the summarizer user prompt ─────────────────
#
# Each entry lists the approach_context field names the LLM should look for
# in the transcript, plus a one-line description of what each field captures.
# Only the relevant modality's hints are injected into the user prompt,
# keeping the system prompt generic and avoiding prompt bloat.

_MODALITY_CONTEXT_HINTS: dict[str, str] = {
    "cbt": (
        "thought_examined (the specific belief or prediction examined),\n"
        "    action_step (concrete next step the user agreed to try),\n"
        "    tool_used (one of: thought_record, behavioral_experiment, activation)"
    ),
    "motivational_interviewing": (
        "readiness_stage (precontemplation / contemplation / preparation / action),\n"
        "    change_talk_themes (user's own reasons for wanting change),\n"
        "    sustain_talk_themes (user's reasons for staying where they are)"
    ),
    "act": (
        "values_identified (life domains or values the user named as important),\n"
        "    fusion_patterns (thoughts/feelings the user was stuck on),\n"
        "    committed_action (values-aligned step the user agreed to try)"
    ),
    "grief_support": (
        "person_lost (name or role of the person lost, if shared),\n"
        "    relationship (nature of the relationship),\n"
        "    time_since_loss (how long ago, in the user's words)"
    ),
    "interpersonal_therapy": (
        "problem_area (grief / role_transition / role_dispute / isolation),\n"
        "    key_relationship (the person or relationship at the center),\n"
        "    communication_step_planned (relational action the user planned)"
    ),
    "dbt_skills": (
        "skills_used (specific DBT skills practiced),\n"
        "    primary_domain (distress_tolerance / emotion_regulation / interpersonal_effectiveness)"
    ),
    "pfa": (
        "crisis_type (what the acute distress was about),\n"
        "    support_connected (resource or person the user was linked to)"
    ),
}


def build_summarization_system_prompt() -> str:
    """Build the system prompt for the session summarizer LLM call.

    The prompt emphasizes:
    - Narrative over transcript
    - Conservative None-return when the session lacks emotional content
    - Allowlisted mood descriptors and crisis-level honesty
    - Strict length caps (summary ≤ 600 chars; ~3-4 sentences)

    Returns:
        The full system prompt as a single string, ready to pass to
        ``generate_structured(system_instruction=...)``.
    """

    return (
        "You are the session summarizer for a mental-health support agent "
        "called OpenCouch. Your only job is to read a completed conversation "
        "and produce ONE structured narrative summary that captures what "
        "the user talked about and how they felt across the session.\n"
        "\n"
        "This runs ONCE per session at session end. The summary becomes "
        "long-term episodic memory — the next time the user starts a "
        "session, this summary will be surfaced as 'last time we talked'. "
        "Write it so that the user would recognize it if they read it "
        "days later.\n"
        "\n"
        "═══ When to produce a SessionArc ═══\n"
        "\n"
        "Return a SessionArc (arc is NOT None) when the session had:\n"
        "- At least one substantive emotional or topical thread, AND\n"
        "- Enough content to paraphrase into 2-4 sentences meaningfully.\n"
        "\n"
        "Return ``arc=None`` when:\n"
        "- The session was only small talk or a capability question.\n"
        "- Fewer than 3 user turns with substantive content.\n"
        "- The user only tested the system or asked abstract questions\n"
        "  without disclosing anything about themselves or their state.\n"
        "- The conversation didn't develop any meaningful emotional or\n"
        "  topical arc, even if the turn count is high.\n"
        "\n"
        "If you're unsure, err toward ``arc=None`` and a reason like\n"
        "'session had turns but no meaningful arc to summarize'. A\n"
        "missing summary is better than a fabricated one.\n"
        "\n"
        "═══ What goes in the summary field ═══\n"
        "\n"
        "- 2-4 sentences (≤ 600 chars total). Narrative prose, not bullets.\n"
        "- Paraphrase the user's words; do NOT quote at length.\n"
        "- Describe what the user talked about AND how they felt, not just\n"
        "  one or the other.\n"
        "- Write in third person referring to 'the user' or 'you' —\n"
        "  whichever sounds more natural to read back later. Avoid\n"
        "  'I' (the assistant's voice should not appear in the summary).\n"
        "- Focus on CONTENT, not process. Don't write 'the user asked\n"
        "  clarifying questions about X' — write 'the user was trying to\n"
        "  understand X'.\n"
        "\n"
        "═══ primary_themes ═══\n"
        "\n"
        "Pick 1-3 short theme tags that describe the session's main topics.\n"
        "Keep them short (1-3 words each) and descriptive. Examples:\n"
        "  ['work stress'], ['grief', 'sleep'], ['relationship conflict'],\n"
        "  ['exam anxiety', 'perfectionism']\n"
        "\n"
        "Prefer existing therapy-adjacent vocabulary over novel tags. If\n"
        "the session has no dominant theme (e.g., pure check-in), return\n"
        "an empty list [].\n"
        "\n"
        "═══ mood_arc ═══\n"
        "\n"
        "Two short descriptor strings (≤ 40 chars each):\n"
        "- ``opened``: how the user sounded at the start of the session.\n"
        "- ``closed``: how the user sounded at the end of the session.\n"
        "\n"
        "Use plain, recognizable descriptors like:\n"
        "  'anxious', 'overwhelmed', 'flat', 'tentatively_okay',\n"
        "  'calmer', 'more grounded', 'still anxious', 'unresolved'\n"
        "\n"
        "It is OK for opened and closed to be the same (e.g., the session\n"
        "didn't shift anything). That honest answer is more useful than a\n"
        "fabricated change.\n"
        "\n"
        "═══ open_loops and resolved_threads ═══\n"
        "\n"
        "- ``open_loops``: things the user raised that were NOT resolved by\n"
        "  session end. Short phrases (≤ 80 chars each). Examples:\n"
        "    ['still hasn't prepped the Monday meeting slides',\n"
        "     'unresolved feelings about the supervisor feedback']\n"
        "- ``resolved_threads``: things the user raised that DID resolve or\n"
        "  were explicitly set aside. Same format.\n"
        "\n"
        "If the session ended cleanly (user said goodbye or thanked the\n"
        "assistant without naming loose ends), both lists may be empty.\n"
        "\n"
        "═══ Modality context ═══\n"
        "\n"
        "If a dominant modality hint is provided in the user prompt,\n"
        "populate ``approach_used`` with the modality name and\n"
        "``approach_context`` with the matching typed schema.\n"
        "\n"
        "Rules:\n"
        "- Only populate fields where the conversation clearly produced\n"
        "  that artifact. When unsure, leave fields as null.\n"
        "- The user prompt lists the specific fields to extract for the\n"
        "  active modality. Do NOT guess fields for other modalities.\n"
        "- If no modality hint is provided, or the session did not\n"
        "  engage in structured therapeutic work, set both\n"
        "  ``approach_used`` and ``approach_context`` to null.\n"
        "- For PFA sessions: do NOT reinterpret crisis severity in\n"
        "  approach_context. Crisis severity is tracked separately.\n"
        "\n"
        "═══ What you do NOT decide ═══\n"
        "\n"
        "You do NOT produce a crisis level, a distress score, or any other\n"
        "numeric severity rating. The runtime tracks crisis severity via\n"
        "the per-turn crisis gate (a separate node that runs on every\n"
        "message) and computes the session-level peak deterministically.\n"
        "Your mood_arc descriptors are the place to express how the user\n"
        "sounded — do that work there, not as a number. If you notice\n"
        "something crisis-adjacent that the mood_arc doesn't capture,\n"
        "mention it in the ``summary`` prose; do NOT try to add a\n"
        "numeric field to your output.\n"
        "\n"
        "═══ Language ═══\n"
        "\n"
        "Write every text field (summary, mood_arc.opened, mood_arc.closed,\n"
        "primary_themes, open_loops, resolved_threads) in the SAME language\n"
        "as the user's messages in the transcript. Do NOT introduce a\n"
        "second language, code-switch mid-string, or drop foreign-language\n"
        "words into a summary rendered in another language — the CLI and\n"
        "future catch-up retrieval display these fields verbatim, and\n"
        "mixed-language strings look like a glitch to the user reading\n"
        "them days later. If the transcript itself mixes languages, pick\n"
        "the dominant one and stay in it for all fields.\n"
        "\n"
        "═══ session_id, started_at, ended_at, duration_seconds, turn_count ═══\n"
        "\n"
        "These are provided in the user prompt as provenance metadata.\n"
        "Copy them verbatim into the SessionArc — do NOT infer or guess.\n"
        "\n"
        "═══ Output shape ═══\n"
        "\n"
        "Return a SummarizationResult with two fields:\n"
        "\n"
        "  arc    : a SessionArc if the session had a summarizable arc,\n"
        "           OR ``None`` if the session was too thin / too short /\n"
        "           too empty to summarize meaningfully.\n"
        "\n"
        "  reason : a short one-sentence explanation. Always populate this.\n"
        "           When arc is populated: a summary-of-your-summary\n"
        "           (e.g., 'captured 8-turn work anxiety arc with 2 open\n"
        "           loops'). When arc is None: a reason (e.g., 'only 2\n"
        "           substantive turns, no arc').\n"
        "\n"
        "When in doubt, prefer shorter summaries, fewer themes, and\n"
        "``arc=None``. The cost of over-summarizing (fabricated details,\n"
        "misremembered focus) is higher than the cost of under-summarizing\n"
        "(user says 'oh, we also talked about X')."
    )


def build_summarization_user_prompt(
    state: AgentState,
    *,
    session_id: str,
    started_at: str,
    ended_at: str,
    duration_seconds: int,
    turn_count: int,
    approach_hint: str | None = None,
) -> str:
    """Build the user/task prompt for a single session-summarization call.

    Injects the full session transcript plus the provenance metadata the
    summarizer needs to populate :class:`SessionArc` fields that aren't
    inferred from the conversation itself (session_id, timestamps,
    turn_count, duration).

    Unlike ``build_extraction_user_prompt`` which uses only the last 6
    history turns, this prompt injects the ENTIRE transcript. That's
    deliberate: the summarizer needs to see the arc of the whole session
    to produce an opened/closed mood pair and a narrative that captures
    the actual trajectory. For very long sessions (50+ turns) the prompt
    will be large, but summarization runs once per session so the cost
    is bounded. When long sessions become common, a future refinement
    could truncate the middle of the transcript.

    Args:
        state: Current graph state. Reads ``transcript`` (the full
            session history, including both user and assistant turns).
        session_id: The session identifier, copied verbatim into the
            SessionArc's ``session_id`` field.
        started_at: ISO-8601 timestamp when the session started.
        ended_at: ISO-8601 timestamp when the session is ending (now).
        duration_seconds: Duration of the session in seconds.
        turn_count: Total number of user turns in the session.
        approach_hint: The dominant therapeutic modality used during the
            session (e.g., "cbt", "act"). When provided, the prompt
            instructs the LLM to extract modality-specific structured
            context. When None, approach_used and approach_context
            should be left null.
    """

    transcript = state.get("transcript", [])
    if not transcript:
        transcript_block = "(empty transcript — the session had no turns)"
    else:
        transcript_block = "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
            for turn in transcript
            if turn.get("content")
        )

    # Build modality extraction block — only for sessions with a clear modality.
    modality_block = ""
    if approach_hint and approach_hint != "none":
        field_hints = _MODALITY_CONTEXT_HINTS.get(approach_hint)
        if field_hints:
            modality_block = (
                f"\n"
                f"Dominant modality this session: {approach_hint}\n"
                f"Set approach_used = {approach_hint!r} and populate\n"
                f"approach_context with these fields (only where the\n"
                f"transcript clearly shows the artifact — leave null\n"
                f"otherwise):\n"
                f"    {field_hints}\n"
            )

    return (
        f"Provenance metadata (copy these into the SessionArc verbatim;\n"
        f"do NOT infer or guess):\n"
        f"  session_id        = {session_id!r}\n"
        f"  started_at        = {started_at!r}\n"
        f"  ended_at          = {ended_at!r}\n"
        f"  duration_seconds  = {duration_seconds}\n"
        f"  turn_count        = {turn_count}\n"
        f"{modality_block}\n"
        f"Full session transcript (every user and assistant turn, in\n"
        f"chronological order):\n"
        f"{transcript_block}\n"
        f"\n"
        f"Produce a SummarizationResult. If this session had at least one\n"
        f"substantive thread and enough content to paraphrase into a few\n"
        f"sentences, return a SessionArc. If the session was only small\n"
        f"talk, a capability check, or otherwise too thin to summarize\n"
        f"meaningfully, return ``arc=None`` with a reason explaining why."
    )
