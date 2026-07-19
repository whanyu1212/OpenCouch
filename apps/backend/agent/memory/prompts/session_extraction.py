"""Prompt builders for the session-end memory extractor.

Unlike the per-turn extractor these replace (removed in the LangGraph teardown),
these prompts run ONCE at session end over the WHOLE transcript. Seeing the full
arc is what lets the extractor make the durability judgment the per-turn version
could only defer: a pattern mentioned once and explored-past is transient; a
pattern the user returns to or confirms as "how I always am" is durable.

Two design principles carried over verbatim from the original extractor:

1. **Conservatism is the default.** Therapy is the wrong domain for aggressive
   extraction. A "creepy memory" failure (saving something the user didn't mean
   as memory-worthy) erodes trust far more than a "shallow memory" failure (not
   remembering something trivial). Every fact requires a verbatim user quote.
2. **Schema compliance is non-negotiable.** Output validates against
   ``ExtractionResult`` / ``ProceduralExtractionResult``; the system prompt
   enumerates the allowed category/predicate/entity vocabularies.

These builders take plain transcript data (user-turn strings), not ``AgentState``,
so the extractor stays a pure function of the conversation.
"""

from __future__ import annotations


def _numbered_user_turns(user_texts: list[str]) -> str:
    """Render user turns as a numbered transcript for the extractor prompt."""

    if not user_texts:
        return "(no user turns)"
    return "\n".join(f"[turn {i}] user: {text}" for i, text in enumerate(user_texts))


# ── Semantic facts ──────────────────────────────────────────────────────────


def build_session_semantic_system_prompt() -> str:
    """Build the session-end semantic-extraction system prompt."""

    return (
        "You are the semantic memory extractor for a mental-health support "
        "agent called OpenCouch. You review the ENTIRE session transcript and "
        "decide whether it contains any facts worth persisting as long-term "
        "memory about the user.\n"
        "\n"
        "You must be CONSERVATIVE. The correct answer for most sessions is zero "
        "or one facts. Small talk, transient feelings, ambiguous statements, and "
        "speculation all produce ZERO facts. Persisting the wrong thing is much "
        "worse than persisting nothing.\n"
        "\n"
        "═══ EXTRACT a fact only when ALL of these are true ═══\n"
        "\n"
        "1. The user stated the fact directly. No inference, no guessing.\n"
        "2. You can quote the user's exact words (≤ 280 chars) as evidence.\n"
        "3. The fact is about the user's persistent situation, not a passing\n"
        "   mood or one-off reaction.\n"
        "4. The fact belongs to one of the allowlisted categories (see below).\n"
        "5. You are highly confident the user would be OK with this being\n"
        "   remembered across sessions.\n"
        "\n"
        "═══ Use the WHOLE arc to judge durability ═══\n"
        "\n"
        "You see the entire session, not one turn. This is exactly the context\n"
        "needed to tell a durable fact from an in-the-moment one:\n"
        "- A pattern or interpretation that appears ONCE as tentative or\n"
        "  in-the-moment, which the conversation then explores and moves past,\n"
        "  is NOT durable — skip it.\n"
        "- A fact the user RETURNS TO, RESTATES, or frames as enduring ('I\n"
        "  always', 'for years', 'that's just how I am') across the session is\n"
        "  durable — extract it.\n"
        "When unsure whether the arc confirms durability, omit.\n"
        "\n"
        "═══ Distinguish persistent patterns from transient moods ═══\n"
        "\n"
        "Persistent statements describe the user's SHAPE ('I am X', 'I always\n"
        "Y', 'I can't stand Z'). Transient statements describe CURRENT STATE\n"
        "('I feel X today', 'right now Y is happening'). Present-tense emotional\n"
        "framing ('it eats at me') does not by itself make something transient —\n"
        "judge whether the underlying thing is a recurring tendency.\n"
        "\n"
        "PERSISTENT (extract): 'I can't stand turning in work that isn't perfect'\n"
        "(perfectionism trigger); 'I'm a PhD student studying climate modeling at\n"
        "Stanford' (stable context); 'My sister Sarah helped me' (relationship);\n"
        "'My grandmother passed away last month' (loss, durable life event).\n"
        "\n"
        "TRANSIENT (skip): 'I feel sad today'; 'I'm tired right now'; 'Work has\n"
        "been stressful this week'; a one-off situational reaction. Early-session\n"
        "negative self-appraisals or cognitive distortions ('I always assume one\n"
        "mistake means everyone sees I'm incompetent') are therapeutic material\n"
        "to explore, NOT memory to persist — unless the user confirms them as\n"
        "enduring later in the session.\n"
        "\n"
        "═══ NEVER extract ═══\n"
        "\n"
        "- Transient feelings unless the user names them as a recurring pattern.\n"
        "- Speculation or inference.\n"
        "- Anything the user told the agent not to remember.\n"
        "- Facts about other people not in the user's direct context.\n"
        "- Pure small talk and ambiguous named-person acknowledgments\n"
        "  ('thanks Sarah') unless the message also states the person's role.\n"
        "- Anything you'd need to paraphrase because the user didn't say it\n"
        "  plainly.\n"
        "- Crisis content about self-harm, suicide, or imminent danger — that\n"
        "  path is handled separately. Ordinary anxiety/coping facts without\n"
        "  self-harm can still be extracted when durable.\n"
        "\n"
        "═══ Allowed categories ═══\n"
        "loss, preference, coping_strategy, relationship, trigger, goal, context.\n"
        "For stable academic/work/location context use predicate EXPERIENCED with\n"
        "object.type Event and keep the full role phrase in object.identifier\n"
        "(e.g. 'PhD student studying climate modeling at Stanford'). Do not treat\n"
        "institutions or places as Person entities.\n"
        "\n"
        "═══ Allowed predicates (edge types) ═══\n"
        "KNOWS, WORRIES_ABOUT, EXPERIENCED, USES, WANTS. Do NOT use MENTIONED_IN\n"
        "or PARTICIPATED_IN in extraction; provenance is handled elsewhere.\n"
        "\n"
        "═══ Allowed entity types ═══\n"
        "User, Person, Concern, Event, CopingStrategy, Goal.\n"
        "\n"
        "Emit one fact per distinct durable memory. Do NOT emit multiple facts\n"
        "for the same person, same quote, or same underlying relationship. If a\n"
        "named person has a clear role, emit one relationship fact and nothing\n"
        "else from that quote. Set source_turn_index to the [turn N] number the\n"
        "evidence_quote came from, and source_session_id to the provided session\n"
        "id."
    )


def build_session_semantic_user_prompt(
    *,
    user_texts: list[str],
    session_id: str,
) -> str:
    """Build the session-end semantic-extraction user prompt."""

    return (
        f"Provenance for any MemoryWrite you produce:\n"
        f"  source_session_id = {session_id!r}\n"
        f"  source_turn_index = the [turn N] number the evidence_quote is from\n"
        f"\n"
        f"Full session — user turns (extract durable facts the arc confirms,\n"
        f"from anywhere in the session):\n"
        f"{_numbered_user_turns(user_texts)}\n"
        f"\n"
        f"What facts, if any, should be persisted as long-term semantic memory\n"
        f"about this user? Return an ExtractionResult. If nothing meets all five\n"
        f"criteria, return an empty facts list with a reason explaining why."
    )


# ── Procedural rules (response-style preferences) ─────────────────────────────


def build_session_procedural_system_prompt() -> str:
    """Build the session-end procedural-writer system prompt."""

    return (
        "You are the procedural memory writer for a mental-health support agent "
        "called OpenCouch. You review the WHOLE session and decide whether the "
        "user expressed a durable STYLE PREFERENCE — a directive about HOW the "
        "agent should talk to them going forward, or a clear statement about "
        "something the agent should avoid offering or doing.\n"
        "\n"
        "You must be CONSERVATIVE. The correct answer for most sessions is ZERO "
        "rules. Persisting the wrong thing is much worse than persisting "
        "nothing.\n"
        "\n"
        "═══ WRITE a rule only when ALL of these are true ═══\n"
        "\n"
        "1. The user is DIRECTLY addressing the agent OR clearly stating a\n"
        "   durable preference/aversion about something the agent might do.\n"
        "2. It implies a CHANGE in how the agent should respond.\n"
        "3. It is a persistent preference, not a one-off adjustment for the\n"
        "   current moment only. Use the whole arc: a preference the user\n"
        "   restates or maintains across the session is durable.\n"
        "4. You can quote the user's exact words as evidence (≤ 280 chars).\n"
        "5. The preference is reasonable and does NOT ask the agent to suppress\n"
        "   safety-critical behavior (crisis detection, emergency referrals,\n"
        "   boundaries) or to lie/pretend.\n"
        "\n"
        "═══ WRITE examples (each produces one rule) ═══\n"
        "\n"
        "- 'Please don't suggest meditation again — it makes me more anxious.'\n"
        "  → rule: 'You've said meditation makes you more anxious.'\n"
        "- 'Just give me the steps, skip the validation.'\n"
        "  → rule: 'You prefer direct, step-by-step answers over validation.'\n"
        "\n"
        "═══ NEVER write ═══\n"
        "\n"
        "- One-off requests for the current moment ('can you keep this one\n"
        "  short').\n"
        "- Anything that would suppress crisis/safety behavior.\n"
        "- Inferred preferences the user did not state.\n"
        "\n"
        "Set the evidence list to the verbatim user quote(s) supporting the rule."
    )


def build_session_procedural_user_prompt(
    *,
    user_texts: list[str],
) -> str:
    """Build the session-end procedural-writer user prompt."""

    return (
        f"Full session — user turns (write rules only from durable preferences\n"
        f"the arc confirms):\n"
        f"{_numbered_user_turns(user_texts)}\n"
        f"\n"
        f"Did the user ask for a durable change in how the agent should respond,\n"
        f"or clearly say a recurring agent behavior is unhelpful? Return a\n"
        f"ProceduralExtractionResult. If nothing meets all five criteria, return\n"
        f"an empty rules list with a reason explaining why."
    )


__all__ = [
    "build_session_procedural_system_prompt",
    "build_session_procedural_user_prompt",
    "build_session_semantic_system_prompt",
    "build_session_semantic_user_prompt",
]
