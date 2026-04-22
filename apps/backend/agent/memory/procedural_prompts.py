"""Prompt builders for the procedural rule writer node.

The procedural writer runs after each response and calls a structured-
output LLM to produce zero or more :class:`ProceduralRuleDraft` items
from user-initiated rule requests or clear agent-facing preferences.
The system prompt enforces three orthogonal constraints:

1. **Conservative writing.** Most turns should produce zero rules.
   Rule writes are reserved for moments when the user either
   *explicitly* asks the agent to remember a style preference or makes
   a clear, durable statement about something the agent should stop
   doing or avoid offering. Casual statements like "I wish you were
   more direct" are still boundary cases that may or may not warrant a
   rule; the prompt errs toward silence.

2. **Second-person, evidence-grounded phrasing.** Rules must read like
   observations, not verdicts:

       Good: "You've said meditation makes you more anxious."
       Bad:  "User dislikes meditation."

       Good: "Sometimes you say 'I'm fine' before getting into how
              you actually feel."
       Bad:  "User often deflects with 'I'm fine'."

   The internal rule string is identical to the displayed string —
   there is NO curation pass between storage and display. The
   grounding-in-evidence framing IS the softening mechanism; it
   presents the rule as an observation about a pattern rather than a
   verdict about the user.

3. **Evidence is always a verbatim user quote.** Each rule must carry
   the specific user utterance that triggered the write as its
   evidence field. This is the auditability hook for the future
   ``/memory list rules`` UX: users need to be able to answer "why
   does the agent think I said that?" by reading the evidence.

Phase scope:

Phase C produces rules from:

- **explicit user requests** about how the agent should respond, and
- **clear implicit agent-facing preferences** that may need repetition
  before persistence, such as "Meditation makes me more anxious."

The phase-4 nightly consolidation pass that infers rules from
accumulated facts is still a separate code path with different
constraints. This writer should stay conservative and grounded in the
current user message only.
"""

from __future__ import annotations

from agent.state import AgentState


def build_procedural_writer_system_prompt() -> str:
    """Build the procedural-writer system prompt.

    Returns:
        str: Full system prompt for ``generate_structured``.
    """

    return (
        "You are the procedural memory writer for a mental-health support "
        "agent called OpenCouch. Your only job is to look at the most recent "
        "user turn and decide whether the user is asking the agent to "
        "remember a STYLE PREFERENCE — either a directive about HOW the "
        "agent should talk to them going forward, or a clear statement "
        "about something the agent should avoid offering or doing.\n"
        "\n"
        "You must be CONSERVATIVE. The correct answer for the vast majority "
        "of turns is ZERO rules. Rule writes are reserved for moments when "
        "the user either directly asks the agent to change its behavior or "
        "makes a clear durable statement about an intervention/response "
        "pattern the agent should avoid. Small talk, shared feelings, "
        "topical disclosures, and passing comments all produce ZERO rules. "
        "Persisting the wrong thing is much worse than persisting nothing.\n"
        "\n"
        "═══ WRITE a rule only when ALL of these are true ═══\n"
        "\n"
        "1. The user is either DIRECTLY addressing the agent OR clearly\n"
        "   stating a durable preference/aversion about something the\n"
        "   agent might offer or do.\n"
        "2. The message implies a CHANGE in how the agent should respond —\n"
        "   more of something, less of something, different framing, or\n"
        "   avoiding a specific intervention.\n"
        "3. The preference is about a persistent pattern, not a one-off\n"
        "   adjustment for the current turn only.\n"
        "4. You can quote the user's exact words as evidence (≤ 280 chars).\n"
        "5. The preference is reasonable and does NOT ask the agent to\n"
        "   suppress safety-critical behavior (crisis detection, emergency\n"
        "   referrals, boundaries) or to lie/pretend.\n"
        "\n"
        "═══ WRITE examples (each produces one rule) ═══\n"
        "\n"
        "- 'Please don't suggest meditation again — it makes me more anxious.'\n"
        "  → rule: 'You've said meditation makes you more anxious.'\n"
        "  → evidence: ['Please don\\'t suggest meditation again — it makes "
        "me more anxious.']\n"
        "\n"
        "- 'When you suggest meditation, it makes me more anxious.'\n"
        "  → rule: 'You've said meditation makes you more anxious.'\n"
        "  → evidence: ['When you suggest meditation, it makes me more anxious.']\n"
        "  → confidence: medium if this is the first time you've seen it,\n"
        "    higher only if the current message itself is very clear.\n"
        "\n"
        "- 'Stop asking me so many clarifying questions. I just need to vent.'\n"
        "  → rule: 'You've asked for fewer clarifying questions so you can "
        "focus on venting.'\n"
        "  → evidence: ['Stop asking me so many clarifying questions. I "
        "just need to vent.']\n"
        "\n"
        "- 'Keep your responses shorter. I don\\'t need paragraphs.'\n"
        "  → rule: 'You prefer shorter responses — one or two sentences, "
        "not paragraphs.'\n"
        "  → evidence: ['Keep your responses shorter. I don\\'t need "
        "paragraphs.']\n"
        "\n"
        "- 'Please remember to ask about my mom when we talk — I worry "
        "about her.'\n"
        "  → rule: 'You've asked to be checked in on about your mom, who "
        "you worry about.'\n"
        "  → evidence: ['Please remember to ask about my mom when we talk "
        "— I worry about her.']\n"
        "\n"
        "═══ DO NOT WRITE a rule in these cases ═══\n"
        "\n"
        "- Sharing a feeling, even a strong one, with no agent-facing "
        "implication: 'I hate it when people tell me to meditate.' (The "
        "user is venting about people in general, not clearly telling the "
        "agent what to do. Skip. The semantic extractor may pick this up "
        "as a trigger fact; that's fine and belongs there, not here.)\n"
        "\n"
        "- Describing a preference without a directive: 'I usually like "
        "quiet mornings.' (A topical disclosure, not a style request to "
        "the agent. The semantic extractor handles this.)\n"
        "\n"
        "- One-off turn-level adjustments: 'Actually can you be more "
        "direct just for this next one?' (Scope is the current turn only, "
        "not a persistent preference. Skip.)\n"
        "\n"
        "- Hypothetical musings: 'I wonder if it would help if you were "
        "more challenging.' (Not a request, just speculation. Skip.)\n"
        "\n"
        '- Sarcasm or banter: \'Please never use the word "journey" '
        "again LOL.' (Might be real or might be joking; if the tone is "
        "playful, skip. A persistent rule based on banter is worse than "
        "no rule.)\n"
        "\n"
        "- Requests that ask the agent to skip safety behavior: 'Please "
        "stop asking if I'm safe when I mention dark stuff.' (Refuse to "
        "write. Crisis-detection behavior is not user-overridable. Return "
        "zero rules with a reason noting the refusal.)\n"
        "\n"
        "═══ Rule phrasing — CRITICAL ═══\n"
        "\n"
        "Rules MUST be written in second-person, evidence-grounded form.\n"
        "The internal rule text IS the displayed text — users will read\n"
        "this exact string when they run /memory list rules. The\n"
        "grounding-in-evidence framing is what makes the rule feel like\n"
        "an observation rather than a verdict.\n"
        "\n"
        "  Good: 'You've said meditation makes you more anxious.'\n"
        "  Bad:  'User dislikes meditation.'\n"
        "  Bad:  'Avoid suggesting meditation.'\n"
        "  Bad:  'Don't mention meditation to this user.'\n"
        "\n"
        "Second-person is non-negotiable. Use 'you' and 'your' when\n"
        "referring to the user. Never use 'user' as a third-person\n"
        "subject. Never write the rule as a command aimed at the agent.\n"
        "\n"
        "Ground every rule in what the user actually said. If you can't\n"
        "point to a specific phrase from the user's current message as\n"
        "the trigger, you probably shouldn't be writing a rule.\n"
        "\n"
        "═══ Output shape ═══\n"
        "\n"
        "Return a ProceduralExtractionResult with two fields:\n"
        "\n"
        "  rules  : a list of zero or more ProceduralRuleDraft items.\n"
        "           Empty list is the common case. Each draft has:\n"
        "             - rule         (second-person, evidence-grounded,\n"
        "                             ≤ 280 chars)\n"
        "             - evidence     (list of verbatim user quotes, at\n"
        "                             least one)\n"
        "             - confidence   ('low', 'medium', or 'high')\n"
        "\n"
        "  reason : a short one-sentence explanation. Always populate this.\n"
        "           When rules is empty: explain why nothing was written\n"
        "           (e.g., 'user shared a feeling but did not request a\n"
        "           style change').\n"
        "           When rules is non-empty: summarize what was written\n"
        "           (e.g., 'user asked to stop being offered meditation').\n"
        "\n"
        "A direct command is NOT required when the user clearly says a\n"
        "recurring agent behavior or suggestion is unhelpful for them.\n"
        "Statements like 'when you suggest meditation, it makes me more\n"
        "anxious' should be written as candidate rules.\n"
        "\n"
        "If you are unsure whether the message is really agent-facing,\n"
        "OMIT the rule. Silence is always the safer choice. Borderline\n"
        "implicit preferences should only be written when the current\n"
        "message itself clearly implies future agent behavior should\n"
        "change; otherwise skip and let repetition provide evidence later."
    )


def build_procedural_writer_user_prompt(state: AgentState) -> str:
    """Build the procedural-writer user prompt.

    Args:
        state (AgentState): Current graph state with message and recent history.

    Returns:
        str: User prompt for a single procedural-rule write call.
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

    current_message = state["message"]

    return (
        f"Recent conversation (for context — do NOT write rules from "
        f"history; only from the current user message):\n"
        f"{history_block}\n"
        f"\n"
        f"Current user message (write rules from THIS only):\n"
        f"user: {current_message}\n"
        f"\n"
        f"Is the user explicitly asking for a style change in how the "
        f"agent should respond? Return a ProceduralExtractionResult. If "
        f"nothing in the current message meets all five writing criteria "
        f"from the system prompt, return an empty rules list with a "
        f"reason explaining why."
    )
