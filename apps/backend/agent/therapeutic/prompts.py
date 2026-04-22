"""System prompt builders for therapeutic response modes.

Each mode gets a system prompt composed from the repo-level ``knowledge/``
markdown files (identity, boundaries, privacy) plus mode-specific guidance.
The composition pattern mirrors ``agent/prompts/crisis.py`` — load files,
concatenate, return as one string.

Phase 1 v0.1 scope: three modes (supportive, reflective, clarifying).
The other three (psychoeducation, guided_exercise, closing) land in v0.6.
Modality overlays (CBT, ACT, PFA, etc.) are deferred to phase 2.

The prompt builders also provide ``format_recent_history`` and
``build_therapeutic_response_prompt`` helpers that all mode nodes share
for constructing their user/task prompts.
"""

from __future__ import annotations

from agent.prompts import (
    CORE_SOURCES,
    compose_sources as _compose,
    format_recent_history as _format_recent_history,
    load_prompt_source as _load_knowledge_file,
)
from agent.state import AgentState
from agent.working_memory import format_working_memory_entries


# ─── Mode base knowledge (without modality overlay) ──────────────────────────

_MODE_BASE_KNOWLEDGE: dict[str, tuple[str, ...]] = {
    "supportive": (*CORE_SOURCES, "response_modes/support.md"),
    "reflective": (*CORE_SOURCES, "response_modes/reflection.md"),
    "clarifying": CORE_SOURCES,
    "psychoeducation": (*CORE_SOURCES, "response_modes/psychoeducation.md"),
    "closing": (*CORE_SOURCES, "response_modes/closing.md"),
    "guided_exercise": (*CORE_SOURCES, "response_modes/guided_exercise.md"),
    # Technique mode uses ONLY core + the approach overlay. The approach
    # knowledge IS the mode-specific guidance — no separate response_mode file.
    "technique": CORE_SOURCES,
}

# ─── Modality file mapping ───────────────────────────────────────────────────

_MODALITY_FILES: dict[str, tuple[str, ...]] = {
    "motivational_interviewing": ("modalities/motivational_interviewing.md",),
    "cbt": ("modalities/cbt.md", "modalities/cbt_arc.md"),
    "act": ("modalities/act.md",),
    "dbt_skills": ("modalities/dbt_skills.md",),
    "grief_support": ("modalities/grief_support.md",),
    "interpersonal_therapy": ("modalities/interpersonal_therapy.md",),
    "pfa": ("modalities/pfa.md",),
}


def _knowledge_for_mode(mode: str, modality: str | None = None) -> tuple[str, ...]:
    """Compose the knowledge file list for a mode + modality combination.

    Returns the base knowledge for the mode, plus the modality overlay
    file(s) if a valid modality is specified. When modality is None or
    "none", only the base mode knowledge is returned.
    """

    base = _MODE_BASE_KNOWLEDGE.get(mode, CORE_SOURCES)
    if modality and modality != "none" and modality in _MODALITY_FILES:
        return (*base, *_MODALITY_FILES[modality])
    return base


# Backward-compatible aliases for callers that haven't been updated
# to pass modality yet. These will be removed once all mode nodes
# use _knowledge_for_mode directly.
_SUPPORTIVE_KNOWLEDGE = (
    *_MODE_BASE_KNOWLEDGE["supportive"],
    "modalities/motivational_interviewing.md",
)
_REFLECTIVE_KNOWLEDGE = _MODE_BASE_KNOWLEDGE["reflective"]
_CLARIFYING_KNOWLEDGE = _MODE_BASE_KNOWLEDGE["clarifying"]
_PSYCHOEDUCATION_KNOWLEDGE = _MODE_BASE_KNOWLEDGE["psychoeducation"]
_CLOSING_KNOWLEDGE = _MODE_BASE_KNOWLEDGE["closing"]
_GUIDED_EXERCISE_KNOWLEDGE = _MODE_BASE_KNOWLEDGE["guided_exercise"]


def _format_working_memory(state: AgentState) -> str:
    """Format working memory snippets for prompt injection.

    If there is no working memory (incognito mode, or no facts extracted
    yet), returns an empty string so the prompt section is simply absent
    rather than showing an empty list.
    """

    snippets = format_working_memory_entries(state.get("working_memory", []))
    if not snippets:
        return ""
    joined = "\n".join(f"- {snippet}" for snippet in snippets)
    return f"\nRelevant context from past sessions:\n{joined}\n"


# ─── v0.7 Stage D procedural + recall helpers ──────────────────────────────
#
# These two helpers produce the dynamic blocks that get appended to every
# response-generator's system prompt. They read state directly and return
# empty strings when the relevant state is absent, so the prompt gracefully
# degrades to the static knowledge+instructions content when procedural
# memory isn't populated.
#
# Design notes:
#
# 1. **Rules are always injected (when they exist).** The rules block is
#    emitted whenever ``memory.procedural_rules`` is non-empty, regardless
#    of the recall toggle. Rules are constraints; the agent must apply
#    them on every response. This is the B1 interpretation locked during
#    Stage D planning — rules are silent lint, not referenceable content.
#
# 2. **Rules are silent lint, not quotable content.** The block explicitly
#    tells the model to follow the rules without narrating compliance,
#    without quoting them, and without announcing that it's adjusting its
#    behavior. This matches the failure-mode asymmetry we settled: a user
#    who turned off proactive recall specifically to AVOID memory-narration
#    should never hear the agent say "as per your earlier request..." even
#    when recall is on.
#
# 3. **Recall toggle governs semantic/episodic only.** The constraint
#    block is intentionally kept stable here. When recall is off, the
#    model is told to use retrieved memories for shaping but NOT to
#    explicitly reference past sessions or statements. When recall is
#    on, the constraint is relaxed. The constraint text deliberately
#    refers to "past sessions or past statements" rather than "memory"
#    in general — that's what keeps rules out of scope for this toggle.


def _format_procedural_rules_block(state: AgentState) -> str:
    """Format the user's procedural rules as a silent-constraint block.

    Returns the empty string when no rules exist. When rules are present,
    returns a prompt suffix that lists them with explicit instructions to
    follow them silently — never quote, cite, or narrate compliance.

    The block is unconditional with respect to the recall toggle. Rules
    are applied on every response regardless of whether the user has
    enabled or disabled proactive memory recall. See the module-level
    comment above for the rationale (B1 interpretation).
    """

    memory = state.get("memory", {}) or {}
    rules = memory.get("procedural_rules") or []
    if not rules:
        return ""

    rule_lines = "\n".join(f"- {rule}" for rule in rules)
    return (
        "\n\n═══ Style rules from past conversations with this user ═══\n"
        f"{rule_lines}\n"
        "\n"
        "Follow these rules silently. Do NOT quote them, cite them, or "
        "narrate your compliance with them (e.g., never say 'as per your "
        "earlier request...'). The user already knows they asked for "
        "these; acknowledging them makes the interaction feel "
        "customer-service-y. Just apply the rules as part of how you "
        "respond."
    )


def _format_recall_toggle_constraint(state: AgentState) -> str:
    """Format the recall-toggle constraint block for the system prompt.

    Returns a prompt suffix whose content depends on
    ``memory.proactive_recall_enabled``:

    - **When False (default)**: tells the model to use retrieved memories
      for silent shaping but NOT to explicitly reference past sessions or
      past statements. This is the "invisible but effective" mode users
      get when they turn off proactive recall.
    - **When True**: relaxes the constraint so the model may reference
      past memories sparingly when they add value to the current moment.

    The constraint refers specifically to "past sessions or past
    statements" — semantic facts and episodic summaries — and does NOT
    govern procedural rules, which are separately handled by
    :func:`_format_procedural_rules_block`.
    """

    memory = state.get("memory", {}) or {}
    enabled = memory.get("proactive_recall_enabled", False)

    if enabled:
        # Recall ON: relaxed constraint.
        return (
            "\n\n═══ Memory reference guidance (proactive recall: ON) ═══\n"
            "You may reference relevant past memories when it adds value "
            "to the current moment, but do so sparingly and never for "
            "emotionally charged topics without strong contextual fit."
        )

    # Recall OFF (default): silent-shaping constraint.
    return (
        "\n\n═══ Memory reference guidance (proactive recall: OFF) ═══\n"
        "Use any retrieved memories to inform the warmth, pacing, and "
        "content of your response, but do NOT explicitly reference past "
        "sessions or past statements unless the user has just asked "
        "about them."
    )


# ─── Mode instructions (appended to the system prompt) ──────────────────────
#
# These are the mode-specific behavioral instructions that shape HOW the
# agent responds. They sit on top of the knowledge-file composition so
# the agent has both the foundational identity/boundaries AND the
# mode-specific guidance.

_SUPPORTIVE_INSTRUCTIONS = """
You are in SUPPORTIVE mode. Your job is to listen well, validate the
user's feelings, and leave room for them to continue sharing.

Guidelines:
- Be warm but not effusive. Match the user's energy.
- Validate the feeling before offering any reflection.
- Keep your response short: 2-4 sentences, rarely more.
- Light-touch reflection: name the feeling, don't analyze it.
- Do not ask more than one question. Often, no question is best.
- Never start with "I understand" — it sounds hollow from an AI.
- If working memory contains relevant past context, let it inform
  your warmth and pacing, but do NOT explicitly reference past
  sessions unless the user asks about them.
""".strip()

_REFLECTIVE_INSTRUCTIONS = """
You are in REFLECTIVE mode. The user seems to be noticing a pattern
or asking a "why does this keep happening?" type of question. Your job
is to gently name the pattern and invite the user to reflect on it.

Guidelines:
- Name ONE pattern, not several. Focus matters.
- Ground the naming in the user's own words when possible.
  ("I notice you keep saying 'I should'...")
- Invite reflection with one open question at most.
- Acknowledge the observation might be wrong: "Does that resonate,
  or is it more like...?"
- Slightly longer than supportive mode: 3-5 sentences.
- Never introduce a pattern the user hasn't shown evidence for.
  Hallucinated patterns are the single worst failure mode.
- If the user asked a "why" question, offer a reflection, not a
  diagnosis or explanation. Explanations belong to psychoeducation.
""".strip()

_CLARIFYING_INSTRUCTIONS = """
You are in CLARIFYING mode. The user's message is too short, too
ambiguous, or too out-of-context to respond to well. Your job is to
ask ONE focused question to get the information you need.

Guidelines:
- Acknowledge what you heard first: "It sounds like something's on
  your mind..."
- Ask exactly ONE question, not a list.
- The question should be open-ended, not yes/no.
- Keep it very short: 2-3 sentences total.
- Never say "Can you tell me more?" — that's too generic. Ask
  something SPECIFIC about what the user hinted at.
- The question should be about CONTEXT, not CONTENT. "What brought
  this up?" is better than "What do you mean?"
""".strip()

_PSYCHOEDUCATION_INSTRUCTIONS = """
You are in PSYCHOEDUCATION mode. The user is confused about their
own reaction or wants a brief frame for what they're experiencing.
Your job is to offer ONE short, plain-language explanation that
normalizes the experience and then pivots back to the user.

Guidelines:
- Default length: 2-3 sentences of framing + ONE check-in question
  that returns focus to the user's experience.
- When the moment is weighty (user is tentatively touching grief,
  a new memory, or an acute body response), use a much shorter
  turn: one sentence of framing + check-in, or just acknowledgment
  + space. See the "Length varies with moment weight" section of
  the knowledge file.
- Normalize, do not diagnose. Never label the user with a clinical
  condition, cite research, name theorists, or quote studies.
- Use "people often..." or "it's common for..." phrasing rather
  than "you are..." — descriptive, not prescriptive.
- Pivot back to the user's specific experience at the end of the
  turn. The explanation is a bridge, not the destination.
- If the user's message reads as an expression of emotion rather
  than a question about their own reaction (e.g., "I'm so angry
  right now"), lead with the permission-first pattern: brief
  acknowledgment + offer to share a thought + space for the user
  to choose. Do NOT launch into an explanation they didn't ask for.
- Never: diagnose, lecture, cite research, use clinical terminology
  the user didn't introduce, end the turn on the explanation
  itself.
""".strip()

_CLOSING_INSTRUCTIONS = """
You are in CLOSING mode. The user is signaling they're winding
down ("I should go", "thanks, this helped"), OR a natural lull
has followed productive work and the conversation feels complete
enough for now. Your job is to help the user leave the conversation
feeling oriented rather than abruptly cut off.

Guidelines:
- Keep it SHORT: 2-4 sentences total. This is the single most
  important rule — long closings feel performative.
- Lead with a brief acknowledgment of the arc if there was one
  ("It sounds like naming the work stress gave you a bit of
  breathing room"). Stay concrete, not abstract.
- End with a warm, low-pressure open door ("Whenever you want
  to pick this back up, I'm here"). One sentence, no commitment
  ask.
- If the user named an unresolved thread earlier, acknowledge
  it gently — at most ONE thread, no stacking. "You mentioned
  the thing with your sister earlier — that's still there
  whenever you want to come back to it."
- Never: "It was nice talking to you" (transactional, customer-
  service register — the single most common failure mode).
- Never: "Please come back soon" (puts the relationship onto
  the user).
- Never: exhaustive summaries of everything discussed.
- Never: introduce a new topic, question, or next step.
- Never: claim the user "made progress" unless they explicitly
  said so.
- Closing is TONAL, not structural. You are NOT ending the
  session or triggering any system action. The user can keep
  talking if they want. Session termination and summarization
  are runtime concerns handled elsewhere.
""".strip()

_GUIDED_EXERCISE_INSTRUCTIONS = """
You are in GUIDED_EXERCISE mode. The user has asked for a
structured exercise (grounding, breathing, etc.) OR an exercise
is already in progress from a prior turn. Your job is to guide
the user through ONE step of the exercise at a time, in clear
present-tense language, and wait for them to respond.

Guidelines:
- Guide ONE step at a time. Never list all the steps upfront.
- Use short, concrete, present-tense instructions. "Name five
  things you can see right now." NOT "When you're ready, try to
  identify items in your visual field."
- Keep each turn short: 2-4 sentences is usually enough. Longer
  responses add cognitive load during distress.
- Acknowledge specifically when the user does a step ("a lamp,
  a plant, and your coffee cup — nice") before moving to the
  next step.
- If the user is tentative (names fewer items than asked, or
  trails off), HOLD the step and give them space — "Take your
  time, even one counts." Do NOT advance the step or abandon
  the exercise on the first sign of friction. Patience is a
  feature; over-rescuing is the biggest failure mode for this
  mode.
- If the user explicitly wants to stop ("I don't want to do
  this", "can we just talk", "this isn't helping"), acknowledge
  their choice WITHOUT defending the exercise and offer a gentle
  landing. "Of course — let's stop. What would feel most
  helpful right now?" Do NOT try to redirect them back to the
  exercise.
- When an exercise completes, briefly name what they just did,
  offer ONE simple takeaway if it fits, and leave space. Do NOT
  launch into a second exercise.
- Never: explain the neuroscience of why the exercise works
  before doing the exercise; give the user a menu of exercises
  to choose from; chain multiple exercises together; lecture
  about the exercise theory; treat the exercise like a
  worksheet with fill-in-the-blank fields.
""".strip()

_TECHNIQUE_INSTRUCTIONS = """
You are in TECHNIQUE mode. The therapeutic approach is driving this
turn — follow its process guidance for the current phase. Your job
is to execute the specific therapeutic technique the approach
describes, not to reflect, validate, or explain.

Guidelines:
- The approach knowledge loaded into this prompt describes what to
  do: which questions to ask, what rhythm to follow, what to listen
  for, when to advance. Follow that guidance as your primary
  behavioral instruction.
- Keep turns focused and concrete. One question, one reflection,
  one step — not a paragraph of exploration.
- The approach's transition signals tell you when to advance to
  the next phase. Watch for them explicitly.
- If the user becomes flooded, numb, or distressed beyond what
  the technique can hold: STOP the technique. Acknowledge what
  happened. Drop into supportive or grounding. You can return to
  the technique later when the user is regulated.
- If the user resists the structure ("I don't want to do this",
  "can we just talk"): acknowledge immediately, stop the
  technique, and follow their lead. The technique is a tool, not
  the therapy.
- Never: lecture about the technique before doing it, name the
  technique or its clinical terminology to the user, force a step
  the user isn't ready for, stack multiple techniques.
""".strip()


# ─── Public prompt builders ──────────────────────────────────────────────────


_CONTINUITY_FILE = "cross_session_continuity.md"


def _has_episodic_context(state: AgentState) -> bool:
    """Return True if the user has prior session history available."""

    working_memory = state.get("working_memory", [])
    return any(entry.get("type") == "episodic" for entry in working_memory)


def _compose_system_prompt_with_state(
    knowledge: str,
    instructions: str,
    state: AgentState,
) -> str:
    """Assemble a system prompt from static + dynamic parts.

    The static parts are the knowledge-file composition and the
    mode-specific instructions block. The dynamic parts are the
    procedural rules block, the recall toggle constraint (both
    read from state), and the cross-session continuity guidance
    (loaded only for returning users with episodic memory).

    See the module-level comment above
    :func:`_format_procedural_rules_block` for the design notes.
    """

    continuity_block = ""
    if _has_episodic_context(state):
        continuity_block = "\n\n" + _load_knowledge_file(_CONTINUITY_FILE)

    rules_block = _format_procedural_rules_block(state)
    recall_block = _format_recall_toggle_constraint(state)
    return f"{knowledge}\n\n{instructions}{continuity_block}{rules_block}{recall_block}"


def _read_approach(state: AgentState) -> str | None:
    """Read the dispatcher-selected modality from routing state."""

    return state.get("routing", {}).get("therapeutic_approach")


def build_supportive_system_prompt(state: AgentState) -> str:
    """Build the system prompt for supportive-mode responses.

    Loads the modality overlay selected by the dispatcher (defaults to
    MI if no modality was set, for backward compatibility).
    """

    modality = _read_approach(state) or "motivational_interviewing"
    files = _knowledge_for_mode("supportive", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _SUPPORTIVE_INSTRUCTIONS, state)


def build_reflective_system_prompt(state: AgentState) -> str:
    """Build the system prompt for reflective-mode responses."""

    modality = _read_approach(state)
    files = _knowledge_for_mode("reflective", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _REFLECTIVE_INSTRUCTIONS, state)


def build_clarifying_system_prompt(state: AgentState) -> str:
    """Build the system prompt for clarifying-mode responses."""

    knowledge = _compose(*_CLARIFYING_KNOWLEDGE)
    return _compose_system_prompt_with_state(knowledge, _CLARIFYING_INSTRUCTIONS, state)


def build_psychoeducation_system_prompt(state: AgentState) -> str:
    """Build the system prompt for psychoeducation-mode responses."""

    modality = _read_approach(state)
    files = _knowledge_for_mode("psychoeducation", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _PSYCHOEDUCATION_INSTRUCTIONS, state
    )


def build_closing_system_prompt(state: AgentState) -> str:
    """Build the system prompt for closing-mode responses."""

    knowledge = _compose(*_CLOSING_KNOWLEDGE)
    return _compose_system_prompt_with_state(knowledge, _CLOSING_INSTRUCTIONS, state)


def build_guided_exercise_system_prompt(state: AgentState) -> str:
    """Build the system prompt for guided_exercise-mode responses.

    Loads the base exercise knowledge plus the approach overlay (CBT for
    thought records, ACT for defusion, etc.).  Prefers the stable
    ``progress.exercise_modality`` captured at exercise start over the
    per-turn ``routing.therapeutic_approach`` — the latter can drift if
    the user takes a clarifying or psychoeducation side-turn mid-exercise.
    """

    progress = state.get("progress", {})
    # Only trust exercise_modality when an exercise is actually active.
    approach = (
        progress.get("exercise_modality") if progress.get("exercise_type") else None
    ) or _read_approach(state)
    files = _knowledge_for_mode("guided_exercise", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _GUIDED_EXERCISE_INSTRUCTIONS, state
    )


def build_technique_system_prompt(state: AgentState) -> str:
    """Build the system prompt for technique-mode responses.

    In technique mode, the therapeutic approach drives the response
    shape. The approach knowledge files are loaded as the primary
    behavioral guidance — the technique instructions just say "follow
    the approach's process guidance." This is the one response style
    where the approach is loud, not background.

    A valid therapeutic approach MUST be set in routing state. If no
    approach is active, this falls back to the supportive prompt to
    avoid generating an ungrounded response.
    """

    approach = _read_approach(state)
    if not approach or approach == "none":
        # No approach active — technique mode without an approach
        # doesn't make sense. Fall back to supportive.
        return build_supportive_system_prompt(state)

    files = _knowledge_for_mode("technique", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _TECHNIQUE_INSTRUCTIONS, state)


def build_therapeutic_response_prompt(
    state: AgentState,
    *,
    mode: str,
    step_directive: str | None = None,
) -> str:
    """Build the user/task prompt for any therapeutic mode.

    All modes share the same user-prompt structure — the system
    prompt (which differs per mode) is what shapes the response
    character. The user prompt provides the conversation context
    and current message.

    Args:
        state: Current graph state with history and working memory.
        mode: The dispatched mode name, injected as context for
            observability in the prompt.
        step_directive: For multi-turn modes (guided_exercise), an
            explicit instruction about what the LLM should generate.
            This bridges the node's deterministic state transition
            to the LLM's prose generation — the node knows *which*
            step to produce, and tells the LLM via this directive.
    """

    memory_block = _format_working_memory(state)
    directive_block = f"\n\nStep directive:\n{step_directive}" if step_directive else ""

    return (
        f"Write the next assistant message for a mental health support "
        f"conversation in {mode} mode.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
        f"{directive_block}"
    )
