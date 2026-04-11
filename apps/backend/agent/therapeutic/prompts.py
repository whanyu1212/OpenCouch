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

from functools import lru_cache
from pathlib import Path

from agent.state import AgentState


# ─── Knowledge file composition (same pattern as prompts/crisis.py) ─────────

_CORE_KNOWLEDGE = (
    "soul.md",
    "identity.md",
    "policy/boundaries.md",
    "policy/privacy.md",
)

_SUPPORTIVE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/support.md",
    "modalities/motivational_interviewing.md",
)

_REFLECTIVE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/reflection.md",
)

_CLARIFYING_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    # No mode-specific knowledge file — the clarifying prompt is
    # self-contained in the mode instructions below.
)

_PSYCHOEDUCATION_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/psychoeducation.md",
)

_CLOSING_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/closing.md",
)

_GUIDED_EXERCISE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/guided_exercise.md",
)


def _knowledge_root() -> Path:
    """Return the absolute path to the repo-level ``knowledge/`` directory."""

    return Path(__file__).resolve().parents[4] / "knowledge"


@lru_cache(maxsize=32)
def _load_knowledge_file(relative_path: str) -> str:
    """Load one markdown knowledge file by its relative path."""

    root = _knowledge_root().resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path.read_text(encoding="utf-8").strip()


def _compose(*relative_paths: str) -> str:
    """Concatenate knowledge files into a single prompt block."""

    parts = [_load_knowledge_file(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _format_recent_history(state: AgentState, *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection."""

    history = state["history"][-limit:]
    if not history:
        return "(no prior history)"

    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in history
        if turn.get("content")
    )


def _format_working_memory(state: AgentState) -> str:
    """Format working memory snippets for prompt injection.

    If there is no working memory (incognito mode, or no facts extracted
    yet), returns an empty string so the prompt section is simply absent
    rather than showing an empty list.
    """

    snippets = state.get("working_memory", [])
    if not snippets:
        return ""
    joined = "\n".join(f"- {s}" for s in snippets)
    return f"\nRelevant context from past sessions:\n{joined}\n"


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


# ─── Public prompt builders ──────────────────────────────────────────────────


def build_supportive_system_prompt() -> str:
    """Build the system prompt for supportive-mode responses."""

    knowledge = _compose(*_SUPPORTIVE_KNOWLEDGE)
    return f"{knowledge}\n\n{_SUPPORTIVE_INSTRUCTIONS}"


def build_reflective_system_prompt() -> str:
    """Build the system prompt for reflective-mode responses."""

    knowledge = _compose(*_REFLECTIVE_KNOWLEDGE)
    return f"{knowledge}\n\n{_REFLECTIVE_INSTRUCTIONS}"


def build_clarifying_system_prompt() -> str:
    """Build the system prompt for clarifying-mode responses."""

    knowledge = _compose(*_CLARIFYING_KNOWLEDGE)
    return f"{knowledge}\n\n{_CLARIFYING_INSTRUCTIONS}"


def build_psychoeducation_system_prompt() -> str:
    """Build the system prompt for psychoeducation-mode responses."""

    knowledge = _compose(*_PSYCHOEDUCATION_KNOWLEDGE)
    return f"{knowledge}\n\n{_PSYCHOEDUCATION_INSTRUCTIONS}"


def build_closing_system_prompt() -> str:
    """Build the system prompt for closing-mode responses."""

    knowledge = _compose(*_CLOSING_KNOWLEDGE)
    return f"{knowledge}\n\n{_CLOSING_INSTRUCTIONS}"


def build_guided_exercise_system_prompt() -> str:
    """Build the system prompt for guided_exercise-mode responses."""

    knowledge = _compose(*_GUIDED_EXERCISE_KNOWLEDGE)
    return f"{knowledge}\n\n{_GUIDED_EXERCISE_INSTRUCTIONS}"


def build_therapeutic_response_prompt(state: AgentState, *, mode: str) -> str:
    """Build the user/task prompt for any therapeutic mode.

    All three modes share the same user-prompt structure — the system
    prompt (which differs per mode) is what shapes the response
    character. The user prompt provides the conversation context and
    current message.

    Args:
        state: Current graph state with history and working memory.
        mode: The dispatched mode name, injected as context for
            observability in the prompt.
    """

    memory_block = _format_working_memory(state)

    return (
        f"Write the next assistant message for a mental health support "
        f"conversation in {mode} mode.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
    )
