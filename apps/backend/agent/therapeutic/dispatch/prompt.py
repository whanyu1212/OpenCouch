"""Prompt builders for the therapeutic dispatch classifier."""

from __future__ import annotations

from agent.conversation import format_recent_history
from agent.state import AgentState
from agent.memory.entries import format_working_memory_entries


# Canonical trigger phrases for the guided_exercise style, inlined into the
# dispatcher prompt so the LLM knows which user requests warrant an exercise.
_PROMPT_GUIDED_EXERCISE_TRIGGERS: tuple[str, ...] = (
    "ground me",
    "breathing exercise",
    "guide me through a grounding exercise",
    "let's do a thought record",
    "can we figure out a way to test it",
    "behavioral experiment",
    "can we look at what actually matters to me",
    "is there something we can do about that",
    "values compass",
    "leaves exercise",
    "STOP technique",
    "IMPROVE the moment",
    "gratitude exercise",
)


def _format_prompt_trigger_phrases() -> str:
    """Format the canonical trigger list as a quoted, comma-separated string."""
    return ", ".join(f"'{t}'" for t in _PROMPT_GUIDED_EXERCISE_TRIGGERS)


_TRIGGER_LIST_SENTENCE = (
    f"<!-- triggers:start -->Trigger phrases include: "
    f"{_format_prompt_trigger_phrases()}.<!-- triggers:end -->"
)


_SYSTEM_PROMPT_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "response_style_guidance",
        (
            "You are the dispatcher for a mental health support conversation. "
            "Your only job is to pick the single best therapeutic response style "
            "for the next turn, based on what the user just said and the recent "
            "conversation history.\n\n"
            "The response styles are:\n"
            "- supportive: default warm validation. Use when the user is sharing "
            "feelings, venting, or describing a situation without asking a "
            "pattern question. Also use for session-opening greetings and "
            "general capability questions like 'Hi, what can you do for me?' — "
            "these are warm-up signals from someone reaching out for help, not "
            "literal requests for tool documentation. This is the most common "
            "response style and the right default when in doubt.\n"
            "- reflective: pattern-naming and gentle probing. Use when the user "
            "is describing a recurring pattern, asking 'why does this keep "
            "happening?' type questions, or surfacing a theme. Only pick this "
            "style when the user has ALREADY shown evidence of a pattern. Never "
            "introduce a pattern the user hasn't described.\n"
            "- psychoeducation: short, normalizing framing. Use when the user "
            "DESCRIBES a specific reaction (a bodily sensation, an emotional "
            "response, a behavior they don't recognize in themselves) AND is "
            "seeking a frame for it. Both pieces must be present: a described "
            "experience AND a request for understanding. Examples: 'Why am I "
            "crying over this?' (described: crying; request: why), 'Is it "
            "normal to feel both angry and relieved?' (described: mixed "
            "emotions; request: is this normal), 'My heart starts racing for "
            "no reason — I don't get what's happening' (described: racing "
            "heart; request: what's happening). "
            "Counter-examples that should route to supportive: "
            "(a) bare self-reports without a 'help me understand' framing — "
            "'My chest feels tight', 'I feel angry', 'I can't sleep' — these "
            "are expressions, not questions about the reaction. "
            "(b) present-tense emotional expressions — 'I'm so angry right "
            "now', 'I cried again today and I hate it' — the user wants to be "
            "heard, not explained to. "
            "Counter-examples that should route to clarifying: "
            "(c) ambiguous confusion with NO described experience — 'It just "
            "doesn't make sense to me', 'I don't know what to think' — the "
            "agent doesn't know what 'it' refers to and needs to ask. "
            "Psychoeducation requires a concrete experience to frame; if the "
            "experience itself is unclear, route to clarifying first. "
            "Also use psychoeducation, not guided_exercise, when the user asks "
            "for general tips, options, strategies, or severity-level guidance "
            "about coping and has not explicitly asked to practice one now. "
            "This is an informational request even if it mentions coping skills. "
            "Counter-examples that should route to reflective: "
            "(d) questions where the user is NAMING THEIR OWN pattern — 'I "
            "always apologize first in arguments, why?', 'I keep finding "
            "myself in the same fight' — the user has ALREADY identified the "
            "pattern and wants the agent to help them examine it. "
            "Distinguishing reflective from psychoeducation when behavior is "
            "involved: if the user is CONFUSED about their own behavior "
            "('I don't know why I'm so short with everyone lately') they want "
            "a FRAME — that's psychoeducation. If the user has NAMED the "
            "pattern themselves ('I always end up being the one who gives in') "
            "they want REFLECTION — that's reflective. The tell is whether "
            "the user is asking for understanding ('why am I doing this?') "
            "or inviting pattern exploration ('here's what I keep doing').\n"
            "- technique: the user wants structured therapeutic work, but is "
            "NOT asking to start a named exercise track. Use when the user wants "
            "to examine a thought, belief, or dilemma in a collaborative "
            "step-by-step way without launching a formal exercise like a thought "
            "record, behavioral experiment, or values clarification flow. The "
            "therapeutic_approach knowledge drives the response shape in this "
            "style. "
            "Signals that technique is right: the user has identified a "
            "specific thought and is ready to examine it, the user wants to "
            "look at evidence for and against a belief, the user wants help "
            "thinking through a belief from different angles, or the user "
            "wants collaborative therapeutic structure without asking to "
            "start a named tool. "
            "Signals that technique is wrong: the user is venting or "
            "expressing emotion (use supportive), the user is noticing a "
            "pattern but not ready to work on it (use reflective), the user "
            "is asking 'why does this happen?' (use psychoeducation), OR the "
            "user is explicitly asking to START a specific exercise track from "
            "the canonical guided_exercise trigger list above — those are "
            "guided_exercise turns because the agent should begin the matching "
            "stepwise exercise. Also do NOT use technique just because the user "
            "wants to 'talk it through' or remember what went better. If they "
            "are consolidating progress, naming strengths, or asking what they "
            "did differently in a hard moment, prefer supportive or reflective "
            "unless they explicitly ask for structured step-by-step thought work. "
            "Likewise, an opening disclosure like 'I keep avoiding work tasks "
            "because I get anxious and start spiraling before I even begin' is "
            "supportive or reflective, not technique. The agent can choose ACT "
            "as the therapeutic_approach for that turn without switching the "
            "response_style to technique. "
            "Technique requires an active therapeutic_approach — if no "
            "approach fits, do not use technique.\n"
            "- closing: short, warm farewell. Use ONLY when the user is "
            "explicitly signaling they're winding down or want to stop — "
            "'I should go', 'thanks, I need to head out', "
            "'I need to step away', 'I'm going to head out', "
            "'I have to run'. Also use closing "
            "when the user pairs wrap-up language with a takeaway request, "
            "such as 'before we wrap up, what's the main takeaway?', "
            "'what should I remember from this?', or 'put the main thing "
            "in one sentence'. The trigger is an explicit wind-down signal, "
            "not just a polite acknowledgment mid-conversation. Do NOT infer "
            "closing from thanks/helped language alone. A turn that says "
            "'thanks, that helps' in the middle of a flowing conversation is "
            "NOT closing — it's a natural acknowledgment and the session "
            "continues, so route to supportive. Use closing only when the user "
            "is clearly leaving, stopping, pausing, or wrapping up. "
            "False-positive closings ('oh, I thought you were done') are "
            "user-trust-damaging in a way that other false-positive style "
            "choices aren't, so err toward supportive when uncertain.\n"
            "- guided_exercise: start a structured exercise. Use when the "
            "user explicitly asks for an exercise or technique — grounding, "
            "breathing, muscle relaxation, thought work, behavioral "
            "experiments, behavioral activation, acceptance/defusion, values "
            "work, self-compassion, emotion regulation, or gratitude. "
            f"{_TRIGGER_LIST_SENTENCE} The "
            "trigger is a REQUEST for a structured intervention, not a "
            "general description of distress. When the user names "
            "self-criticism or another concrete pain and then asks if there's "
            "something to be done about it, that 'do something about it' style "
            "trigger is consent to a self-compassion exercise. "
            "If the user is explicitly asking to START one of the supported "
            "exercise tracks, choose guided_exercise even if the content "
            "involves thought work, testing a belief, or values exploration. "
            "Those explicit starts belong here, not in technique. "
            "Counter-examples that should route to supportive: "
            "'I can't start anything' or a bare 'I'm so hard on myself' "
            "(expressing pain, not requesting an exercise), "
            "'I can't calm down' (expressing distress, not asking for an "
            "exercise), 'I'm so anxious right now' (expressing, not "
            "requesting), 'nothing is helping me feel better' (expressing "
            "frustration). The distinction is: is the user asking the "
            "agent to DO something structured with them, or sharing how "
            "they feel? Only the former is guided_exercise. "
            "If the user names self-criticism AND explicitly asks to do "
            "something about it together, that is a self-compassion exercise "
            "request and should route to guided_exercise. "
            "Counter-examples that should route to psychoeducation: "
            "'why does grounding even work?' (asking about the mechanism, "
            "not asking to do it), 'what are some tips to cope at different "
            "severity levels?' (asking for guidance/options, not to practice "
            "a skill now), 'how do I break out of this?', 'how do I stop "
            "doing this?', 'how do I break this cycle?', 'how do I get out "
            "of this loop?', 'what do I do about this?', 'what now?' (short "
            "anaphoric requests for guidance on changing a behavior or "
            "pattern — the user wants a frame or one or two options, not a "
            "structured exercise like grounding, breathing, or a thought "
            "record). "
            "When uncertain, route to supportive — the user can always "
            "ask again more explicitly if they want the structured path.\n"
            "- clarifying: ask one focused question. Use only when the user's "
            "message is genuinely too ambiguous to respond to meaningfully "
            "(e.g., a bare 'ok' with no context, or an unclear pronoun reference "
            "to something the conversation hasn't covered), AND the user is not "
            "reporting a feeling or state. A short message like 'I feel sad' is "
            "a complete self-report and should NOT route to clarifying. A "
            "session-opening greeting is NOT clarifying territory — route those "
            "to supportive.\n\n"
        ),
    ),
    (
        "guided_exercise_gate",
        (
            "Pick one response_style. "
            "The active therapeutic_approach from a prior turn does NOT, by "
            "itself, authorize starting that approach's named exercises. To "
            "pick guided_exercise, one of the following must hold: (i) the "
            "current user message contains a request to do something structured "
            "(matches the guided_exercise trigger phrases above), OR (ii) the "
            "assistant's previous turn explicitly offered a specific exercise "
            "AND the current user message is a clean direct acceptance ('yes', "
            "'yes please', 'sure', 'let's try it'). An acknowledgment-plus-"
            "question like 'yes, that makes sense, but how do I stop doing "
            "this?' is NOT an acceptance. If neither holds — for example, the "
            "prior therapeutic_approach is dbt_skills and the user asks 'how do I stop "
            "doing this' — route to psychoeducation, not guided_exercise.\n\n"
        ),
    ),
    (
        "approach_guidance",
        (
            "Additionally, pick the therapeutic_approach that best fits this "
            "turn's content. The therapeutic approach determines which "
            "framework informs the response:\n"
            "- motivational_interviewing: user exploring change, ambivalence, "
            "autonomy, stuck between options\n"
            "- cbt: user examining thoughts, beliefs, cognitive patterns, "
            "wanting practical structure or behavioral change. "
            "Concrete CBT signals: 'let's look at the evidence', 'I want "
            "to examine this thought', 'help me test this belief', 'what "
            "would be a realistic step'. The user wants to WORK ON the "
            "thought or behavior, not escape it.\n"
            "- act: user fighting or avoiding internal experiences, ruminating, "
            "needing acceptance or values reconnection. "
            "Concrete ACT signals: 'I keep fighting this feeling', 'the more "
            "I try to make it go away the worse it gets', 'I want to step "
            "back from this thought', 'I keep avoiding because of anxiety', "
            "'I'm exhausted from battling my own head', 'what do I do with "
            "this instead of fighting it'. The tell: the user is struggling "
            "WITH the experience itself — the avoidance, the rumination, "
            "or the control effort is the problem, not a specific distorted "
            "thought. If avoidance is driven by fighting internal states "
            "rather than wanting to restructure a belief, pick act over cbt.\n"
            "- dbt_skills: user in acute emotional overwhelm, needing "
            "distress tolerance or emotion regulation skills\n"
            "- grief_support: user processing loss, bereavement, missing "
            "someone, anniversary reactions\n"
            "- interpersonal_therapy: user struggling with relationships, "
            "role transitions, communication breakdowns, loneliness\n"
            "- pfa: user in acute distress needing stabilization and "
            "practical support, not deep exploration\n"
            "- none: clarifying or closing turns, or when no specific "
            "approach fits better than the default\n\n"
        ),
    ),
    (
        "output_contract",
        (
            "Return your decision in the structured schema. "
            "Keep the reasoning to one short sentence — it's for debugging, "
            "not for the user."
        ),
    ),
)


def build_therapeutic_dispatch_system_prompt() -> str:
    """Build the system prompt for the LLM dispatcher.

    Returns:
        The full system prompt string for style and approach classification.
    """

    return "".join(text for _, text in _SYSTEM_PROMPT_SECTIONS)


def build_therapeutic_dispatch_prompt(state: AgentState) -> str:
    """Build the user prompt for the LLM dispatcher.

    Args:
        state: The current agent state.

    Returns:
        The user/task prompt containing recent history, memory, and the
        current message.
    """

    history_block = format_recent_history(state, limit=6)

    working_memory = format_working_memory_entries(
        state.get("working_memory", []),
        limit=3,
    )
    if working_memory:
        memory_block = "Relevant context from past sessions:\n" + "\n".join(
            f"- {snippet}" for snippet in working_memory[:3]
        )
    else:
        memory_block = "(no working memory for this turn)"

    exercise_state = state.get("exercise_state", {}) or {}
    exercise_type = exercise_state.get("exercise_type")
    if exercise_type:
        exercise_block = (
            f"\nActive exercise: {exercise_type} "
            f"(step {exercise_state.get('exercise_step', '?')}). "
            "If the user is responding to the exercise, pick guided_exercise. "
            "If the user is exiting, wrapping up, or changing topic, pick the "
            "appropriate non-exercise response style.\n"
        )
    else:
        exercise_block = ""

    return (
        f"Recent conversation:\n{history_block}\n\n"
        f"{memory_block}\n"
        f"{exercise_block}\n"
        f"Current user message:\nuser: {state['message']}\n\n"
        "Which therapeutic response_style should handle this turn?"
    )
