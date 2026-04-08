"""Session-context helpers for long-horizon conversations."""

from __future__ import annotations

import re


MAX_HISTORY_TURNS = 8
MAX_ACTIVE_CONCERNS = 3
MAX_OPEN_LOOPS = 3

META_TURN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow does this work\b", re.IGNORECASE),
    re.compile(r"\bwhat can you help with\b", re.IGNORECASE),
    re.compile(r"\bwhat can you do(?: for me)?\b", re.IGNORECASE),
    re.compile(r"\bhow can you help(?: me)?\b", re.IGNORECASE),
    re.compile(r"\bwhat are you\b", re.IGNORECASE),
    re.compile(r"\bwho are you\b", re.IGNORECASE),
    re.compile(r"\bi'?m new here\b", re.IGNORECASE),
    re.compile(r"\bfirst time here\b", re.IGNORECASE),
)

SESSION_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\b(i want|let'?s|can we|help me)\b.*\b(cbt|thought record|reframe)\b",
            re.IGNORECASE,
        ),
        "guided_cbt_work",
        "explicit",
    ),
    (
        re.compile(
            r"\b(i want|let'?s|can we|help me)\b.*\b(grounding|breathe|breathing|calm down)\b",
            re.IGNORECASE,
        ),
        "grounding_or_calm_down",
        "explicit",
    ),
    (
        re.compile(
            r"\b(i want|let'?s|can we|help me)\b.*\b(reflect|reflection|patterns?|make sense)\b",
            re.IGNORECASE,
        ),
        "reflection_and_pattern_finding",
        "explicit",
    ),
    (
        re.compile(
            r"\b(i just want to vent|just let me vent|i don't want advice)\b",
            re.IGNORECASE,
        ),
        "just_need_to_vent",
        "explicit",
    ),
    (
        re.compile(
            r"\b(i don't need advice|don't need advice right now)\b", re.IGNORECASE
        ),
        "just_need_to_vent",
        "explicit",
    ),
    (
        re.compile(
            r"\b(i want support|just support|talk this through|be heard)\b",
            re.IGNORECASE,
        ),
        "supportive_conversation",
        "explicit",
    ),
    (
        re.compile(
            r"\b(explain|help me understand|what is|why does)\b.*\b(anxiety|panic|stress|body|nervous system|react like this)\b",
            re.IGNORECASE,
        ),
        "psychoeducation",
        "explicit",
    ),
    (
        re.compile(
            r"\b(explain|help me understand)\b.*\b(burned? out|exhaustion|overwhelm)\b",
            re.IGNORECASE,
        ),
        "psychoeducation",
        "explicit",
    ),
)

CONCERN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "overwhelm or stress",
        (r"\boverwhelm", r"\bstress", r"\bburn(?:ed)? out", r"\bdrain(?:ed|ing)?"),
    ),
    (
        "anxiety or rumination",
        (r"\banxious", r"\banxiety", r"\bruminat", r"\bcan't switch off", r"\bspiral"),
    ),
    ("grief or loss", (r"\bgrief", r"\bloss", r"\bdied", r"\bfuneral", r"\bbereave")),
    (
        "self-worth or shame",
        (r"\bworthless", r"\bfailure", r"\bshame", r"\bguilt", r"\bnot enough"),
    ),
    (
        "relationship strain",
        (
            r"\bpartner",
            r"\brelationship",
            r"\bfriend",
            r"\bfamily",
            r"\bsister",
            r"\bbrother",
            r"\bargument",
        ),
    ),
    (
        "work or school pressure",
        (r"\bwork", r"\bjob", r"\bmanager", r"\bschool", r"\bclass", r"\bproject"),
    ),
    (
        "sleep or exhaustion",
        (r"\bsleep", r"\binsomnia", r"\btired", r"\bexhaust", r"\brest"),
    ),
)

GOAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bhelp me understand\b", re.IGNORECASE),
        "understand a recurring pattern",
    ),
    (
        re.compile(r"\bwhat patterns do you notice\b", re.IGNORECASE),
        "reflect on patterns",
    ),
    (
        re.compile(r"\bgrounding\b|\bcalm down\b|\bbreathing\b", re.IGNORECASE),
        "feel calmer right now",
    ),
    (
        re.compile(r"\bthought record\b|\breframe\b", re.IGNORECASE),
        "work through a structured exercise",
    ),
    (
        re.compile(
            r"\b(explain|help me understand|what is|why does)\b.*\b(anxiety|panic|stress|body|nervous system|react like this)\b",
            re.IGNORECASE,
        ),
        "understand what may be happening in mind and body",
    ),
    (
        re.compile(
            r"\bhow does this work\b|\bwhat can you help with\b|\bnew here\b",
            re.IGNORECASE,
        ),
        "understand how to use OpenCouch",
    ),
)

OPEN_LOOP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhy do i keep\b", re.IGNORECASE),
    re.compile(r"\bhelp me understand\b", re.IGNORECASE),
    re.compile(r"\bwhat should i do\b", re.IGNORECASE),
    re.compile(r"\bcan you help me\b", re.IGNORECASE),
    re.compile(r"\bi want to figure out\b", re.IGNORECASE),
)

NEGATED_GROUNDING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdon't need grounding\b", re.IGNORECASE),
    re.compile(r"\bdo not need grounding\b", re.IGNORECASE),
    re.compile(r"\bdo not think i need grounding\b", re.IGNORECASE),
    re.compile(r"\bi don't think i need grounding\b", re.IGNORECASE),
)


def _user_turns(history: list[dict[str, str]]) -> list[str]:
    """Return the user-authored contents from serialized history."""

    return [
        turn.get("content", "").strip()
        for turn in history
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]


def _is_meta_turn(text: str) -> bool:
    """Return whether a user turn is product-orientation rather than therapeutic."""

    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in META_TURN_PATTERNS)


def _therapeutic_user_turns(history: list[dict[str, str]]) -> list[str]:
    """Return user turns that should influence therapeutic session context."""

    return [turn for turn in _user_turns(history) if not _is_meta_turn(turn)]


def trim_history(
    history: list[dict[str, str]], *, limit: int = MAX_HISTORY_TURNS
) -> list[dict[str, str]]:
    """Trim serialized history to the most recent turns.

    Args:
        history: Serialized conversation turns.
        limit: Maximum number of recent turns to keep.

    Returns:
        The most recent subset of serialized turns.
    """

    return history[-limit:] if len(history) > limit else history


def extract_active_concerns(
    history: list[dict[str, str]],
    *,
    current_message: str,
) -> list[str]:
    """Extract the active user concerns from the session.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.

    Returns:
        Up to three normalized concern labels.
    """

    therapeutic_turns = _therapeutic_user_turns(history)
    if current_message.strip() and not _is_meta_turn(current_message):
        therapeutic_turns.append(current_message.strip())
    text = " ".join(therapeutic_turns).lower()
    concerns: list[str] = []

    for label, patterns in CONCERN_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            concerns.append(label)
        if len(concerns) >= MAX_ACTIVE_CONCERNS:
            break

    if concerns:
        return concerns

    recent_user_turns = _therapeutic_user_turns(history)[-2:]
    if current_message.strip() and not _is_meta_turn(current_message):
        recent_user_turns.append(current_message.strip())

    fallback = [turn[:80].rstrip(" .!?") for turn in recent_user_turns if turn]
    return fallback[-MAX_ACTIVE_CONCERNS:]


def extract_open_loops(
    history: list[dict[str, str]],
    *,
    current_message: str,
) -> list[str]:
    """Extract unresolved themes or asks from the session.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.

    Returns:
        Up to three concise unresolved threads.
    """

    loops: list[str] = []
    seen: set[str] = set()

    candidate_turns = _therapeutic_user_turns(history)[-6:]
    if current_message.strip() and not _is_meta_turn(current_message):
        candidate_turns.append(current_message.strip())

    for turn in candidate_turns:
        if not turn:
            continue
        lowered = turn.lower()
        if (
            any(pattern.search(lowered) for pattern in OPEN_LOOP_PATTERNS)
            or "?" in turn
        ):
            candidate = turn[:96].rstrip(" .!?")
            if candidate and candidate not in seen:
                loops.append(candidate)
                seen.add(candidate)
        if len(loops) >= MAX_OPEN_LOOPS:
            break

    return loops


def infer_current_goal(
    history: list[dict[str, str]],
    *,
    current_message: str,
) -> str | None:
    """Infer the user's current session goal.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.

    Returns:
        A concise goal string when one is detectable.
    """

    if current_message.strip() and _is_meta_turn(current_message):
        return None

    therapeutic_turns = _therapeutic_user_turns(history)
    text = current_message.strip() or (
        therapeutic_turns[-1] if therapeutic_turns else ""
    )
    if not text:
        return None

    blocked_goal = (
        "feel calmer right now"
        if any(pattern.search(text) for pattern in NEGATED_GROUNDING_PATTERNS)
        else None
    )

    for pattern, goal in GOAL_PATTERNS:
        if goal == blocked_goal:
            continue
        if pattern.search(text):
            return goal

    lowered = text.lower()
    if "help me" in lowered or "can you" in lowered:
        return text[:96].rstrip(" .!?")
    return None


def build_session_summary(
    history: list[dict[str, str]],
    *,
    current_message: str,
    active_concerns: list[str],
    current_goal: str | None,
) -> str:
    """Build a compact summary of the current session.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.
        active_concerns: Extracted concern labels for the session.
        current_goal: Inferred current user goal.

    Returns:
        A deterministic rolling session summary.
    """

    recent_user_turns = _therapeutic_user_turns(history)[-3:]
    if current_message.strip() and not _is_meta_turn(current_message):
        recent_user_turns.append(current_message.strip())

    snippets = [turn[:90].rstrip(" .!?") for turn in recent_user_turns if turn]
    parts: list[str] = []

    if active_concerns:
        parts.append(f"Active concerns include {', '.join(active_concerns)}.")
    if current_goal:
        parts.append(f"The current goal seems to be to {current_goal}.")
    if snippets:
        parts.append(f"Recent user themes: {' | '.join(snippets[-3:])}.")

    return (
        " ".join(parts) if parts else "New conversation with no prior session context."
    )


def infer_session_intent(
    history: list[dict[str, str]],
    *,
    current_message: str,
) -> tuple[str | None, str | None]:
    """Infer the user's overall session intent.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.

    Returns:
        A tuple of `(intent, source)` where source is `explicit`, `inferred`, or `None`.
    """

    text = current_message.strip()
    if not text:
        return None, None
    if _is_meta_turn(text):
        return None, None

    for pattern, intent, source in SESSION_INTENT_PATTERNS:
        if pattern.search(text):
            return intent, source

    lowered = text.lower()
    grounding_blocked = any(
        pattern.search(text) for pattern in NEGATED_GROUNDING_PATTERNS
    )
    if any(term in lowered for term in ("cbt", "thought record", "reframe")):
        return "guided_cbt_work", "inferred"
    if not grounding_blocked and any(
        term in lowered for term in ("grounding", "breathing", "calm down")
    ):
        return "grounding_or_calm_down", "inferred"
    if any(
        term in lowered
        for term in (
            "pattern",
            "reflect",
            "make sense",
            "understand why i keep",
            "understanding why i keep",
            "what keeps happening",
            "is there a theme",
            "do you see a connection",
        )
    ):
        return "reflection_and_pattern_finding", "inferred"
    if any(
        term in lowered
        for term in (
            "what is anxiety",
            "why does my body",
            "why do i react like this",
            "nervous system",
            "stress response",
            "what's happening in my body",
            "how does anxiety work",
            "how does stress work",
            "what is burnout",
        )
    ):
        return "psychoeducation", "inferred"
    if "is it normal" in lowered and any(
        term in lowered
        for term in ("anxiety", "panic", "stress", "body", "shake", "shaking")
    ):
        return "psychoeducation", "inferred"
    if "vent" in lowered:
        return "just_need_to_vent", "inferred"
    if any(
        term in lowered
        for term in ("talk", "support", "listen", "rough day", "overwhelmed")
    ):
        return "supportive_conversation", "inferred"
    if any(
        term in lowered
        for term in (
            "anxious",
            "anxiety",
            "stressed",
            "stress",
            "drained",
            "exhausted",
            "tired",
            "rest",
            "lonely",
            "upset",
            "sad",
        )
    ):
        return "supportive_conversation", "inferred"
    return None, None


def update_session_intent(
    history: list[dict[str, str]],
    *,
    current_message: str,
    existing_intent: str | None,
    existing_source: str | None,
) -> tuple[str | None, str | None]:
    """Update the sticky session intent with explicit override rules.

    Args:
        history: Serialized conversation turns.
        current_message: Current inbound user message.
        existing_intent: Previously stored session intent, if any.
        existing_source: Source for the previous session intent, if any.

    Returns:
        The next `(intent, source)` tuple for the session.
    """

    next_intent, next_source = infer_session_intent(
        history,
        current_message=current_message,
    )
    if next_source == "explicit":
        return next_intent, next_source
    if existing_intent and existing_source == "explicit":
        return existing_intent, existing_source
    if (
        existing_intent
        in {
            "reflection_and_pattern_finding",
            "psychoeducation",
            "guided_cbt_work",
            "grounding_or_calm_down",
        }
        and next_source == "inferred"
        and next_intent == "supportive_conversation"
    ):
        return existing_intent, existing_source
    if next_source == "inferred" and next_intent == "supportive_conversation":
        for prior_turn in reversed(_therapeutic_user_turns(history)[-3:]):
            prior_intent, prior_source = infer_session_intent(
                history,
                current_message=prior_turn,
            )
            if prior_intent in {
                "reflection_and_pattern_finding",
                "psychoeducation",
                "guided_cbt_work",
                "grounding_or_calm_down",
            }:
                return prior_intent, prior_source
    if next_intent is not None:
        return next_intent, next_source
    if existing_intent is None:
        for prior_turn in reversed(_user_turns(history)[-3:]):
            prior_intent, prior_source = infer_session_intent(
                history,
                current_message=prior_turn,
            )
            if prior_intent is not None:
                return prior_intent, prior_source
    return existing_intent, existing_source


def infer_session_stage_deterministically(
    *,
    previous_stage: str | None,
    session_intent: str | None,
    current_message: str,
    turn_count: int,
    recent_modes: list[str],
    needs_crisis_response: bool,
    needs_clarification: bool,
) -> tuple[str, str]:
    """Infer the session stage from deterministic conversational signals.

    Args:
        previous_stage: Previously stored session stage, if any.
        session_intent: Current session intent, if any.
        current_message: Current inbound user message.
        turn_count: Count of user turns including the current turn.
        recent_modes: Recent selected response modes from the transcript/session.
        needs_crisis_response: Whether the crisis path is required for this turn.
        needs_clarification: Whether a safety check is required for this turn.

    Returns:
        A tuple of `(stage, reason)` for the current turn.
    """

    text = current_message.lower()

    if any(
        phrase in text
        for phrase in (
            "before we wrap up",
            "wrap this up",
            "wrap up",
            "one last thing",
            "i need to go",
            "that's enough for today",
            "can you summarize this",
        )
    ):
        return "closing", "Detected explicit wrap-up language from the user."

    if needs_crisis_response or needs_clarification:
        return (
            previous_stage or "opening",
            "Safety-sensitive turn; keep ordinary stage progression conservative.",
        )

    intent_family = (
        "structured_work"
        if session_intent in {"guided_cbt_work", "grounding_or_calm_down"}
        else "exploratory_support"
    )

    if turn_count <= 2 and (
        not recent_modes
        or recent_modes[-1] in {"orientation", "supportive_conversation"}
    ):
        return "opening", "Early turn count with orientation/support pattern."

    if any(
        phrase in text
        for phrase in (
            "that helped",
            "that makes sense",
            "i feel calmer",
            "i feel a bit better",
            "what should i do next",
            "what should i try this week",
        )
    ):
        return (
            "stabilizing",
            "User language suggests integration, grounding, or next-step planning.",
        )

    if intent_family == "structured_work":
        if turn_count >= 3 and session_intent == "guided_cbt_work":
            return (
                "deepening",
                "Structured CBT intent with enough turns to move into active work.",
            )
        if recent_modes and recent_modes[-1] == "guided_exercise":
            return (
                "deepening",
                "Structured-work intent with guided exercise in progress.",
            )
        if previous_stage == "deepening":
            return (
                "deepening",
                "Preserving deepening stage for ongoing structured work.",
            )
    else:
        if (
            any(
                mode in {"pattern_reflection", "supportive_conversation"}
                for mode in recent_modes[-2:]
            )
            and turn_count >= 3
        ):
            return (
                "deepening",
                "Reflective/supportive pattern suggests deeper exploration.",
            )

    if previous_stage in {"deepening", "stabilizing"} and turn_count >= 6:
        return (
            "stabilizing",
            "Later session turn count with no stronger cue to deepen further.",
        )

    return (
        previous_stage or "opening",
        "No stronger stage cue detected; preserving the current stage.",
    )
