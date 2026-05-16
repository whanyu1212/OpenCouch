"""Generate therapeutic prompt dumps as a matrix for inspection.

This script exports:
1) Response-generator prompts for all mode/modality composition variants.
2) State-driven variants (recall toggle, procedural rules, episodic continuity).
3) Dispatcher prompt variants (inactive vs active exercise context).

Usage:
    python3 scripts/dump_prompts.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
backend_dir = REPO_ROOT / "apps" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from agent.therapeutic.dispatch import (  # noqa: E402
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)
from agent.therapeutic.prompts import (  # noqa: E402
    build_clarifying_system_prompt,
    build_closing_system_prompt,
    build_guided_exercise_system_prompt,
    build_psychoeducation_system_prompt,
    build_reflective_system_prompt,
    build_supportive_system_prompt,
    build_technique_system_prompt,
    build_therapeutic_response_prompt,
)

dump_dir = backend_dir / ".store/prompt_dumps"
dump_dir.mkdir(parents=True, exist_ok=True)

MODALITIES: list[str] = [
    "motivational_interviewing",
    "cbt",
    "act",
    "dbt_skills",
    "grief_support",
    "interpersonal_therapy",
    "pfa",
]

MODE_BUILDERS = {
    "supportive": build_supportive_system_prompt,
    "reflective": build_reflective_system_prompt,
    "clarifying": build_clarifying_system_prompt,
    "psychoeducation": build_psychoeducation_system_prompt,
    "closing": build_closing_system_prompt,
    "guided_exercise": build_guided_exercise_system_prompt,
    "technique": build_technique_system_prompt,
}

MODE_MESSAGES: dict[str, str] = {
    "supportive": "I'm exhausted and everything feels heavy today.",
    "reflective": "I keep apologizing first even when I didn't do anything wrong.",
    "clarifying": "huh?",
    "psychoeducation": "Why does my heart race when nothing is wrong?",
    "closing": "Thanks, this helped. I should go for now.",
    "guided_exercise": "Can you walk me through a grounding exercise?",
    "technique": "I want to examine the thought that I ruined everything.",
}

MODE_HISTORY: dict[str, list[dict[str, str]]] = {
    "supportive": [
        {"role": "user", "content": "Work has been crushing me lately."},
        {"role": "assistant", "content": "That sounds like a lot to carry."},
    ],
    "reflective": [
        {"role": "user", "content": "I said yes again when I wanted to say no."},
        {"role": "assistant", "content": "You noticed that pattern quickly this time."},
    ],
    "clarifying": [
        {"role": "user", "content": "I don't know how to explain this."},
        {"role": "assistant", "content": "Take your time — I'm here."},
    ],
    "psychoeducation": [
        {"role": "user", "content": "My chest gets tight out of nowhere."},
        {"role": "assistant", "content": "That sounds scary when it hits."},
    ],
    "closing": [
        {"role": "user", "content": "This was helpful."},
        {"role": "assistant", "content": "I'm glad this gave you a little room."},
    ],
    "guided_exercise": [
        {"role": "user", "content": "I'm spiraling right now."},
        {
            "role": "assistant",
            "content": "Let's ground for a minute, one step at a time.",
        },
    ],
    "technique": [
        {"role": "user", "content": "I keep thinking I messed everything up."},
        {"role": "assistant", "content": "Let's look at the exact thought together."},
    ],
}


def _base_state(*, mode: str, modality: str | None) -> dict:
    state: dict = {
        "message": MODE_MESSAGES[mode],
        "history": MODE_HISTORY[mode],
        "therapeutic_approach": modality,
        "session_memory": {"summary": ""},
        "procedural_profile": {
            "proactive_recall_enabled": False,
            "procedural_rules": [],
        },
        "working_memory": [],
        "session_progress": {"turn_count": 1},
        "exercise_state": {},
    }

    # Exercise-active context for guided exercise by default so we can inspect
    # step directives and exercise continuity behavior in prompts.
    if mode == "guided_exercise":
        state["exercise_state"] = {
            "exercise_type": "grounding_5_4_3_2_1",
            "exercise_step": 1,
            "exercise_therapeutic_approach": modality,
        }

    return state


def _apply_state_variant(state: dict, *, variant: str) -> dict:
    s = deepcopy(state)

    if variant == "default":
        return s

    if variant == "recall_on":
        s.setdefault("procedural_profile", {})
        s["procedural_profile"]["proactive_recall_enabled"] = True
        return s

    if variant == "rules_on":
        s.setdefault("procedural_profile", {})
        s["procedural_profile"]["procedural_rules"] = [
            "Keep responses brief and concrete.",
            "Avoid giving numbered lists unless I ask.",
        ]
        return s

    if variant == "episodic_on":
        s["working_memory"] = [
            {
                "type": "episodic",
                "summary": "User explored work anxiety and found that uncertainty spikes before meetings.",
                "primary_themes": ["work anxiety", "anticipatory worry"],
                "is_catch_up": False,
                "approach_used": "cbt",
                "approach_context": {
                    "thought_examined": "If I speak up, I'll look incompetent.",
                    "action_step": "Ask one question in next team meeting.",
                },
            }
        ]
        return s

    if variant == "exercise_therapeutic_approach_drift":
        # Explicitly simulate side-turn drift:
        # top-level approach says ACT, but active exercise remains CBT.
        s["therapeutic_approach"] = "act"
        s.setdefault("exercise_state", {})
        s["exercise_state"]["exercise_type"] = "simple_thought_record"
        s["exercise_state"]["exercise_step"] = 2
        s["exercise_state"]["exercise_therapeutic_approach"] = "cbt"
        return s

    raise ValueError(f"Unknown variant: {variant}")


def _composition_pairs() -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []

    # Modes without modality overlays.
    for mode in ("clarifying", "closing"):
        pairs.append((mode, None))

    # Modes with optional modality overlays.
    for mode in ("supportive", "reflective", "psychoeducation", "guided_exercise"):
        pairs.append((mode, None))
        for modality in MODALITIES:
            pairs.append((mode, modality))

    # Technique mode requires an explicit approach.
    for modality in MODALITIES:
        pairs.append(("technique", modality))

    return pairs


def _slug(mode: str, modality: str | None, variant: str) -> str:
    m = modality if modality is not None else "none"
    return f"{mode}__{m}__{variant}"


def _render_response_dump(*, mode: str, modality: str | None, variant: str) -> str:
    state = _apply_state_variant(
        _base_state(mode=mode, modality=modality), variant=variant
    )
    builder = MODE_BUILDERS[mode]
    system_prompt = builder(state)
    user_prompt = build_therapeutic_response_prompt(
        state,
        response_style=mode,
        step_directive=(
            "Continue the current exercise step." if mode == "guided_exercise" else None
        ),
    )

    lines = [
        f"VARIANT: mode={mode}, modality={modality}, state_variant={variant}",
        "",
        "===== SYSTEM PROMPT =====",
        system_prompt,
        "",
        "===== USER/TASK PROMPT =====",
        user_prompt,
        "",
    ]
    return "\n".join(lines)


def _render_dispatch_dump(*, active_exercise: bool) -> str:
    if active_exercise:
        state = {
            "message": "Can we just talk for a bit?",
            "history": [
                {"role": "user", "content": "I'm overwhelmed."},
                {
                    "role": "assistant",
                    "content": "Let's do one grounding step together.",
                },
            ],
            "working_memory": [],
            "session_progress": {"turn_count": 2},
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
                "exercise_therapeutic_approach": "dbt_skills",
            },
            "therapeutic_approach": "dbt_skills",
            "session_memory": {"summary": ""},
            "procedural_profile": {
                "proactive_recall_enabled": False,
                "procedural_rules": [],
            },
        }
        name = "dispatch__active_exercise"
    else:
        state = {
            "message": "Why do I keep doing this?",
            "history": [
                {
                    "role": "user",
                    "content": "I keep repeating this same fight pattern.",
                },
                {
                    "role": "assistant",
                    "content": "You're noticing the pattern clearly.",
                },
            ],
            "working_memory": [],
            "session_progress": {"turn_count": 1},
            "exercise_state": {"exercise_type": None, "exercise_step": None},
            "therapeutic_approach": "cbt",
            "session_memory": {"summary": ""},
            "procedural_profile": {
                "proactive_recall_enabled": False,
                "procedural_rules": [],
            },
        }
        name = "dispatch__inactive_exercise"

    system_prompt = build_therapeutic_dispatch_system_prompt()
    user_prompt = build_therapeutic_dispatch_prompt(state)

    lines = [
        f"VARIANT: {name}",
        "",
        "===== SYSTEM PROMPT =====",
        system_prompt,
        "",
        "===== USER/TASK PROMPT =====",
        user_prompt,
        "",
    ]
    return "\n".join(lines), name


def main() -> None:
    total = 0
    for stale_dump in dump_dir.glob("*.txt"):
        stale_dump.unlink()

    # Full composition matrix with default state.
    for mode, modality in _composition_pairs():
        content = _render_response_dump(mode=mode, modality=modality, variant="default")
        filename = f"{_slug(mode, modality, 'default')}.txt"
        (dump_dir / filename).write_text(content, encoding="utf-8")
        total += 1

    # State-driven variants (to exercise dynamic prompt branches).
    representative_pairs = [
        ("supportive", "motivational_interviewing"),
        ("reflective", "cbt"),
        ("psychoeducation", "act"),
        ("guided_exercise", "cbt"),
        ("technique", "cbt"),
    ]
    state_variants = ["recall_on", "rules_on", "episodic_on"]
    for mode, modality in representative_pairs:
        for variant in state_variants:
            content = _render_response_dump(
                mode=mode, modality=modality, variant=variant
            )
            filename = f"{_slug(mode, modality, variant)}.txt"
            (dump_dir / filename).write_text(content, encoding="utf-8")
            total += 1

    # Explicit guided-exercise modality-drift branch.
    content = _render_response_dump(
        mode="guided_exercise",
        modality="act",
        variant="exercise_therapeutic_approach_drift",
    )
    filename = f"{_slug('guided_exercise', 'act', 'exercise_therapeutic_approach_drift')}.txt"
    (dump_dir / filename).write_text(content, encoding="utf-8")
    total += 1

    # Dispatcher prompt variants.
    for active in (False, True):
        content, name = _render_dispatch_dump(active_exercise=active)
        (dump_dir / f"{name}.txt").write_text(content, encoding="utf-8")
        total += 1

    print(f"Wrote {total} prompt dump files to {dump_dir}")


if __name__ == "__main__":
    main()
