from __future__ import annotations

from agent.runtime.runtime import _compact_voice_memory_context


def _compact_context(*, proactive_recall_enabled: bool) -> str:
    return _compact_voice_memory_context(
        _memory_delta(proactive_recall_enabled=proactive_recall_enabled)
    )


def _memory_delta(*, proactive_recall_enabled: bool) -> dict[str, object]:
    return {
        "working_memory": [
            {"evidence_quote": "The user is training for the Berlin marathon."}
        ],
        "session_memory": {
            "summary": "Last session focused on sleep anxiety and pacing."
        },
        "procedural_profile": {
            "proactive_recall_enabled": proactive_recall_enabled,
            "procedural_rules": [{"rule": "Use concise check-ins."}],
        },
    }


def test_voice_memory_compaction_hides_semantic_facts_when_recall_disabled() -> None:
    context = _compact_context(proactive_recall_enabled=False)

    assert "Berlin marathon" not in context


def test_voice_memory_compaction_hides_session_summary_when_recall_disabled() -> None:
    context = _compact_context(proactive_recall_enabled=False)

    assert "sleep anxiety" not in context


def test_voice_memory_compaction_keeps_procedural_rules_when_recall_disabled() -> None:
    context = _compact_context(proactive_recall_enabled=False)

    assert "Use concise check-ins." in context
    assert "Proactive saved-memory recall is disabled." in context


def test_voice_memory_compaction_exposes_recall_memory_when_enabled() -> None:
    context = _compact_context(proactive_recall_enabled=True)

    assert "Berlin marathon" in context
    assert "sleep anxiety" in context
    assert "Use concise check-ins." in context
    assert "Proactive memory recall is enabled." in context
