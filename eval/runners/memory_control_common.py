"""Shared helpers for memory-control evals."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from eval.runners.therapeutic_common import deep_update

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_NOW = "2026-05-09T00:00:00Z"


class EvalRuntime:
    """Small runtime-shaped object exposing ``.context`` for node calls."""

    def __init__(
        self,
        *,
        store: Any,
        llm_client: Any | None = None,
        memory_mode: Any | None = None,
    ) -> None:
        from agent.audit.crisis_log import InMemoryCrisisLogBackend
        from agent.memory.modes import MemoryMode
        from agent.runtime_context import WorkflowContext

        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=store,
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode or MemoryMode.LOCAL,
        )


class ScriptedTurnDispatchLLM:
    """LLM-shaped fake for memory-control trajectory evals."""

    def __init__(
        self,
        decision: Mapping[str, Any],
        *,
        preference_rule_text: str | None = None,
    ) -> None:
        self.decision = dict(decision)
        self.preference_rule_text = preference_rule_text
        self.structured_calls: dict[str, int] = {}

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "scripted response"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "scripted response"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )
        if schema_name == "TurnDispatchDecision":
            return response_schema(**self.decision)
        if schema_name == "PreferenceRuleDecision":
            if self.preference_rule_text is None:
                raise RuntimeError("Case did not script preference_rule_text.")
            return response_schema(
                rule_text=self.preference_rule_text,
                reasoning="scripted preference rule",
                confidence="high",
            )
        if schema_name == "ProceduralReconciliationDecision":
            return response_schema(
                action="append",
                replace_indexes=[],
                reason="scripted memory-control reconciliation",
                confidence="high",
            )
        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


def build_eval_state(
    *,
    message: str,
    case_id: str,
    owner_id: str | None,
    history: list[dict[str, str]] | None = None,
    state_patch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an agent graph state for a memory-control eval case."""

    from agent.graph import build_initial_state
    from agent.models import AgentInput, Message

    messages = [Message.model_validate(item) for item in history or []]
    state = dict(
        build_initial_state(
            AgentInput(
                message=message,
                user_id=owner_id,
                session_id=case_id if owner_id is not None else None,
                history=messages,
            ),
            include_input_history=True,
        )
    )
    if state_patch:
        deep_update(state, state_patch)
    return state


async def seed_memory_store(
    store: Any,
    *,
    owner_id: str,
    seed: Mapping[str, Any],
) -> None:
    """Seed a memory store with production-shaped memory records."""

    for index, item in enumerate(_list_of_mappings(seed.get("semantic_facts", []))):
        await _seed_semantic_fact(store, owner_id=owner_id, raw=item, index=index)

    for index, item in enumerate(_list_of_mappings(seed.get("episodic_sessions", []))):
        await _seed_episodic_session(store, owner_id=owner_id, raw=item, index=index)

    await _seed_procedural_profile(store, owner_id=owner_id, seed=seed)


async def memory_snapshot(store: Any, *, owner_id: str) -> dict[str, Any]:
    """Return a compact, user-visible snapshot of seeded memory."""

    from agent.memory.procedural_profile import aget_procedural_profile

    semantic_records = await store.asearch(
        (owner_id, "semantic"), query=None, limit=100
    )
    episodic_records = await store.asearch(
        (owner_id, "episodic"), query=None, limit=100
    )
    profile = await aget_procedural_profile(store, user_id=owner_id)
    return {
        "semantic_count": len(semantic_records),
        "episodic_count": len(episodic_records),
        "rule_count": len(profile.rules),
        "proactive_recall_enabled": profile.proactive_recall_enabled,
        "facts": [
            {
                "key": record.key,
                "evidence_quote": str(record.value.get("evidence_quote", "")),
                "object_identifier": _entity_identifier(record.value.get("object")),
                "predicate": str(record.value.get("predicate", "")),
            }
            for record in semantic_records
        ],
        "sessions": [
            {
                "key": record.key,
                "summary": str(record.value.get("summary", "")),
                "primary_themes": list(record.value.get("primary_themes", [])),
            }
            for record in episodic_records
        ],
        "rules": [{"id": rule.id, "rule": rule.rule} for rule in profile.rules],
    }


def grade_text_expectations(
    failures: list[str],
    *,
    text: str,
    expected: Mapping[str, Any],
    field_name: str = "response_text",
) -> None:
    """Grade simple response text containment expectations."""

    for phrase in expected.get(f"{field_name}_contains", []):
        if str(phrase).casefold() not in text.casefold():
            failures.append(f"{field_name} missing {str(phrase)!r}")
    for phrase in expected.get(f"{field_name}_not_contains", []):
        if str(phrase).casefold() in text.casefold():
            failures.append(f"{field_name} contains forbidden {str(phrase)!r}")


def grade_store_expectations(
    failures: list[str],
    *,
    snapshot: Mapping[str, Any],
    expected: Mapping[str, Any],
    prefix: str = "store_after",
) -> None:
    """Grade expected memory-store snapshot fields."""

    store_expected = expected.get(prefix)
    if not isinstance(store_expected, Mapping):
        return

    for key in (
        "semantic_count",
        "episodic_count",
        "rule_count",
        "proactive_recall_enabled",
    ):
        if key in store_expected and snapshot.get(key) != store_expected[key]:
            failures.append(
                f"{prefix}.{key}: expected {store_expected[key]!r}, "
                f"got {snapshot.get(key)!r}"
            )

    _grade_collection_phrases(
        failures,
        label=f"{prefix}.facts",
        values=[
            " ".join(
                [
                    str(item.get("evidence_quote", "")),
                    str(item.get("object_identifier", "")),
                    str(item.get("predicate", "")),
                ]
            )
            for item in _mapping_list(snapshot.get("facts", []), "facts")
        ],
        contains=store_expected.get("facts_contain", []),
        absent=store_expected.get("facts_absent", []),
    )
    _grade_collection_phrases(
        failures,
        label=f"{prefix}.sessions",
        values=[
            " ".join(
                [str(item.get("summary", "")), " ".join(item.get("primary_themes", []))]
            )
            for item in _mapping_list(snapshot.get("sessions", []), "sessions")
        ],
        contains=store_expected.get("sessions_contain", []),
        absent=store_expected.get("sessions_absent", []),
    )
    _grade_collection_phrases(
        failures,
        label=f"{prefix}.rules",
        values=[
            str(item.get("rule", ""))
            for item in _mapping_list(snapshot.get("rules", []), "rules")
        ],
        contains=store_expected.get("rules_contain", []),
        absent=store_expected.get("rules_absent", []),
    )


def expect_equal(
    failures: list[str],
    *,
    name: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    """Append a failure when an expected exact value does not match."""

    if name not in expected:
        return
    if actual != expected[name]:
        failures.append(f"{name}: expected {expected[name]!r}, got {actual!r}")


def jsonify(value: Any) -> Any:
    """Convert model-ish values into JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    return value


def optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return an optional mapping field from a JSON case."""

    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def list_of_mappings(value: Any, field_name: str) -> list[dict[str, Any]]:
    """Parse a JSON list of objects."""

    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        result.append(dict(item))
    return result


async def _seed_semantic_fact(
    store: Any,
    *,
    owner_id: str,
    raw: Mapping[str, Any],
    index: int,
) -> None:
    from agent.memory.models import EntityRef, SemanticFact

    fact_id = str(raw.get("id") or f"eval-fact-{index + 1}")
    now = str(raw.get("created_at") or _DEFAULT_NOW)
    fact = SemanticFact(
        id=fact_id,
        category=str(raw.get("category", "context")),
        subject=EntityRef(
            type=str(raw.get("subject_type", "User")),
            identifier=str(raw.get("subject_identifier", owner_id)),
        ),
        predicate=str(raw.get("predicate", "MENTIONED_IN")),
        object=EntityRef(
            type=str(raw.get("object_type", "Event")),
            identifier=str(raw.get("object_identifier", "personal context")),
        ),
        evidence_quote=str(raw["evidence_quote"]),
        confidence=str(raw.get("confidence", "high")),
        source_session_id=str(raw.get("source_session_id", "seed-session")),
        source_turn_index=int(raw.get("source_turn_index", index)),
        created_at=now,
        last_referenced_at=str(raw.get("last_referenced_at") or now),
        dormant_at=raw.get("dormant_at"),
        superseded_by=raw.get("superseded_by"),
        user_visible=bool(raw.get("user_visible", True)),
        write_reason=str(raw.get("write_reason", "seeded memory-control eval fact")),
    )
    await store.aput((owner_id, "semantic"), fact.id, fact.model_dump(mode="json"))


async def _seed_episodic_session(
    store: Any,
    *,
    owner_id: str,
    raw: Mapping[str, Any],
    index: int,
) -> None:
    from agent.memory.models import MoodArc, StoredSessionArc

    session_id = str(raw.get("session_id") or f"eval-session-{index + 1}")
    started_at = str(raw.get("started_at") or "2026-05-01T09:00:00Z")
    ended_at = str(raw.get("ended_at") or "2026-05-01T09:35:00Z")
    mood = raw.get("mood_arc") if isinstance(raw.get("mood_arc"), Mapping) else {}
    stored = StoredSessionArc(
        id=str(raw.get("id") or f"eval-episode-{index + 1}"),
        owner_id=owner_id,
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=int(raw.get("duration_seconds", 2100)),
        turn_count=int(raw.get("turn_count", 8)),
        primary_themes=[str(item) for item in raw.get("primary_themes", [])],
        summary=str(raw["summary"]),
        mood_arc=MoodArc(
            opened=str(mood.get("opened", "tense")),
            closed=str(mood.get("closed", "steadier")),
        ),
        open_loops=[str(item) for item in raw.get("open_loops", [])],
        resolved_threads=[str(item) for item in raw.get("resolved_threads", [])],
        approach_used=raw.get("approach_used"),
        approach_context=raw.get("approach_context"),
        created_at=str(raw.get("created_at") or ended_at),
        last_referenced_at=str(raw.get("last_referenced_at") or ended_at),
        write_reason=str(raw.get("write_reason", "seeded memory-control eval session")),
        crisis_level_max=int(raw.get("crisis_level_max", 0)),
    )
    await store.aput((owner_id, "episodic"), stored.id, stored.model_dump(mode="json"))


async def _seed_procedural_profile(
    store: Any,
    *,
    owner_id: str,
    seed: Mapping[str, Any],
) -> None:
    from agent.memory.models import ProceduralProfile
    from agent.memory.procedural_profile import (
        aput_procedural_profile,
        build_procedural_rule,
    )

    rules = []
    for item in _list_of_mappings(seed.get("procedural_rules", [])):
        rules.append(
            build_procedural_rule(
                rule_text=str(item["rule"]),
                evidence=[str(value) for value in item.get("evidence", [])],
                confidence=str(item.get("confidence", "high")),
                source=str(item.get("source", "manual")),
                write_reason=str(
                    item.get("write_reason", "seeded memory-control eval rule")
                ),
            )
        )
    if not rules and "proactive_recall_enabled" not in seed:
        return
    profile = ProceduralProfile(
        proactive_recall_enabled=bool(seed.get("proactive_recall_enabled", False)),
        rules=rules,
    )
    await aput_procedural_profile(store, user_id=owner_id, profile=profile)


def _entity_identifier(entity: object) -> str:
    if isinstance(entity, Mapping):
        return str(entity.get("identifier", ""))
    return ""


def _grade_collection_phrases(
    failures: list[str],
    *,
    label: str,
    values: list[str],
    contains: Any,
    absent: Any,
) -> None:
    haystack = "\n".join(values).casefold()
    contains_list = contains if isinstance(contains, list) else [contains]
    absent_list = absent if isinstance(absent, list) else [absent]
    for phrase in [item for item in contains_list if item is not None]:
        if str(phrase).casefold() not in haystack:
            failures.append(f"{label} missing {str(phrase)!r}")
    for phrase in [item for item in absent_list if item is not None]:
        if str(phrase).casefold() in haystack:
            failures.append(f"{label} contains forbidden {str(phrase)!r}")


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("memory seed entries must be lists.")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("memory seed entries must be objects.")
        result.append(item)
    return result


def _mapping_list(value: Any, field_name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        result.append(item)
    return result
