"""Shared slash-command dispatch helpers for both terminal UIs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from agent.memory.procedural_profile import (
    aclear_procedural_rules,
    adelete_procedural_rule,
    aget_procedural_profile,
)
from agent.memory.reconciliation import filter_active_semantic_records


def get_history_command_limit(
    tokens: list[str],
    *,
    default_limit: int = 6,
) -> tuple[str | None, int | None]:
    """Return usage error or parsed limit for `/history`."""

    if len(tokens) > 2:
        return ("Usage: /history [n]", None)
    if len(tokens) == 1:
        return (None, default_limit)
    try:
        return (None, max(1, int(tokens[1])))
    except ValueError:
        return ("Usage: /history [n]", None)


def get_history_window(
    history: Sequence[Any] | None,
    *,
    limit: int,
) -> tuple[str | None, list[Any] | None]:
    """Return runtime error or sliced history window for `/history`."""

    if history is None:
        return ("Runtime session is not ready.", None)
    recent = list(history[-max(1, limit) :])
    return (None, recent)


async def get_threads_command_summaries(
    tokens: list[str],
    *,
    runtime: Any,
    default_limit: int = 12,
) -> tuple[str | None, list[Any] | None]:
    """Return usage error or fetched thread summaries for `/threads`."""

    if len(tokens) > 2:
        return ("Usage: /threads [n]", None)
    limit = default_limit
    if len(tokens) == 2:
        try:
            limit = max(1, int(tokens[1]))
        except ValueError:
            return ("Usage: /threads [n]", None)
    return (None, await runtime.list_threads(limit=limit))


def parse_memory_overview_command(
    tokens: list[str],
) -> tuple[str | None, tuple[str, str | None] | None]:
    """Parse the shared `/memory status` and `/memory list` command forms.

    Returns ``(error, parsed)`` where ``parsed`` is:
    - ``("status", None)`` for `/memory` or `/memory status`
    - ``("list", "all"|"facts"|"sessions"|"rules")`` for `/memory list ...`

    Any other `/memory ...` command returns ``(None, None)`` so callers can
    continue handling UI-specific or advanced subcommands locally.
    """

    if not tokens or tokens[0] != "/memory":
        return (None, None)

    if len(tokens) == 1 or tokens[1] == "status":
        if len(tokens) > 2:
            return ("Usage: /memory status", None)
        return (None, ("status", None))

    if tokens[1] == "list":
        if len(tokens) > 3 or (
            len(tokens) == 3 and tokens[2] not in {"facts", "sessions", "rules"}
        ):
            return ("Usage: /memory list [facts|sessions|rules]", None)
        return (None, ("list", tokens[2] if len(tokens) == 3 else "all"))

    return (None, None)


def parse_search_command(
    tokens: list[str],
) -> tuple[str | None, tuple[str, str] | None]:
    """Parse `/search <history|memory|all> <query>`."""

    if not tokens or tokens[0] != "/search":
        return (None, None)
    if len(tokens) == 1:
        return ("Usage: /search <history|memory|all> <query>", None)

    mode = tokens[1]
    if mode not in {"history", "memory", "all"}:
        return ("Unknown /search subcommand. Available: history, memory, all", None)

    query = " ".join(tokens[2:]).strip()
    if not query:
        return (f"Usage: /search {mode} <query>", None)
    return (None, (mode, query))


def parse_memory_recall_command(
    tokens: list[str],
) -> tuple[str | None, bool | None]:
    """Parse `/memory recall [on|off]`."""

    if len(tokens) == 2:
        return (None, None)
    if len(tokens) != 3 or tokens[2] not in {"on", "off"}:
        return ("Usage: /memory recall on  |  /memory recall off", None)
    return (None, tokens[2] == "on")


def parse_memory_purge_crisis_days(
    tokens: list[str],
    *,
    default_days: int = 30,
) -> tuple[str | None, int | None]:
    """Parse `/memory purge-crisis [days]`."""

    if len(tokens) == 2:
        return (None, default_days)
    if len(tokens) != 3:
        return ("Usage: /memory purge-crisis [days]", None)
    try:
        days = int(tokens[2])
    except ValueError:
        return (
            f"Usage: /memory purge-crisis [days]  (got: {tokens[2]!r}, expected an integer)",
            None,
        )
    return (None, days)


def parse_memory_forget_command(
    tokens: list[str],
) -> tuple[str | None, tuple[str, str] | None]:
    """Parse `/memory forget <fact|session|rule> <n>`."""

    if len(tokens) < 3 or tokens[1] != "forget":
        return (None, None)

    kind = tokens[2]
    if kind not in {"fact", "session", "rule"}:
        return ("Usage: /memory forget <fact|session|rule> <n>", None)
    if len(tokens) != 4:
        return (f"Usage: /memory forget {kind} <n>", None)
    return (None, (kind, tokens[3]))


def get_memory_forget_index(
    index_str: str,
    *,
    kind_label: str,
) -> tuple[str | None, int | None]:
    """Return the parsed 1-indexed value or the exact CLI warning text."""

    try:
        index_1based = int(index_str)
    except ValueError:
        return (f"Usage: /memory forget {kind_label} <n>  (got: {index_str!r})", None)

    if index_1based < 1:
        return (
            f"{kind_label.capitalize()} index must be 1 or greater (got: {index_1based}).",
            None,
        )

    return (None, index_1based)


def should_save_summary_on_exit(answer: str) -> bool:
    """Return whether `/exit` should continue into summary-save flow."""

    return answer.strip().lower() != "n"


def get_yes_no_confirmation_prompt(*, subject: str) -> dict[str, Any]:
    """Return the shared y/N prompt contract for single-item destructive actions."""

    return {
        "prompt": f"[muted]Delete this {subject}?[/muted] [accent][y/N][/accent]",
        "choices": ["y", "Y", "n", "N", ""],
        "default": "n",
        "show_choices": False,
        "show_default": False,
    }


def get_exit_save_confirmation_prompt() -> dict[str, Any]:
    """Return the shared `/exit` save-summary prompt contract."""

    return {
        "prompt": "[muted]Save a session summary before exiting?[/muted] [accent][Y/n][/accent]",
        "choices": ["y", "Y", "n", "N", ""],
        "default": "y",
        "show_choices": False,
        "show_default": False,
    }


def get_typed_confirmation_prompt() -> dict[str, Any]:
    """Return the shared typed-word confirmation prompt contract."""

    return {
        "prompt": "[muted]Type the word to confirm[/muted]",
        "default": "",
        "show_default": False,
    }


def confirmation_prompt_accepts(
    answer: str,
    *,
    expected_word: str | None = None,
) -> bool:
    """Return whether a confirmation answer should be treated as accepted."""

    if expected_word is not None:
        return answer.strip() == expected_word
    return answer.strip().lower() == "y"


def get_memory_forget_target(
    items: Sequence[Any],
    *,
    index_1based: int,
    kind_title: str,
    empty_message: str,
    count_label: str,
) -> tuple[str | None, Any | None]:
    """Return the selected forget target or the exact CLI warning text."""

    if not items:
        return (empty_message, None)
    if index_1based > len(items):
        return (
            f"{kind_title} #{index_1based} does not exist "
            f"(only {len(items)} {count_label} for this thread).",
            None,
        )
    return (None, items[index_1based - 1])


def build_fact_forget_preview(
    value: dict[str, Any],
    *,
    format_entity_identifier: Any,
) -> list[str]:
    """Return the compact confirmation preview lines for one semantic fact."""

    category = str(value.get("category", "?"))
    predicate = str(value.get("predicate", "?"))
    object_id = format_entity_identifier(value.get("object"))
    quote = str(value.get("evidence_quote", ""))
    if len(quote) > 120:
        quote = quote[:117].rstrip() + "…"
    return [
        f"category:  {category}",
        f"predicate: {predicate}",
        f"object:    {object_id}",
        f"evidence:  {quote}",
    ]


def build_session_forget_preview(value: dict[str, Any]) -> list[str]:
    """Return the compact confirmation preview lines for one episodic session."""

    summary = str(value.get("summary", ""))
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "…"
    themes_value = value.get("primary_themes")
    themes_display = "—"
    if isinstance(themes_value, list) and themes_value:
        themes_display = ", ".join(str(t) for t in themes_value)
    ended_at = str(value.get("ended_at", ""))
    date_display = ended_at[:10] if len(ended_at) >= 10 else "—"
    return [
        f"date:    {date_display}",
        f"themes:  {themes_display}",
        f"summary: {summary}",
    ]


async def execute_memory_forget(
    runtime: Any,
    *,
    owner_id: str,
    kind: str,
    target: Any,
) -> dict[str, Any]:
    """Execute a previously planned `/memory forget` deletion.

    Returns a small status dict so the caller can preserve exact UI wording
    while keeping the store mutation logic shared.
    """

    store = runtime.memory_store

    if kind == "rule":
        deleted = await adelete_procedural_rule(
            store,
            user_id=owner_id,
            rule_id=target.id,
        )
        if deleted is None:
            return {"deleted": False, "reason": "changed"}
        profile, _removed_rule = deleted
        return {"deleted": True, "remaining": len(profile.rules)}

    if kind in {"fact", "session"}:
        namespace, key, _value = target
        deleted = await store.adelete(namespace, key)
        if not deleted:
            return {"deleted": False, "reason": "missing"}
        return {"deleted": True}

    raise ValueError(f"Unsupported forget kind: {kind}")


async def _get_owner_records_with_namespace(
    runtime: Any,
    *,
    owner_id: str,
    namespace_kind: str,
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    """Return owner-scoped records with namespace tuples for delete sweeps."""

    store = runtime.memory_store
    target_namespace = (owner_id, namespace_kind)
    namespaces = await store.anamespaces()
    if target_namespace not in namespaces:
        return []

    records = await store.asearch(target_namespace, query=None, limit=1000)
    if namespace_kind == "semantic":
        records = filter_active_semantic_records(records)
    return [(target_namespace, record.key, record.value) for record in records]


async def execute_memory_clear(
    runtime: Any,
    *,
    owner_id: str,
    kind: str,
) -> dict[str, int]:
    """Execute a confirmed `/memory clear` sweep for the requested scope."""

    store = runtime.memory_store
    deleted_counts: dict[str, int] = {"facts": 0, "sessions": 0, "rules": 0}

    if kind in {"facts", "all"}:
        records = await _get_owner_records_with_namespace(
            runtime,
            owner_id=owner_id,
            namespace_kind="semantic",
        )
        for namespace, key, _value in records:
            if await store.adelete(namespace, key):
                deleted_counts["facts"] += 1

    if kind in {"sessions", "all"}:
        records = await _get_owner_records_with_namespace(
            runtime,
            owner_id=owner_id,
            namespace_kind="episodic",
        )
        for namespace, key, _value in records:
            if await store.adelete(namespace, key):
                deleted_counts["sessions"] += 1

    if kind in {"rules", "all"}:
        _profile, deleted_counts["rules"] = await aclear_procedural_rules(
            store,
            user_id=owner_id,
        )

    return deleted_counts


def parse_memory_clear_command(tokens: list[str]) -> tuple[str | None, str | None]:
    """Parse `/memory clear <facts|sessions|rules|all>`."""

    if len(tokens) < 2 or tokens[1] != "clear":
        return (None, None)
    if len(tokens) != 3:
        return ("Usage: /memory clear <facts|sessions|rules|all>", None)

    kind = tokens[2]
    if kind not in {"facts", "sessions", "rules", "all"}:
        return ("Usage: /memory clear <facts|sessions|rules|all>", None)
    return (None, kind)


async def get_memory_clear_plan(
    runtime: Any,
    *,
    owner_id: str,
    kind: str,
) -> dict[str, int]:
    """Return the per-kind record counts affected by `/memory clear`."""

    store = runtime.memory_store
    counts: dict[str, int] = {"facts": 0, "sessions": 0, "rules": 0}
    if kind in {"facts", "all"}:
        counts["facts"] = await store.arecord_count((owner_id, "semantic"))
    if kind in {"sessions", "all"}:
        counts["sessions"] = await store.arecord_count((owner_id, "episodic"))
    if kind in {"rules", "all"}:
        profile = await aget_procedural_profile(store, user_id=owner_id)
        counts["rules"] = len(profile.rules)
    return counts


async def get_crisis_purge_plan(
    runtime: Any,
    *,
    days: int,
) -> tuple[int, datetime.date]:
    """Return the current crisis-log size and cutoff date for a purge."""

    total_before = await runtime.crisis_log_backend.arecord_count()
    today_utc = datetime.now(UTC).date()
    cutoff = today_utc - timedelta(days=days)
    return (total_before, cutoff)
