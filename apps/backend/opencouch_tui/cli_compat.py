"""Shared legacy-CLI parser and startup prompt helpers."""

from __future__ import annotations

import argparse
from typing import Callable

from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
)
from config import PersistenceBackend, get_settings


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments used by both the Rich CLI and Textual TUI.

    Covers the seven flags that are identical across surfaces: mode,
    thread-id, user-id, sqlite-path, memory-sqlite-path,
    crisis-log-sqlite-path, and response-model-tier.

    Args:
        parser: Argument parser to extend in-place.
    """

    parser.add_argument(
        "--mode",
        choices=["deterministic", "hybrid", "auto"],
        default="auto",
        help="How to resolve the LLM client for crisis classification.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Stable thread identifier to resume a prior local conversation.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=(
            "Stable owner identifier for long-term memory (semantic facts, "
            "episodic arcs, procedural rules). When set, memory writes are "
            "namespaced by this user_id rather than the thread_id, so "
            "switching threads preserves memory across sessions. Only "
            "meaningful in persistent memory mode — guest mode has no "
            "long-term storage to namespace. When omitted, falls back to "
            "the thread_id for backward compatibility."
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(DEFAULT_THREAD_DB_PATH),
        help=(
            "Legacy SQLite path for persisted session state. Deprecated "
            "for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--memory-sqlite-path",
        default=str(DEFAULT_MEMORY_DB_PATH),
        help=(
            "Legacy SQLite path for the memory store (semantic facts + "
            "episodic arcs). Deprecated for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--crisis-log-sqlite-path",
        default=str(DEFAULT_CRISIS_LOG_DB_PATH),
        help=(
            "Legacy SQLite path for the crisis log (safety audit trail). "
            "Deprecated for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--response-model-tier",
        "--response-tier",
        dest="response_model_tier",
        choices=["fast", "quality"],
        default="fast",
        help=(
            "Text response tier for therapeutic prose generation. "
            "'fast' favors lower latency; 'quality' favors richer replies."
        ),
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the legacy OpenCouch CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the interactive OpenCouch CLI.",
        epilog=(
            "Example: uv run python -m opencouch_cli --mode auto "
            "--thread-id local-demo --sqlite-path .opencouch_threads.sqlite3"
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--memory-mode",
        choices=["guest", "persistent", "ask"],
        default="ask",
        help="Local memory behavior: guest (ephemeral), persistent (configured backend), or ask at startup.",
    )
    parser.add_argument(
        "--disable-tracing",
        action="store_true",
        default=False,
        help=(
            "Disable optional tracing integrations for this CLI run. "
            "Equivalent to setting OPENCOUCH_DISABLE_TRACING=1."
        ),
    )
    return parser


def persistent_mode_hint(persistence_backend: PersistenceBackend) -> str:
    """Return backend-aware copy for the persistent memory choice."""

    if persistence_backend == "postgres":
        return "save memory using Postgres"
    return "save memory using SQLite"


def resolve_cli_memory_mode(
    memory_mode: str,
    *,
    console: Console,
    prompt_ask: Callable[..., str] = Prompt.ask,
    persistence_backend: PersistenceBackend | None = None,
    render_warning: Callable[[str], None] | None = None,
) -> str:
    """Resolve legacy CLI memory mode, prompting when ``ask`` is requested."""

    if memory_mode in {"guest", "persistent"}:
        return memory_mode

    backend = persistence_backend or get_settings().persistence_backend
    persistent_hint = persistent_mode_hint(backend)
    console.print()
    console.print(Rule(style="panel", characters="─"))
    console.print("  [primary]Choose Memory Mode[/primary]", highlight=False)
    console.print()
    console.print(
        "  [accent]1[/accent]  [info]Guest Mode[/info]  [hint]— private, in-memory only[/hint]"
    )
    console.print(
        f"  [accent]2[/accent]  [info]Persistent Mode[/info]  [hint]— {persistent_hint}[/hint]"
    )
    console.print()
    console.print(Rule(style="panel", characters="─"))
    while True:
        choice = prompt_ask("  [muted]select[/muted]", default="1").strip()
        if choice == "1":
            return "guest"
        if choice == "2":
            return "persistent"
        if render_warning is not None:
            render_warning("Please choose 1 (guest) or 2 (persistent).")
