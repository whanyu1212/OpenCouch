"""Argument parser helpers for the OpenCouch Textual TUI."""

from __future__ import annotations

import argparse

from agent.runtime import (
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments for the Textual TUI.

    Covers the flags common to all local terminal surfaces.

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
