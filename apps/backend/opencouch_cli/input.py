"""Prompt-toolkit input helpers for the interactive CLI."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.prompt import Prompt

from collections import deque

from opencouch_cli.commands import ALIASES, child_commands

PROMPT_THEMES: dict[str, dict[str, str]] = {
    "mono": {
        "prompt.symbol": "#8FAE9D bold",
        "prompt.user": "#A7BFA3 bold",
        "bottom-toolbar": "bg:#1E2822 #B8D4C8",
        "bottom-toolbar.text": "bg:#1E2822 #B8D4C8",
        "toolbar": "bg:#1E2822 #B8D4C8",
        "toolbar.label": "bg:#1E2822 #B8D4C8",
        "toolbar.punct": "bg:#1E2822 #B8D4C8",
        "toolbar.separator": "bg:#1E2822 #B8D4C8",
        "toolbar.mode": "bg:#1E2822 #B8D4C8",
        "toolbar.memory": "bg:#1E2822 #B8D4C8",
        "toolbar.response": "bg:#1E2822 #B8D4C8",
        "toolbar.thread": "bg:#1E2822 #B8D4C8",
        "toolbar.pending": "bg:#1E2822 #B8D4C8",
        "toolbar.last": "bg:#1E2822 #B8D4C8",
        "toolbar.hint": "bg:#1E2822 #B8D4C8",
        "completion-menu.completion": "bg:#1E2822 #D8DDD8",
        "completion-menu.completion.current": "bg:#3D4844 #FFFFFF bold",
        "completion-menu.meta.completion": "bg:#1E2822 #7B817C",
        "completion-menu.meta.completion.current": "bg:#3D4844 #FFFFFF",
    },
    "contrast": {
        "prompt.symbol": "#9BC4B0 bold",
        "prompt.user": "#D5E7DE bold",
        "bottom-toolbar": "bg:#121614 #D5E7DE",
        "bottom-toolbar.text": "bg:#121614 #D5E7DE",
        "toolbar": "bg:#121614 #D5E7DE",
        "toolbar.label": "bg:#121614 #9FB6AA",
        "toolbar.punct": "bg:#121614 #8AA398",
        "toolbar.separator": "bg:#121614 #6A8178",
        "toolbar.mode": "bg:#121614 #8FD0F3 bold",
        "toolbar.memory": "bg:#121614 #9FE3BB bold",
        "toolbar.response": "bg:#121614 #F0D192 bold",
        "toolbar.thread": "bg:#121614 #D5E7DE",
        "toolbar.pending": "bg:#121614 #F3B78A bold",
        "toolbar.last": "bg:#121614 #D5E7DE",
        "toolbar.hint": "bg:#121614 #9FE3BB bold",
        "completion-menu.completion": "bg:#181F1B #E9EFEB",
        "completion-menu.completion.current": "bg:#36423C #FFFFFF bold",
        "completion-menu.meta.completion": "bg:#181F1B #9FB0A8",
        "completion-menu.meta.completion.current": "bg:#36423C #FFFFFF",
    },
    "calm": {
        "prompt.symbol": "#8FAE9D bold",
        "prompt.user": "#A7BFA3 bold",
        "bottom-toolbar": "bg:#1A211D #B9C8BF",
        "bottom-toolbar.text": "bg:#1A211D #B9C8BF",
        "toolbar": "bg:#1A211D #B9C8BF",
        "toolbar.label": "bg:#1A211D #95A69C",
        "toolbar.punct": "bg:#1A211D #82938A",
        "toolbar.separator": "bg:#1A211D #6F8178",
        "toolbar.mode": "bg:#1A211D #AFC5D3",
        "toolbar.memory": "bg:#1A211D #B3C7A9",
        "toolbar.response": "bg:#1A211D #C5BFA3",
        "toolbar.thread": "bg:#1A211D #C4CEC9",
        "toolbar.pending": "bg:#1A211D #C9A56D",
        "toolbar.last": "bg:#1A211D #B9C8BF",
        "toolbar.hint": "bg:#1A211D #AFC5B5",
        "completion-menu.completion": "bg:#232C27 #D8DDD8",
        "completion-menu.completion.current": "bg:#45524B #FFFFFF bold",
        "completion-menu.meta.completion": "bg:#232C27 #8A938E",
        "completion-menu.meta.completion.current": "bg:#45524B #FFFFFF",
    },
}


@dataclass(frozen=True, slots=True)
class PromptToolbarState:
    """Session metadata shown in the prompt-toolkit bottom toolbar."""

    resolved_mode: str
    memory_mode: str
    response_model_tier: str
    thread_id: str
    user_id: str | None
    pending_status: str | None = None
    ui_mode: str = "full"
    last_action: str | None = None


class SlashCommandCompleter(Completer):
    """Complete OpenCouch slash commands and static command arguments."""

    def get_completions(self, document: Document, complete_event):
        """Yield prompt-toolkit completions for slash-command input.

        Args:
            document (Document): Current prompt document.
            complete_event: Prompt-toolkit completion event metadata.

        Yields:
            Completion: Matching slash command or argument completion.
        """

        _ = complete_event
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parent, prefix, start_position = _completion_context(text)

        # At root level, prepend recent commands and alias hints.
        if not parent or parent == ("/",):
            yield from self._recent_completions(prefix, start_position)

        for command in child_commands(parent):
            token = command.path[len(parent)]
            if prefix and not token.startswith(prefix):
                continue
            yield Completion(
                token,
                start_position=start_position,
                display=token,
                display_meta=_completion_meta(command),
            )

        # Show aliases at root level after main commands.
        if not parent or parent == ("/",):
            yield from self._alias_completions(prefix, start_position)

    def _recent_completions(self, prefix: str, start_position: int):
        """Yield completions for recently used commands.

        Args:
            prefix (str): Current typing prefix.
            start_position (int): Cursor offset for replacement.

        Yields:
            Completion: Recent command completions.
        """

        for cmd in _RECENT_COMMANDS:
            # cmd is like "/help" — the token is the full command including slash
            token = cmd
            if prefix and not token.startswith(prefix):
                continue
            yield Completion(
                token,
                start_position=start_position,
                display=f"↻ {token}",
                display_meta="(recent)",
            )

    def _alias_completions(self, prefix: str, start_position: int):
        """Yield completions for command aliases.

        Args:
            prefix (str): Current typing prefix.
            start_position (int): Cursor offset for replacement.

        Yields:
            Completion: Alias completions.
        """

        for alias, target in ALIASES.items():
            if prefix and not alias.startswith(prefix):
                continue
            yield Completion(
                alias,
                start_position=start_position,
                display=alias,
                display_meta=f"→ {target}",
            )


def _completion_meta(command) -> str:
    """Return completion metadata text with optional usage hint and example.

    Args:
        command: Slash command metadata entry.

    Returns:
        str: Completion menu meta text.
    """

    parts = [command.description]
    if command.example:
        parts.append(f"e.g. {command.example}")
    elif command.usage and command.usage != " ".join(command.path):
        parts.append(f"({command.usage})")
    return "  ".join(parts)


def _completion_context(text: str) -> tuple[tuple[str, ...], str, int]:
    """Return parent path, active token prefix, and replacement offset.

    Args:
        text (str): Text before the cursor.

    Returns:
        tuple[tuple[str, ...], str, int]: Parent command tokens, active prefix,
            and prompt-toolkit start position for the completion.
    """

    if text.endswith(" "):
        return tuple(text.split()), "", 0

    tokens = text.split()
    if not tokens:
        return (), "", 0

    prefix = tokens[-1]
    return tuple(tokens[:-1]), prefix, -len(prefix)


_INPUT_HISTORY = InMemoryHistory()
_PROMPT_SESSION: PromptSession[str] | None = None
_ACTIVE_PROMPT_THEME = "mono"

# Recent commands for the quick-picker (most recent first, max 5).
_RECENT_COMMANDS: deque[str] = deque(maxlen=5)


def record_recent_command(command: str) -> None:
    """Record a command as recently used (for the quick-picker).

    Args:
        command (str): The canonical command name (e.g. "/help").
    """

    # Avoid duplicates: remove if already present, then prepend.
    if command in _RECENT_COMMANDS:
        _RECENT_COMMANDS.remove(command)
    _RECENT_COMMANDS.appendleft(command)


def available_prompt_themes() -> tuple[str, ...]:
    """Return supported prompt theme preset names.

    Returns:
        tuple[str, ...]: Supported prompt theme names.
    """

    return tuple(PROMPT_THEMES.keys())


def set_prompt_theme(theme_name: str) -> bool:
    """Set the active prompt theme and rebuild the prompt session.

    Args:
        theme_name (str): Target theme name.

    Returns:
        bool: True when updated, False for unknown theme names.
    """

    global _ACTIVE_PROMPT_THEME, _PROMPT_SESSION
    if theme_name not in PROMPT_THEMES:
        return False
    _ACTIVE_PROMPT_THEME = theme_name
    # Session caches Style at construction time; rebuild to apply new colors.
    _PROMPT_SESSION = None
    return True


def _prompt_style() -> Style:
    """Return the prompt-toolkit style for OpenCouch input controls.

    Returns:
        Style: Prompt-toolkit style configuration.
    """

    return Style.from_dict(PROMPT_THEMES[_ACTIVE_PROMPT_THEME])


def _key_bindings() -> KeyBindings:
    """Return input key bindings for the enhanced REPL.

    Returns:
        KeyBindings: Prompt-toolkit key-binding registry.
    """

    bindings = KeyBindings()

    @bindings.add("c-l")
    def _clear(event) -> None:
        """Clear the prompt surface.

        Args:
            event: Prompt-toolkit key event.

        Returns:
            None.
        """

        event.app.output.erase_screen()
        event.app.output.cursor_goto(0, 0)

    @bindings.add("/")
    def _slash(event) -> None:
        """Open the slash-command menu as soon as `/` starts the input.

        Args:
            event: Prompt-toolkit key event.

        Returns:
            None.
        """

        _insert_slash_and_maybe_complete(event)

    return bindings


def _insert_slash_and_maybe_complete(event) -> None:
    """Insert `/` and open completions when it begins the prompt.

    Args:
        event: Prompt-toolkit key event.

    Returns:
        None.
    """

    buffer = event.current_buffer
    should_open_menu = buffer.document.text_before_cursor == ""
    buffer.insert_text("/")
    if should_open_menu:
        buffer.start_completion(select_first=False)


def _prompt_session() -> PromptSession[str]:
    """Return the process-wide prompt session.

    Returns:
        PromptSession[str]: Reused session with history and slash completion.
    """

    global _PROMPT_SESSION
    if _PROMPT_SESSION is None:
        _PROMPT_SESSION = PromptSession(
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            enable_history_search=True,
            history=_INPUT_HISTORY,
            key_bindings=_key_bindings(),
            reserve_space_for_menu=8,
            style=_prompt_style(),
        )
    return _PROMPT_SESSION


def _prompt_message() -> AnyFormattedText:
    """Return the styled prompt prefix.

    Returns:
        AnyFormattedText: Prompt-toolkit formatted prompt text.
    """

    return FormattedText(
        [
            ("class:prompt.symbol", "  · "),
            ("class:prompt.user", "you "),
        ]
    )


def _clip_toolbar_message(message: str, *, max_length: int = 42) -> str:
    """Clip a toolbar message to avoid prompt overflow.

    Args:
        message (str): Raw message.
        max_length (int): Maximum visible length.

    Returns:
        str: Clipped message.
    """

    if len(message) <= max_length:
        return message
    return message[: max_length - 1].rstrip() + "…"


def prompt_toolbar(state: PromptToolbarState) -> AnyFormattedText:
    """Return the bottom toolbar for the current prompt.

    Args:
        state (PromptToolbarState): Current session metadata.

    Returns:
        AnyFormattedText: Prompt-toolkit formatted toolbar text.
    """

    identity = state.user_id or state.thread_id

    if state.ui_mode == "compact":
        fragments: list[tuple[str, str]] = [
            ("class:toolbar.label", "  "),
            ("class:toolbar.mode", state.resolved_mode),
            ("class:toolbar.separator", " · "),
            ("class:toolbar.memory", state.memory_mode),
            ("class:toolbar.separator", " · "),
            ("class:toolbar.response", state.response_model_tier),
            ("class:toolbar.separator", " · "),
            ("class:toolbar.thread", identity),
        ]
        if state.pending_status:
            fragments.extend(
                [
                    ("class:toolbar.separator", " · "),
                    ("class:toolbar.pending", state.pending_status),
                ]
            )
        if state.last_action:
            fragments.extend(
                [
                    ("class:toolbar.separator", " · "),
                    ("class:toolbar.last", _clip_toolbar_message(state.last_action)),
                ]
            )
        fragments.extend(
            [
                ("class:toolbar.separator", "   "),
                ("class:toolbar.hint", "/"),
                ("class:toolbar.label", " commands"),
            ]
        )
        return FormattedText(fragments)

    fragments = [
        ("class:toolbar.label", "  mode"),
        ("class:toolbar.punct", ": "),
        ("class:toolbar.mode", state.resolved_mode),
        ("class:toolbar.separator", "   "),
        ("class:toolbar.label", "memory"),
        ("class:toolbar.punct", ": "),
        ("class:toolbar.memory", state.memory_mode),
        ("class:toolbar.separator", "   "),
        ("class:toolbar.label", "response"),
        ("class:toolbar.punct", ": "),
        ("class:toolbar.response", state.response_model_tier),
        ("class:toolbar.separator", "   "),
        ("class:toolbar.label", "thread"),
        ("class:toolbar.punct", ": "),
        ("class:toolbar.thread", identity),
    ]
    if state.pending_status:
        fragments.extend(
            [
                ("class:toolbar.separator", "   "),
                ("class:toolbar.label", "status"),
                ("class:toolbar.punct", ": "),
                ("class:toolbar.pending", state.pending_status),
            ]
        )
    if state.last_action:
        fragments.extend(
            [
                ("class:toolbar.separator", "   "),
                ("class:toolbar.label", "last"),
                ("class:toolbar.punct", ": "),
                ("class:toolbar.last", _clip_toolbar_message(state.last_action)),
            ]
        )
    fragments.extend(
        [
            ("class:toolbar.separator", "   "),
            ("class:toolbar.hint", "/"),
            ("class:toolbar.label", " commands"),
        ]
    )
    return FormattedText(fragments)


async def read_user_input(state: PromptToolbarState) -> str:
    """Read one user input line with enhanced slash-command UX.

    Args:
        state (PromptToolbarState): Current prompt toolbar metadata.

    Returns:
        str: User-entered input stripped of surrounding whitespace.
    """

    if not sys.stdin.isatty():
        return (
            await asyncio.to_thread(
                Prompt.ask,
                "  [primary]·[/primary] [accent]you[/accent]",
            )
        ).strip()

    with patch_stdout():
        value = await _prompt_session().prompt_async(
            _prompt_message(),
            bottom_toolbar=lambda: prompt_toolbar(state),
            complete_style=CompleteStyle.COLUMN,
        )
    return value.strip()
