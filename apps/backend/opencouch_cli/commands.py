"""Slash-command metadata shared by help rendering and completions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """Metadata for one slash command or completion-only command path."""

    path: tuple[str, ...]
    description: str
    category: str
    usage: str | None = None
    show_in_help: bool = True
    example: str | None = None

    @property
    def display(self) -> str:
        """Return the command usage shown in menus and help.

        Returns:
            str: Usage text when supplied, otherwise the literal command path.
        """

        return self.usage or " ".join(self.path)


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(("/help",), "Show available commands.", "session", example="/help"),
    SlashCommand(
        ("/status",),
        "Show current mode and session stats.",
        "session",
        example="/status",
    ),
    SlashCommand(
        ("/doctor",),
        "Check runtime readiness for the current CLI session.",
        "session",
        example="/doctor",
    ),
    SlashCommand(
        ("/history",),
        "Show the last n transcript messages. Default: 6.",
        "session",
        usage="/history [n]",
        example="/history 10",
    ),
    SlashCommand(
        ("/context",),
        "Show the latest derived session context snapshot.",
        "session",
    ),
    SlashCommand(
        ("/summary",),
        "Generate a recap of the current session.",
        "session",
        usage="/summary [short|full]",
        example="/summary short",
    ),
    SlashCommand(
        ("/summary", "short"),
        "Generate a brief recap of the current session.",
        "session",
        show_in_help=False,
    ),
    SlashCommand(
        ("/summary", "full"),
        "Generate a detailed recap of the current session.",
        "session",
        show_in_help=False,
    ),
    SlashCommand(
        ("/export",),
        "Export the current session transcript to a file.",
        "session",
        usage="/export <md|json|txt> [filename]",
        example="/export md",
    ),
    SlashCommand(
        ("/export", "md"),
        "Export the current session transcript as Markdown.",
        "session",
        show_in_help=False,
    ),
    SlashCommand(
        ("/export", "json"),
        "Export the current session transcript as JSON.",
        "session",
        show_in_help=False,
    ),
    SlashCommand(
        ("/export", "txt"),
        "Export the current session transcript as plain text.",
        "session",
        show_in_help=False,
    ),
    SlashCommand(("/keys",), "Show keyboard shortcuts and prompt tips.", "display"),
    SlashCommand(
        ("/ui",),
        "Switch prompt toolbar density.",
        "display",
        usage="/ui <compact|full>",
        example="/ui compact",
    ),
    SlashCommand(
        ("/ui", "compact"),
        "Use compact toolbar layout.",
        "display",
        show_in_help=False,
    ),
    SlashCommand(
        ("/ui", "full"),
        "Use full toolbar layout.",
        "display",
        show_in_help=False,
    ),
    SlashCommand(
        ("/theme",),
        "Switch prompt theme preset.",
        "display",
        usage="/theme <mono|contrast|calm>",
        example="/theme calm",
    ),
    SlashCommand(
        ("/theme", "mono"),
        "Use the neutral mono theme.",
        "display",
        show_in_help=False,
    ),
    SlashCommand(
        ("/theme", "contrast"),
        "Use the high-contrast theme.",
        "display",
        show_in_help=False,
    ),
    SlashCommand(
        ("/theme", "calm"),
        "Use the calmer sage theme.",
        "display",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory",),
        "Inspect or manage memory.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "status"),
        "Show memory layer state (counts, mode, crisis log, recall toggle).",
        "memory",
        example="/memory status",
    ),
    SlashCommand(
        ("/memory", "list"),
        "List semantic facts, episodic arcs, and procedural rules.",
        "memory",
        usage="/memory list [facts|sessions|rules]",
        example="/memory list facts",
    ),
    SlashCommand(
        ("/memory", "list", "facts"),
        "List semantic facts.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "list", "sessions"),
        "List episodic session arcs.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "list", "rules"),
        "List procedural response-style rules.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "recall"),
        "Toggle whether the agent proactively references past memory content.",
        "memory",
        usage="/memory recall on|off",
    ),
    SlashCommand(
        ("/memory", "recall", "on"),
        "Enable proactive memory references.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "recall", "off"),
        "Disable proactive memory references.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "forget"),
        "Delete one owner-scoped record by its displayed index.",
        "memory",
        usage="/memory forget <fact|session|rule> <n>",
        example="/memory forget fact 3",
    ),
    SlashCommand(
        ("/memory", "forget", "fact"),
        "Forget one semantic fact by index.",
        "memory",
        usage="/memory forget fact <n>",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "forget", "session"),
        "Forget one episodic session by index.",
        "memory",
        usage="/memory forget session <n>",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "forget", "rule"),
        "Forget one procedural rule by index.",
        "memory",
        usage="/memory forget rule <n>",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "clear"),
        "Wipe a memory namespace for the active user after typed confirmation.",
        "memory",
        usage="/memory clear <facts|sessions|rules|all>",
        example="/memory clear facts",
    ),
    SlashCommand(
        ("/memory", "clear", "facts"),
        "Clear all semantic facts.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "clear", "sessions"),
        "Clear all episodic session arcs.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "clear", "rules"),
        "Clear all procedural rules.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "clear", "all"),
        "Clear semantic, episodic, and procedural memory.",
        "memory",
        show_in_help=False,
    ),
    SlashCommand(
        ("/memory", "purge-crisis"),
        "Delete crisis log records older than the retention window.",
        "memory",
        usage="/memory purge-crisis [days]",
    ),
    SlashCommand(
        ("/threads",),
        "List persisted thread ids. Default: 12.",
        "threads",
        usage="/threads [n]",
    ),
    SlashCommand(
        ("/resume",),
        "Switch to an existing persisted thread.",
        "threads",
        usage="/resume <thread-id>",
    ),
    SlashCommand(
        ("/new",),
        "Start a fresh thread without restarting the CLI.",
        "threads",
        usage="/new [thread-id]",
    ),
    SlashCommand(("/reset",), "Clear the conversation history.", "session"),
    SlashCommand(("/clear",), "Clear the terminal and redraw the header.", "session"),
    SlashCommand(
        ("/mode",),
        "Switch LLM resolution mode for future turns.",
        "runtime",
        usage="/mode <deterministic|hybrid|auto>",
        example="/mode hybrid",
    ),
    SlashCommand(
        ("/mode", "deterministic"),
        "Use deterministic local behavior.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/mode", "hybrid"),
        "Use configured control model calls.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/mode", "auto"),
        "Use configured models when available, otherwise deterministic.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/response-tier",),
        "Switch the therapeutic response quality/latency tradeoff.",
        "runtime",
        usage="/response-tier <fast|quality>",
        example="/response-tier quality",
    ),
    SlashCommand(
        ("/response-tier", "fast"),
        "Favor lower response latency.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/response-tier", "quality"),
        "Favor richer therapeutic prose.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/verbosity",),
        "Switch turn observability detail.",
        "runtime",
        usage="/verbosity <compact|verbose>",
        example="/verbosity verbose",
    ),
    SlashCommand(
        ("/verbosity", "compact"),
        "Show compact route and tool badges after each turn.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/verbosity", "verbose"),
        "Show fuller route, tool, and state activity after each turn.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/trace",),
        "Show or hide the routing trace overlay.",
        "runtime",
        usage="/trace on|off|once",
    ),
    SlashCommand(
        ("/trace", "on"),
        "Show the routing trace after every turn.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/trace", "off"),
        "Hide the routing trace.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/trace", "once"),
        "Show the routing trace for the next turn only.",
        "runtime",
        show_in_help=False,
    ),
    SlashCommand(
        ("/debug",),
        "Inspect raw diagnostic state.",
        "debug",
        show_in_help=False,
    ),
    SlashCommand(
        ("/debug", "state"),
        "Dump the raw persisted state for the active thread.",
        "debug",
        usage="/debug state",
    ),
    SlashCommand(
        ("/end",),
        "End the session; summarize it and save the arc to episodic memory.",
        "session",
        usage="/end [new [thread-id]]",
    ),
    SlashCommand(
        ("/end", "new"),
        "Save the current session and continue in a fresh thread.",
        "session",
        usage="/end new [thread-id]",
        show_in_help=False,
    ),
    SlashCommand(
        ("/exit",),
        "End the session; prompt to save a summary before closing.",
        "session",
    ),
    SlashCommand(
        ("/quit",),
        "Alias for /exit.",
        "session",
        show_in_help=False,
    ),
)


def help_commands() -> tuple[SlashCommand, ...]:
    """Return commands that should appear in `/help`.

    Returns:
        tuple[SlashCommand, ...]: Help-visible slash commands.
    """

    return tuple(command for command in SLASH_COMMANDS if command.show_in_help)


def child_commands(parent: tuple[str, ...]) -> tuple[SlashCommand, ...]:
    """Return direct child command paths under a parent path.

    Args:
        parent (tuple[str, ...]): Already typed command path tokens.

    Returns:
        tuple[SlashCommand, ...]: Direct child command metadata.
    """

    depth = len(parent)
    seen: set[tuple[str, ...]] = set()
    children: list[SlashCommand] = []
    for command in SLASH_COMMANDS:
        if len(command.path) <= depth or command.path[:depth] != parent:
            continue
        child_path = command.path[: depth + 1]
        if child_path in seen:
            continue
        seen.add(child_path)
        children.append(command)
    return tuple(children)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

ALIASES: dict[str, str] = {
    "/h": "/help",
    "/s": "/status",
    "/m": "/memory",
    "/k": "/keys",
    "/t": "/theme",
    "/q": "/quit",
    "/c": "/clear",
}


def resolve_alias(command: str) -> str:
    """Resolve a command alias to its canonical form.

    Args:
        command (str): Raw command token (e.g. "/h").

    Returns:
        str: Canonical command (e.g. "/help"), or the original if not an alias.
    """

    return ALIASES.get(command, command)


def all_command_names() -> list[str]:
    """Return all unique top-level command names (for fuzzy matching).

    Returns:
        list[str]: Sorted list of top-level command names like ["/clear", "/help", ...].
    """

    names: set[str] = set()
    for cmd in SLASH_COMMANDS:
        names.add(cmd.path[0])
    return sorted(names)


def format_commands_for_llm() -> str:
    """Return a compact help-visible command list for system-prompt injection.

    Returns:
        str: One command per line using exact user-facing syntax.
    """

    lines = [
        f"- {command.display} — {command.description}" for command in help_commands()
    ]
    return "\n".join(lines)
