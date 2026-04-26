import type { CommandActionId } from "@/lib/command-actions";

export type SlashCommandResolution =
  | {
      kind: "action";
      actionId: CommandActionId;
      disabledMessage?: string;
    }
  | {
      kind: "unsupported";
      message: string;
    };

const HELP_TEXT = [
  "Available web shortcuts:",
  "",
  "/help - show this list",
  "/end - end the current persistent session",
  "/new - start a new session",
  "/threads - browse previous sessions",
  "/memory - open memory",
  "/memory recall on - enable proactive recall",
  "/memory recall off - disable proactive recall",
  "/state - open graph state",
  "/chat - return to chat",
  "/response-tier fast - use faster replies",
  "/response-tier quality - use higher quality replies",
].join("\n");

export function getSlashCommandHelpText(): string {
  return HELP_TEXT;
}

export function resolveSlashCommand(
  input: string
): SlashCommandResolution | null {
  const normalized = input.trim().replace(/\s+/g, " ").toLowerCase();
  if (!normalized.startsWith("/")) return null;

  if (normalized === "/help") {
    return { kind: "action", actionId: "show_help" };
  }

  if (normalized === "/end") {
    return {
      kind: "action",
      actionId: "end_session",
      disabledMessage: "There is no active persistent session to end.",
    };
  }

  if (normalized === "/new") {
    return {
      kind: "action",
      actionId: "new_session",
      disabledMessage:
        "A response or voice session is still active. Wait for it to finish before starting a new session.",
    };
  }

  if (normalized === "/threads") {
    return { kind: "action", actionId: "open_threads" };
  }

  if (
    normalized === "/memory" ||
    normalized === "/memory status" ||
    normalized.startsWith("/memory list")
  ) {
    return { kind: "action", actionId: "open_memory" };
  }

  if (normalized === "/memory recall on") {
    return {
      kind: "action",
      actionId: "memory_recall_on",
      disabledMessage: "Proactive recall is available only in persistent mode.",
    };
  }

  if (normalized === "/memory recall off") {
    return {
      kind: "action",
      actionId: "memory_recall_off",
      disabledMessage: "Proactive recall is available only in persistent mode.",
    };
  }

  if (normalized === "/state" || normalized === "/debug state") {
    return { kind: "action", actionId: "open_state" };
  }

  if (normalized === "/context") {
    return { kind: "action", actionId: "open_state" };
  }

  if (normalized === "/chat") {
    return { kind: "action", actionId: "open_chat" };
  }

  if (normalized === "/response-tier fast") {
    return { kind: "action", actionId: "response_fast" };
  }

  if (normalized === "/response-tier quality") {
    return { kind: "action", actionId: "response_quality" };
  }

  if (
    normalized === "/reset" ||
    normalized.startsWith("/memory clear") ||
    normalized.startsWith("/memory purge-crisis")
  ) {
    return {
      kind: "unsupported",
      message:
        "That shortcut changes or deletes stored state. Use the web controls with confirmation instead.",
    };
  }

  if (normalized.startsWith("/memory forget")) {
    return {
      kind: "unsupported",
      message:
        "Open Memory to delete individual facts, sessions, or rules from the web UI.",
    };
  }

  if (normalized === "/exit") {
    return {
      kind: "unsupported",
      message: "The browser version does not use /exit. Use End Session if you want to close the current session.",
    };
  }

  return {
    kind: "unsupported",
    message: `I do not recognize that shortcut.\n\n${HELP_TEXT}`,
  };
}
