import type { Message } from "@ag-ui/core";

import { SignedOutError } from "./auth/client";
import { cardKeysIn } from "./crisis-catalog";

// Past this, the chat gives up waiting and lets the user carry on.
const HISTORY_TIMEOUT_MS = 10_000;

/** A conversation's saved messages, which replies were cut off, and its resource card. */
export type LoadedHistory = {
  messages: Message[];
  interrupted: Set<string>;
  crisisResources: string[];
};

type ShownMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  interrupted: boolean;
};

// The response is untrusted input, so it is checked field by field.
function isShownMessage(value: unknown): value is ShownMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    typeof value.id === "string" &&
    "role" in value &&
    (value.role === "user" || value.role === "assistant") &&
    "content" in value &&
    typeof value.content === "string" &&
    "interrupted" in value &&
    typeof value.interrupted === "boolean"
  );
}

/**
 * Load a conversation's saved messages from the agent.
 *
 * @throws {SignedOutError} If the agent doesn't accept the sign-in.
 * @throws If the agent can't be reached in time, or answers with something unexpected.
 */
export async function loadHistory(
  agentUrl: string,
  threadId: string,
  headers: Record<string, string>,
  signal: AbortSignal,
): Promise<LoadedHistory> {
  // Resolved next to the agent endpoint (`…/api/agent` → `…/api/threads/…`), so
  // a path prefix in front of `/api` is kept. Assumes no trailing slash.
  const url = new URL(`threads/${encodeURIComponent(threadId)}/messages`, agentUrl);
  const response = await fetch(url, {
    headers,
    signal: AbortSignal.any([signal, AbortSignal.timeout(HISTORY_TIMEOUT_MS)]),
  });
  if (response.status === 401) throw new SignedOutError();
  if (!response.ok) throw new Error(`History request failed with ${response.status}`);
  const body: unknown = await response.json();
  const raw =
    typeof body === "object" && body !== null && "messages" in body ? body.messages : null;
  if (!Array.isArray(raw) || !raw.every(isShownMessage)) {
    throw new Error("History response was not in the expected shape");
  }
  return {
    messages: raw.map(({ id, role, content }) => ({ id, role, content })),
    interrupted: new Set(raw.filter((message) => message.interrupted).map(({ id }) => id)),
    crisisResources: cardKeysIn(body),
  };
}
