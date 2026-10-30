import type { Message } from "@ag-ui/core";

import { SignedOutError } from "./auth/client";
import { type CrisisUiState, parseCrisisUiState } from "./crisis-state";
import { isRecord } from "./is-record";
import { isNotInvited } from "./not-invited";

// Past this, the chat gives up waiting and lets the user carry on.
const HISTORY_TIMEOUT_MS = 10_000;

/** The saved conversation isn't this person's, such as one from before sign-in. */
export class ConversationNotFoundError extends Error {
  constructor() {
    super("No such conversation");
    this.name = "ConversationNotFoundError";
  }
}

/** The person is signed in but not invited to the private beta (HY-353). */
export class NotInvitedError extends Error {
  constructor() {
    super("Not invited to the beta");
    this.name = "NotInvitedError";
  }
}

/** A conversation's saved messages, which replies were cut off, and its resource card. */
export type LoadedHistory = {
  messages: Message[];
  interrupted: Set<string>;
  crisis: CrisisUiState;
};

type ShownMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  interrupted: boolean;
};

// The response is untrusted input, so it is checked field by field.
function isShownMessage(value: unknown): value is ShownMessage {
  if (!isRecord(value)) return false;
  const { id, role, content, interrupted } = value;
  return (
    typeof id === "string" &&
    (role === "user" || role === "assistant") &&
    typeof content === "string" &&
    typeof interrupted === "boolean"
  );
}

/**
 * Load a conversation's saved messages from the agent.
 *
 * @throws {SignedOutError} If the agent doesn't accept the sign-in.
 * @throws {NotInvitedError} If the person isn't invited to the beta.
 * @throws {ConversationNotFoundError} If the conversation isn't this person's.
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
  if (response.status === 403 && isNotInvited(await response.json().catch(() => null))) {
    throw new NotInvitedError();
  }
  if (response.status === 404) throw new ConversationNotFoundError();
  if (!response.ok) throw new Error(`History request failed with ${response.status}`);
  return parseHistory(await response.json());
}

/**
 * Check the agent's history response and split it into messages, cut-off
 * replies and the resource card.
 *
 * @throws If the response, or any message in it, isn't in the expected shape.
 */
export function parseHistory(body: unknown): LoadedHistory {
  const raw =
    typeof body === "object" && body !== null && "messages" in body ? body.messages : null;
  if (!Array.isArray(raw) || !raw.every(isShownMessage)) {
    throw new Error("History response was not in the expected shape");
  }
  return {
    messages: raw.map(({ id, role, content }) => ({ id, role, content })),
    interrupted: new Set(raw.filter((message) => message.interrupted).map(({ id }) => id)),
    crisis: parseCrisisUiState(body),
  };
}
