"use client";

import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { agentUrl } from "./agent-url";
import { agentAuthHeaders, SignedOutError } from "./auth/client";
import {
  type CrisisCheckInAnswer,
  crisisCheckInRunParameters,
  crisisCheckInText,
} from "./crisis-check-in";
import { crisisResourcesAt, EMPTY_CRISIS_UI_STATE, parseCrisisUiState } from "./crisis-state";
import { ConversationNotFoundError, loadHistory, NotInvitedError } from "./history";
import { isNotInvited } from "./not-invited";
import { NewestMessageAgent } from "./newest-message-agent";
import { onThreadChangedElsewhere, savedThreadId, startThread } from "./thread-id";

/**
 * Why something failed.
 *
 * `busy`, `not_sent` and `unreachable` mean the server never took the message,
 * so it is removed from the conversation and handed back to the composer.
 * `timeout` and `reply_failed` mean the server took it and the reply broke
 * off, so the message stays and the same text can be sent again.
 * `history_unavailable` means the saved conversation couldn't be loaded.
 * `signed_out` means the sign-in ended, so nothing more can be sent or loaded.
 */
export type ChatError =
  | { kind: "signed_out" }
  | { kind: "busy" }
  | { kind: "not_sent" }
  | { kind: "unreachable" }
  | { kind: "history_unavailable" }
  | { kind: "timeout"; text: string }
  | { kind: "reply_failed"; text: string };

/** Whether the message stayed, was taken back out, or lost a same-tick send race. */
export type SendResult = "kept" | "removed" | "ignored";

/** Whether `@ag-ui/client` refused a send because the agent says they aren't invited. */
function refusedAsNotInvited(error: unknown): boolean {
  return (
    httpStatus(error) === 403 &&
    error instanceof Error &&
    "payload" in error &&
    isNotInvited(error.payload)
  );
}

/** The HTTP status `@ag-ui/client` attaches to errors from non-2xx responses. */
function httpStatus(error: unknown): number | undefined {
  if (error instanceof Error && "status" in error && typeof error.status === "number") {
    return error.status;
  }
  return undefined;
}

// Why the saved conversation couldn't be shown.
function historyError(error: unknown): ChatError {
  return { kind: error instanceof SignedOutError ? "signed_out" : "history_unavailable" };
}

// Why a message the server never took wasn't sent.
function sendError(error: unknown): ChatError {
  const status = httpStatus(error);
  if (error instanceof SignedOutError || status === 401) return { kind: "signed_out" };
  if (status === undefined) return { kind: "unreachable" };
  return { kind: status === 409 ? "busy" : "not_sent" };
}

export function useCompanion() {
  // The conversation this browser is in. Local storage only exists in the
  // browser, so there is none while rendering on the server. Reading (and on a
  // first visit, writing) it here is safe: a second call reads what the first
  // wrote, and nothing rendered depends on the id.
  const [threadId, setThreadId] = useState<string | null>(() =>
    typeof window === "undefined" ? null : savedThreadId(),
  );
  // One agent per conversation: a new conversation gets a fresh one.
  const agent = useMemo(
    () => (threadId === null ? null : new NewestMessageAgent({ url: agentUrl(), threadId })),
    [threadId],
  );
  const [messages, setMessages] = useState<Message[]>([]);
  // True until the saved conversation has loaded, so the empty state doesn't
  // flash before it.
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<ChatError | null>(null);
  // Signed in but not invited to the private beta: the chat gives way to the waitlist.
  const [notInvited, setNotInvited] = useState(false);
  // Replies cut off by an error, a timeout or a disconnect.
  const [interrupted, setInterrupted] = useState<ReadonlySet<string>>(new Set());
  // Resource visibility and check-in offers are all server-owned.
  const [crisis, setCrisis] = useState(EMPTY_CRISIS_UI_STATE);
  const [resourceClock, setResourceClock] = useState(() => Date.now());
  // State disables the UI on the next render; this closes the smaller window
  // before that render where a second click could start another request.
  const sending = useRef(false);
  // Set by the stream as it arrives, read after `runAgent` settles.
  const runStarted = useRef(false);
  const runError = useRef<"timeout" | "reply_failed" | null>(null);

  useEffect(() => {
    if (!agent) return;
    const { unsubscribe } = agent.subscribe({
      onMessagesChanged: ({ messages }) => setMessages([...messages]),
      onRunStartedEvent: () => {
        runStarted.current = true;
        // A tap can be submitted once. Keep resources pinned until the next
        // snapshot replaces them, including if this run later disconnects.
        setCrisis((current) => ({ ...current, crisisCheckIn: null }));
      },
      onStateSnapshotEvent: ({ event }) => setCrisis(parseCrisisUiState(event.snapshot)),
      // A failure mid-reply arrives as an event; `runAgent` still resolves.
      // Its message is server text, so only its code is used.
      onRunErrorEvent: ({ event }) => {
        runError.current = event.code === "turn_timeout" ? "timeout" : "reply_failed";
      },
    });
    return () => {
      unsubscribe();
      agent.abortRun();
    };
  }, [agent]);

  // Expire a pinned card without waiting for another turn or a page reload.
  useEffect(() => {
    const expiry = crisis.crisisResourcesUntil;
    if (expiry === null) return;
    let timer: number;
    const expire = () => {
      const now = Date.now();
      setResourceClock(now);
      if (now < expiry) timer = window.setTimeout(expire, expiry - now);
    };
    timer = window.setTimeout(expire, Math.max(0, expiry - Date.now()));
    return () => window.clearTimeout(timer);
  }, [crisis.crisisResourcesUntil]);

  // Someone signed out, or in as someone else, in another tab. Reloading
  // checks the session again, so this tab never sends one person's messages
  // into another's conversation.
  useEffect(() => onThreadChangedElsewhere(() => window.location.reload()), []);

  // A fresh conversation on screen and in storage; any error stays showing.
  const openNewThread = useCallback(() => {
    setMessages([]);
    setInterrupted(new Set());
    setCrisis(EMPTY_CRISIS_UI_STATE);
    setLoading(true);
    setThreadId(startThread());
  }, []);

  // Load what was already said in this conversation; a new one has nothing.
  useEffect(() => {
    if (!agent) return;
    const controller = new AbortController();
    agentAuthHeaders(controller.signal)
      .then((headers) => loadHistory(agent.url, agent.threadId, headers, controller.signal))
      .then((history) => {
        // A newer conversation started while this one was loading.
        if (controller.signal.aborted) return;
        agent.setMessages(history.messages);
        setInterrupted(history.interrupted);
        setCrisis(history.crisis);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        if (err instanceof ConversationNotFoundError) {
          // Nothing of theirs to show, so they start a conversation of their
          // own. It stays loading until that one's history is in, so nothing
          // sent meanwhile can be wiped by it.
          openNewThread();
          return;
        }
        if (err instanceof NotInvitedError) {
          setNotInvited(true);
        } else {
          setError(historyError(err));
        }
        setLoading(false);
      });
    return () => controller.abort();
  }, [agent, openNewThread]);

  const markLastReplyInterrupted = useCallback(() => {
    const last = agent?.messages.at(-1);
    if (last?.role === "assistant") {
      setInterrupted((current) => new Set(current).add(last.id));
    }
  }, [agent]);

  const sendMessage = useCallback(
    async (
      text: string,
      checkIn?: { offerId: string; answer: CrisisCheckInAnswer },
    ): Promise<SendResult> => {
      if (!agent) return "removed";
      if (sending.current) return "ignored";
      sending.current = true;
      const id = crypto.randomUUID();
      runStarted.current = false;
      runError.current = null;
      setError(null);
      setRunning(true);
      try {
        // Before the message is added, so a sign-out leaves nothing to take back out.
        agent.setAuthHeaders(await agentAuthHeaders());
        agent.addMessage({ id, role: "user", content: text });
        await agent.runAgent(
          checkIn ? crisisCheckInRunParameters(checkIn.offerId, checkIn.answer) : undefined,
        );
        if (runError.current) {
          markLastReplyInterrupted();
          setError({ kind: runError.current, text });
        }
      } catch (err) {
        if (runStarted.current) {
          // The connection broke after the server took the message.
          markLastReplyInterrupted();
          setError({ kind: "reply_failed", text });
        } else {
          // Nothing reached the server, or it refused the turn before
          // streaming; it stores the message only once a turn starts.
          agent.setMessages(agent.messages.filter((message) => message.id !== id));
          if (refusedAsNotInvited(err)) {
            setNotInvited(true);
          } else {
            setError(sendError(err));
          }
          // The conversation isn't theirs (say, saved before threads had
          // owners), so they go on in a new one; the message is handed back.
          if (httpStatus(err) === 404) openNewThread();
        }
      } finally {
        sending.current = false;
        setRunning(false);
      }
      return agent.messages.some((message) => message.id === id) ? "kept" : "removed";
    },
    [agent, markLastReplyInterrupted, openNewThread],
  );

  const send = useCallback((text: string) => sendMessage(text), [sendMessage]);

  const sendCrisisCheckIn = useCallback(
    (answer: CrisisCheckInAnswer): Promise<SendResult> => {
      if (crisis.crisisCheckIn === null) return Promise.resolve("removed");
      return sendMessage(crisisCheckInText(answer), {
        offerId: crisis.crisisCheckIn,
        answer,
      });
    },
    [crisis.crisisCheckIn, sendMessage],
  );

  const startNewConversation = useCallback(() => {
    setError(null);
    openNewThread();
  }, [openNewThread]);

  const dismissError = useCallback(() => setError(null), []);

  return {
    messages,
    interrupted,
    crisisResources: crisisResourcesAt(crisis, resourceClock),
    getHelpResources: crisis.getHelpResources,
    crisisCheckIn: crisis.crisisCheckIn,
    loading,
    running,
    error,
    notInvited,
    send,
    sendCrisisCheckIn,
    startNewConversation,
    dismissError,
  };
}
