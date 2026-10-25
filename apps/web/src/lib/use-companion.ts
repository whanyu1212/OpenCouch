"use client";

import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { agentUrl } from "./agent-url";
import { agentAuthHeaders, SignedOutError } from "./auth/client";
import { cardKeysIn } from "./crisis-catalog";
import { loadHistory } from "./history";
import { NewestMessageAgent } from "./newest-message-agent";
import { savedThreadId, startThread } from "./thread-id";

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

/** Whether the message stayed in the conversation or was taken back out. */
export type SendResult = "kept" | "removed";

/** The HTTP status `@ag-ui/client` attaches to errors from non-2xx responses. */
function httpStatus(error: unknown): number | undefined {
  if (error instanceof Error && "status" in error && typeof error.status === "number") {
    return error.status;
  }
  return undefined;
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
  // Replies cut off by an error, a timeout or a disconnect.
  const [interrupted, setInterrupted] = useState<ReadonlySet<string>>(new Set());
  // Catalog keys for the crisis resource card. Only the server sets them.
  const [crisisResources, setCrisisResources] = useState<string[]>([]);
  // Set by the stream as it arrives, read after `runAgent` settles.
  const runStarted = useRef(false);
  const runError = useRef<"timeout" | "reply_failed" | null>(null);

  useEffect(() => {
    if (!agent) return;
    const { unsubscribe } = agent.subscribe({
      onMessagesChanged: ({ messages }) => setMessages([...messages]),
      onRunStartedEvent: () => {
        runStarted.current = true;
        // The card belongs to crisis turns, which send it straight after this.
        setCrisisResources([]);
      },
      onStateSnapshotEvent: ({ event }) => setCrisisResources(cardKeysIn(event.snapshot)),
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

  // Load what was already said in this conversation; a new one has nothing.
  useEffect(() => {
    if (!agent) return;
    const controller = new AbortController();
    agentAuthHeaders()
      .then((headers) => loadHistory(agent.url, agent.threadId, headers, controller.signal))
      .then((history) => {
        // A newer conversation started while this one was loading.
        if (controller.signal.aborted) return;
        agent.setMessages(history.messages);
        setInterrupted(history.interrupted);
        setCrisisResources(history.crisisResources);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError({ kind: err instanceof SignedOutError ? "signed_out" : "history_unavailable" });
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [agent]);

  const markLastReplyInterrupted = useCallback(() => {
    const last = agent?.messages.at(-1);
    if (last?.role === "assistant") {
      setInterrupted((current) => new Set(current).add(last.id));
    }
  }, [agent]);

  const send = useCallback(
    async (text: string): Promise<SendResult> => {
      if (!agent) return "removed";
      const id = crypto.randomUUID();
      runStarted.current = false;
      runError.current = null;
      setError(null);
      setRunning(true);
      try {
        // Before the message is added, so a sign-out leaves nothing to take back out.
        agent.setAuthHeaders(await agentAuthHeaders());
        agent.addMessage({ id, role: "user", content: text });
        await agent.runAgent();
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
          setError(sendError(err));
        }
      } finally {
        setRunning(false);
      }
      return agent.messages.some((message) => message.id === id) ? "kept" : "removed";
    },
    [agent, markLastReplyInterrupted],
  );

  const startNewConversation = useCallback(() => {
    setMessages([]);
    setInterrupted(new Set());
    setCrisisResources([]);
    setError(null);
    setLoading(true);
    setThreadId(startThread());
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  return {
    messages,
    interrupted,
    crisisResources,
    loading,
    running,
    error,
    send,
    startNewConversation,
    dismissError,
  };
}
