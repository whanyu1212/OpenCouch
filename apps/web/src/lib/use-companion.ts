"use client";

import { HttpAgent } from "@ag-ui/client";
import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8080/api/agent";

/**
 * Why the last send failed.
 *
 * `busy`, `not_sent` and `unreachable` mean the server never took the message,
 * so it is removed from the conversation and handed back to the composer.
 * `timeout` and `reply_failed` mean the server took it and the reply broke
 * off, so the message stays and the same text can be sent again.
 */
export type ChatError =
  | { kind: "busy" }
  | { kind: "not_sent" }
  | { kind: "unreachable" }
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

export function useCompanion() {
  const agent = useMemo(() => new HttpAgent({ url: AGENT_URL }), []);
  const [messages, setMessages] = useState<Message[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<ChatError | null>(null);
  // Set by the stream as it arrives, read after `runAgent` settles.
  const runStarted = useRef(false);
  const runError = useRef<"timeout" | "reply_failed" | null>(null);

  useEffect(() => {
    const { unsubscribe } = agent.subscribe({
      onMessagesChanged: ({ messages }) => setMessages([...messages]),
      onRunStartedEvent: () => {
        runStarted.current = true;
      },
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

  const send = useCallback(
    async (text: string): Promise<SendResult> => {
      const id = crypto.randomUUID();
      runStarted.current = false;
      runError.current = null;
      setError(null);
      setRunning(true);
      try {
        agent.addMessage({ id, role: "user", content: text });
        await agent.runAgent();
        if (runError.current) {
          setError({ kind: runError.current, text });
        }
      } catch (err) {
        if (runStarted.current) {
          // The connection broke after the server took the message.
          setError({ kind: "reply_failed", text });
        } else {
          // Nothing reached the server, or it refused the turn before
          // streaming; it stores the message only once a turn starts.
          agent.setMessages(agent.messages.filter((message) => message.id !== id));
          const status = httpStatus(err);
          if (status === undefined) {
            setError({ kind: "unreachable" });
          } else {
            setError({ kind: status === 409 ? "busy" : "not_sent" });
          }
        }
      } finally {
        setRunning(false);
      }
      return agent.messages.some((message) => message.id === id) ? "kept" : "removed";
    },
    [agent],
  );

  const dismissError = useCallback(() => setError(null), []);

  return { messages, running, error, send, dismissError };
}
