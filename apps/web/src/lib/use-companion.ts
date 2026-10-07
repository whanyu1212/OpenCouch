"use client";

import { HttpAgent } from "@ag-ui/client";
import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useState } from "react";

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8080/api/agent";

const BUSY_MESSAGE =
  "Your message wasn't sent because the previous reply hasn't finished. Try again in a moment.";
const NOT_SENT_MESSAGE = "Your message couldn't be sent. Please try again.";

/** Whether the message stayed in the conversation or was taken back out. */
export type SendResult = "kept" | "removed";

/** The HTTP status `@ag-ui/client` attaches to errors from non-2xx responses. */
function httpStatus(error: unknown): number | undefined {
  if (error instanceof Error && "status" in error && typeof error.status === "number") {
    return error.status;
  }
  return undefined;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function useCompanion() {
  const agent = useMemo(() => new HttpAgent({ url: AGENT_URL }), []);
  const [messages, setMessages] = useState<Message[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const { unsubscribe } = agent.subscribe({
      onMessagesChanged: ({ messages }) => setMessages([...messages]),
      // A failure mid-reply arrives as an event; `runAgent` still resolves.
      onRunErrorEvent: ({ event }) => setError(event.message),
    });
    return () => {
      unsubscribe();
      agent.abortRun();
    };
  }, [agent]);

  const send = useCallback(
    async (text: string): Promise<SendResult> => {
      const id = crypto.randomUUID();
      setError(null);
      setRunning(true);
      try {
        agent.addMessage({ id, role: "user", content: text });
        await agent.runAgent();
      } catch (err) {
        const status = httpStatus(err);
        if (status === undefined) {
          setError(errorMessage(err));
        } else {
          // An HTTP error status means the server refused the turn before
          // streaming, and it stores the message only once a turn starts.
          agent.setMessages(agent.messages.filter((message) => message.id !== id));
          setError(status === 409 ? BUSY_MESSAGE : NOT_SENT_MESSAGE);
        }
      } finally {
        setRunning(false);
      }
      return agent.messages.some((message) => message.id === id) ? "kept" : "removed";
    },
    [agent],
  );

  return { messages, running, error, send };
}
