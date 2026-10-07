"use client";

import { HttpAgent } from "@ag-ui/client";
import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useState } from "react";

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8080/api/agent";

// The server rejects these before writing anything, so the message was never
// stored and must not stay in the conversation.
const NOT_STORED_MESSAGES: Partial<Record<number, string>> = {
  409: "A reply is still on its way. Your message wasn't sent; try again in a moment.",
  422: "Your message couldn't be sent.",
};

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
      onRunFailed: ({ error }) => setError(error.message),
    });
    return () => {
      unsubscribe();
      agent.abortRun();
    };
  }, [agent]);

  const send = useCallback(
    async (text: string): Promise<SendResult> => {
      setError(null);
      setRunning(true);
      const id = crypto.randomUUID();
      agent.addMessage({ id, role: "user", content: text });
      try {
        await agent.runAgent();
        return "kept";
      } catch (err) {
        const status = httpStatus(err);
        const rejection = status === undefined ? undefined : NOT_STORED_MESSAGES[status];
        if (rejection === undefined) {
          setError(errorMessage(err));
          return "kept";
        }
        agent.setMessages(agent.messages.filter((message) => message.id !== id));
        setError(rejection);
        return "removed";
      } finally {
        setRunning(false);
      }
    },
    [agent],
  );

  return { messages, running, error, send };
}
