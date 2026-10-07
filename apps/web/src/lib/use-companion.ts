"use client";

import { HttpAgent } from "@ag-ui/client";
import type { Message } from "@ag-ui/core";
import { useCallback, useEffect, useMemo, useState } from "react";

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8080/api/agent";

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
    async (text: string) => {
      setError(null);
      setRunning(true);
      agent.addMessage({ id: crypto.randomUUID(), role: "user", content: text });
      try {
        await agent.runAgent();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setRunning(false);
      }
    },
    [agent],
  );

  return { messages, running, error, send };
}
