"use client";

import { type FormEvent, useState } from "react";

import { useCompanion } from "@/lib/use-companion";

export default function Home() {
  const { messages, running, error, send } = useCompanion();
  const [draft, setDraft] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || running) return;
    setDraft("");
    void send(text);
  }

  const visible = messages.filter(
    (m) => (m.role === "user" || m.role === "assistant") && m.content,
  );

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-4 p-4">
      <h1 className="text-xl font-semibold">OpenCouch</h1>

      <ol className="flex flex-1 flex-col gap-3">
        {visible.map((m) => (
          <li
            key={m.id}
            className={
              m.role === "user"
                ? "self-end rounded-2xl bg-foreground px-4 py-2 text-background"
                : "self-start rounded-2xl border px-4 py-2"
            }
          >
            {typeof m.content === "string" ? m.content : null}
          </li>
        ))}
      </ol>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="What's on your mind?"
          className="flex-1 rounded-full border px-4 py-2"
          aria-label="Message"
        />
        <button
          type="submit"
          disabled={running || !draft.trim()}
          className="rounded-full bg-foreground px-4 py-2 text-background disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </main>
  );
}
