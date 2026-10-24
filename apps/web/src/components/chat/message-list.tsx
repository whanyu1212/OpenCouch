import type { Message } from "@ag-ui/core";
import { useEffect, useRef } from "react";

import { CoachReply } from "./coach-reply";

// Follow new text only while the reader is near the bottom, so scrolling back
// to reread isn't interrupted.
const FOLLOW_DISTANCE_PX = 160;

function textOf(message: Message): string {
  return typeof message.content === "string" ? message.content : "";
}

export function MessageList({ messages, thinking }: { messages: Message[]; thinking: boolean }) {
  const end = useRef<HTMLDivElement>(null);
  const visible = messages.filter(
    (message) => (message.role === "user" || message.role === "assistant") && textOf(message),
  );
  const last = visible.at(-1);

  useEffect(() => {
    const page = document.documentElement;
    const nearBottom =
      page.scrollHeight - (window.scrollY + window.innerHeight) < FOLLOW_DISTANCE_PX;
    // Always follow the user's own message; follow the reply only when near.
    // Instant, not smooth: a smooth scroll lags behind a fast reply, falls out
    // of following range, and stops following.
    if (last?.role === "user" || nearBottom) {
      end.current?.scrollIntoView({ block: "end" });
    }
  }, [last?.id, last?.content, last?.role, thinking]);

  return (
    <>
      {/* Busy while a reply streams, so screen readers read it once it's whole. */}
      <ol role="log" aria-label="Conversation" aria-busy={thinking} className="flex flex-col gap-6">
        {visible.map((message) =>
          message.role === "user" ? (
            <li
              key={message.id}
              className="max-w-5/6 self-end whitespace-pre-wrap rounded-3xl bg-bubble px-4 py-2.5 sm:max-w-md"
            >
              {textOf(message)}
            </li>
          ) : (
            <li key={message.id}>
              <CoachReply text={textOf(message)} />
            </li>
          ),
        )}
      </ol>
      {thinking && last?.role === "user" && (
        <p role="status" className="mt-6 flex items-center gap-2.5 text-sm text-ink-soft">
          <span
            aria-hidden="true"
            className="thinking-dot size-2.5 rounded-full bg-accent ring-4 ring-accent-wash"
          />
          Thinking about your message…
        </p>
      )}
      {/* Stops above the floating composer, not behind it. */}
      <div ref={end} className="scroll-mb-28" />
    </>
  );
}
