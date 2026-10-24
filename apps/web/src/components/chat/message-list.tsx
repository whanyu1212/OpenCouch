import type { Message } from "@ag-ui/core";
import { useEffect, useRef } from "react";

import { CoachReply } from "./coach-reply";

// Follow new text only while the reader is near the bottom, so scrolling back
// to reread isn't interrupted.
const FOLLOW_DISTANCE_PX = 160;
// Space kept between the end of the conversation and the composer.
const CLEARANCE_PX = 16;

function textOf(message: Message): string {
  return typeof message.content === "string" ? message.content : "";
}

export function MessageList({
  messages,
  interrupted,
  thinking,
  bottomInset,
}: {
  messages: Message[];
  /** Ids of replies that were cut off. */
  interrupted: ReadonlySet<string>;
  thinking: boolean;
  /** Height of whatever floats over the bottom of the page (the composer). */
  bottomInset: number;
}) {
  const end = useRef<HTMLDivElement>(null);
  // A reloaded conversation opens at its latest message, however long it is.
  const shownLatest = useRef(false);
  const visible = messages.filter(
    (message) => (message.role === "user" || message.role === "assistant") && textOf(message),
  );
  const last = visible.at(-1);

  useEffect(() => {
    const element = end.current;
    if (!element) return;
    if (!shownLatest.current) {
      shownLatest.current = true;
      element.scrollIntoView({ block: "end" });
      return;
    }
    // Measured against the composer's top, not the window's bottom, so a
    // composer that grows with a long draft doesn't stop the following.
    const visibleBottom = window.innerHeight - bottomInset;
    const nearBottom = element.getBoundingClientRect().bottom - visibleBottom < FOLLOW_DISTANCE_PX;
    // Always follow the user's own message; follow the reply only when near.
    // Instant, not smooth: a smooth scroll lags behind a fast reply, falls out
    // of following range, and stops following.
    if (last?.role === "user" || nearBottom) {
      element.scrollIntoView({ block: "end" });
    }
  }, [last?.id, last?.content, last?.role, thinking, bottomInset]);

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
              <CoachReply text={textOf(message)} interrupted={interrupted.has(message.id)} />
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
      <div ref={end} style={{ scrollMarginBottom: bottomInset + CLEARANCE_PX }} />
    </>
  );
}
