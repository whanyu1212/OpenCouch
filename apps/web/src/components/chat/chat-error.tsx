import { useEffect, useRef } from "react";

import type { ChatError } from "@/lib/use-companion";

const EXPLANATIONS: Record<ChatError["kind"], string> = {
  busy: "The last reply is still being written. Send your message again in a moment.",
  not_sent: "Your message didn't go through. It's back in the box below, ready to send again.",
  unreachable:
    "OpenCouch can't be reached right now. Check your connection, then send your message again.",
  history_unavailable:
    "Your earlier messages couldn't be loaded. They're still saved; reload the page to try again.",
  timeout: "The reply took too long and stopped.",
  reply_failed: "The reply stopped partway through.",
};

export function ChatErrorNotice({
  error,
  bottomInset,
  onSendAgain,
  onDismiss,
}: {
  error: ChatError;
  /** Height of the composer floating over the bottom of the page. */
  bottomInset: number;
  onSendAgain: (text: string) => void;
  onDismiss: () => void;
}) {
  const notice = useRef<HTMLDivElement>(null);
  // The server kept these messages, so the same text can be sent again.
  const retryText = error.kind === "timeout" || error.kind === "reply_failed" ? error.text : null;

  // Bring the notice up from behind the floating composer when it appears.
  useEffect(() => {
    notice.current?.scrollIntoView({ block: "end" });
  }, [error]);

  return (
    <div
      ref={notice}
      role="alert"
      style={{ scrollMarginBottom: bottomInset + 16 }}
      className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl bg-notice-wash px-4 py-3 text-sm text-notice"
    >
      <p className="flex-1 basis-56">{EXPLANATIONS[error.kind]}</p>
      <div className="flex gap-2">
        {retryText !== null && (
          <button
            type="button"
            onClick={() => onSendAgain(retryText)}
            className="rounded-full bg-accent px-4 py-1.5 font-medium text-accent-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Send again
          </button>
        )}
        <button
          type="button"
          onClick={onDismiss}
          className="rounded-full px-3 py-1.5 font-medium underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-notice"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}
