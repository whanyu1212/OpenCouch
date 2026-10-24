import { type KeyboardEvent, type SubmitEvent, useEffect, useRef } from "react";

const MAX_HEIGHT_PX = 160;
// The agent refuses longer messages (`MAX_MESSAGE_CHARS` in turn.py). The
// browser counts some characters, like emoji, as two, so this is never looser.
const MAX_MESSAGE_CHARS = 4000;

export function Composer({
  draft,
  onDraftChange,
  onSend,
  onHeightChange,
  disabled,
}: {
  draft: string;
  onDraftChange: (draft: string) => void;
  onSend: () => void;
  onHeightChange: (height: number) => void;
  disabled: boolean;
}) {
  const form = useRef<HTMLFormElement>(null);
  const field = useRef<HTMLTextAreaElement>(null);

  // The composer floats over the conversation and grows with the draft, so
  // whatever scrolls the conversation needs its real height.
  useEffect(() => {
    const element = form.current;
    if (!element) return;
    const observer = new ResizeObserver(() =>
      onHeightChange(element.getBoundingClientRect().height),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [onHeightChange]);

  // Grow with the text, up to a few lines, then scroll.
  useEffect(() => {
    const element = field.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;
    element.style.overflowY = element.scrollHeight > MAX_HEIGHT_PX ? "auto" : "hidden";
  }, [draft]);

  const canSend = !disabled && draft.trim().length > 0;

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    onSend();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends and Shift+Enter starts a new line. Not while an input method
    // is composing (Chinese, Japanese...), where Enter picks a character.
    // While a reply is coming, Enter does nothing, so the text stays put.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (canSend) onSend();
    }
  }

  return (
    <form
      ref={form}
      onSubmit={handleSubmit}
      className="sticky bottom-0 z-10 bg-linear-to-t from-canvas from-40% to-transparent px-3 pt-6 pb-[max(1rem,env(safe-area-inset-bottom))]"
    >
      <div className="glass-pill mx-auto flex max-w-2xl items-end gap-2 rounded-3xl py-2 pr-2 pl-5 focus-within:border-accent">
        <label htmlFor="message" className="sr-only">
          Message
        </label>
        <textarea
          id="message"
          ref={field}
          rows={1}
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Write what's on your mind"
          maxLength={MAX_MESSAGE_CHARS}
          className="min-h-10.5 flex-1 resize-none bg-transparent py-2 text-ink placeholder:text-ink-soft focus:outline-none"
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send"
          className="grid size-10.5 shrink-0 place-items-center rounded-full bg-accent text-accent-ink transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-35"
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="size-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 19V5M5 12l7-7 7 7" />
          </svg>
        </button>
      </div>
    </form>
  );
}
