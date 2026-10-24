"use client";

import { useState } from "react";

import { useCompanion } from "@/lib/use-companion";

import { ChatErrorNotice } from "./chat-error";
import { Composer } from "./composer";
import { MessageList } from "./message-list";

export function Chat() {
  const {
    messages,
    interrupted,
    loading,
    running,
    error,
    send,
    startNewConversation,
    dismissError,
  } = useCompanion();
  const [draft, setDraft] = useState("");
  const [composerHeight, setComposerHeight] = useState(0);

  function sendText(text: string) {
    if (running) return;
    void send(text).then((result) => {
      // Give the text back so it can be sent again, ahead of anything typed since.
      if (result === "removed") {
        setDraft((current) => (current.trim() ? `${text} ${current}` : text));
      }
    });
  }

  function sendDraft() {
    const text = draft.trim();
    // Checked before clearing, so a message isn't lost while a reply runs.
    if (!text || running) return;
    setDraft("");
    sendText(text);
  }

  const empty = messages.length === 0;

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="glass-bar sticky top-0 z-10 border-b border-line px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
        <div className="mx-auto flex max-w-2xl items-center justify-between gap-4">
          <p className="text-xl font-semibold tracking-tight">opencouch</p>
          {!empty && (
            <button
              type="button"
              onClick={startNewConversation}
              disabled={running}
              className="rounded-full border border-line px-3.5 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
            >
              New conversation
            </button>
          )}
        </div>
      </header>

      <main className="flex flex-1 flex-col px-5 pt-6 pb-8">
        <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col">
          {loading ? null : empty ? (
            <div className="mt-24 max-w-lg">
              <h1 className="text-3xl font-semibold tracking-tight text-balance">
                What&apos;s on your mind?
              </h1>
              <p className="mt-3 text-ink-soft">
                Write it here in your own words. The coach will help you work through it, one
                small step at a time.
              </p>
            </div>
          ) : (
            <>
              <h1 className="sr-only">Conversation</h1>
              <MessageList
                messages={messages}
                interrupted={interrupted}
                thinking={running}
                bottomInset={composerHeight}
              />
            </>
          )}

          {error && (
            <div className="mt-6">
              <ChatErrorNotice
                error={error}
                bottomInset={composerHeight}
                onSendAgain={sendText}
                onDismiss={dismissError}
              />
            </div>
          )}
        </div>
      </main>

      <Composer
        draft={draft}
        onDraftChange={setDraft}
        onSend={sendDraft}
        onHeightChange={setComposerHeight}
        disabled={running || loading}
      />
    </div>
  );
}
