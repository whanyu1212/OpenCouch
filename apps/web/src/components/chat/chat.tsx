"use client";

import { type RefObject, useEffect, useRef, useState } from "react";

import { useCrisisCatalog } from "@/lib/crisis-catalog";
import type { CrisisCheckInAnswer } from "@/lib/crisis-check-in";
import { useCompanion } from "@/lib/use-companion";

import { ChatErrorNotice } from "./chat-error";
import { Composer } from "./composer";
import { CrisisCard } from "./crisis-card";
import { CrisisCheckIn } from "./crisis-check-in";
import { MessageList } from "./message-list";
import { SignOutButton } from "./sign-out-button";
import { Waitlist } from "./waitlist";

function ChatHeader({
  empty,
  running,
  hasHelp,
  helpExpanded,
  onToggleHelp,
  onNewConversation,
}: {
  empty: boolean;
  running: boolean;
  hasHelp: boolean;
  helpExpanded: boolean;
  onToggleHelp: () => void;
  onNewConversation: () => void;
}) {
  return (
    <header className="glass-bar sticky top-0 z-10 border-b border-line px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
      <div className="mx-auto flex max-w-2xl flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <p className="text-xl font-semibold tracking-tight">opencouch</p>
        <div className="ml-auto flex items-center gap-1">
          {hasHelp && (
            <button
              type="button"
              onClick={onToggleHelp}
              aria-expanded={helpExpanded}
              aria-controls="crisis-resource-card"
              className="rounded-full border border-line px-3.5 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              Get help
            </button>
          )}
          {!empty && (
            <button
              type="button"
              onClick={onNewConversation}
              disabled={running}
              className="rounded-full border border-line px-3.5 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
            >
              New conversation
            </button>
          )}
          <SignOutButton disabled={running} />
        </div>
      </div>
    </header>
  );
}

function CrisisFooter({
  crisisResources,
  getHelpResources,
  crisisCheckIn,
  showHelp,
  disabled,
  catalog,
  cardRef,
  onAnswer,
}: {
  crisisResources: readonly string[];
  getHelpResources: readonly string[];
  crisisCheckIn: string | null;
  showHelp: boolean;
  disabled: boolean;
  catalog: ReturnType<typeof useCrisisCatalog>;
  cardRef: RefObject<HTMLDivElement | null>;
  onAnswer: (answer: CrisisCheckInAnswer) => void;
}) {
  const activeResources = crisisResources.length > 0;
  const showResourceCard = activeResources || (showHelp && getHelpResources.length > 0);
  if (crisisCheckIn === null && !showResourceCard) return null;
  return (
    <>
      {crisisCheckIn !== null && <CrisisCheckIn disabled={disabled} onAnswer={onAnswer} />}
      {showResourceCard && (
        <div
          id="crisis-resource-card"
          ref={cardRef}
          tabIndex={-1}
          className="rounded-3xl focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <CrisisCard
            keys={activeResources ? crisisResources : getHelpResources}
            catalog={catalog}
          />
        </div>
      )}
    </>
  );
}

function hasCrisisFooter(
  crisisResources: readonly string[],
  getHelpResources: readonly string[],
  crisisCheckIn: string | null,
  showHelp: boolean,
): boolean {
  return (
    crisisCheckIn !== null ||
    crisisResources.length > 0 ||
    (showHelp && getHelpResources.length > 0)
  );
}

export function Chat() {
  const {
    messages,
    interrupted,
    crisisResources,
    getHelpResources,
    crisisCheckIn,
    loading,
    running,
    error,
    notInvited,
    send,
    sendCrisisCheckIn,
    startNewConversation,
    dismissError,
  } = useCompanion();
  const catalog = useCrisisCatalog(crisisResources.length + getHelpResources.length > 0);
  const [draft, setDraft] = useState("");
  const [composerHeight, setComposerHeight] = useState(0);
  const [showHelp, setShowHelp] = useState(false);
  const helpCard = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (showHelp) helpCard.current?.focus();
  }, [showHelp]);

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

  function answerCheckIn(answer: CrisisCheckInAnswer) {
    void sendCrisisCheckIn(answer);
  }

  function startConversation() {
    setShowHelp(false);
    startNewConversation();
  }

  const empty = messages.length === 0;
  const showCrisisFooter = hasCrisisFooter(
    crisisResources,
    getHelpResources,
    crisisCheckIn,
    showHelp,
  );

  if (notInvited) return <Waitlist />;

  return (
    <div className="flex min-h-dvh flex-col">
      <ChatHeader
        empty={empty}
        running={running}
        hasHelp={getHelpResources.length > 0}
        helpExpanded={showHelp || crisisResources.length > 0}
        onToggleHelp={() => setShowHelp((shown) => !shown)}
        onNewConversation={startConversation}
      />

      <main className="flex flex-1 flex-col px-5 pt-6 pb-8">
        <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col">
          {loading ? null : empty ? (
            <div className="mt-24 max-w-lg">
              <h1 className="text-3xl font-semibold tracking-tight text-balance">
                What&apos;s on your mind?
              </h1>
              <p className="mt-3 text-ink-soft">
                Write it here in your own words. The coach will help you work through it, one small
                step at a time.
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
                footer={
                  showCrisisFooter ? (
                    <CrisisFooter
                      crisisResources={crisisResources}
                      getHelpResources={getHelpResources}
                      crisisCheckIn={crisisCheckIn}
                      showHelp={showHelp}
                      disabled={running || loading}
                      catalog={catalog}
                      cardRef={helpCard}
                      onAnswer={answerCheckIn}
                    />
                  ) : null
                }
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
