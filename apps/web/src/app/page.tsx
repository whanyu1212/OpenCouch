"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  getHistory,
  getMemoryStatus,
  getMemoryFacts,
  getThreads,
  type Message,
  type MemoryStatus,
  type ThreadSummary,
} from "@/lib/api";
import {
  startTextChatStream,
  useSessionStore,
  type ChatMessage,
  type EndedSessionResult,
} from "@/lib/session";
import { useCommandActions } from "@/lib/command-actions";
import { resolveSlashCommand } from "@/lib/slash-commands";
import { CouchLogo } from "@/components/logo";
import { MemoryPanel, MemoryToggleButton } from "@/components/memory-panel";
import { SessionPill } from "@/components/conversation-shell";

const PROMPT_CARDS = [
  {
    id: "checkin",
    label: "Sort out what happened",
    description: "Walk me through the situation, what I felt, and what I need next.",
    prompt:
      "I want to sort out something that happened. Please help me name the situation, what I felt, what I needed, and what might help next.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M11 20A7 7 0 0 1 4 13c0-7 8-9 16-9 0 8-2 16-9 16z" />
        <path d="M4 20l9-9" />
      </svg>
    ),
  },
  {
    id: "breath",
    label: "Regulate first",
    description: "Help me settle my body before we figure out the problem.",
    prompt:
      "I feel activated right now. Please help me settle first, then ask a few questions to understand what triggered it.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M3 8h12a3 3 0 1 0-3-3" />
        <path d="M3 12h17a3 3 0 1 1-3 3" />
        <path d="M3 16h9" />
      </svg>
    ),
  },
  {
    id: "examine",
    label: "Untangle a thought",
    description: "Help me test a looping worry against what I actually know.",
    prompt:
      "A thought keeps looping in my head. Please help me separate facts, assumptions, feelings, and a more balanced way to look at it.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <circle cx="12" cy="12" r="9" />
        <path d="M15 9l-2 5-5 2 2-5z" />
      </svg>
    ),
  },
  {
    id: "stuck",
    label: "Find the next step",
    description: "Help me turn avoidance into one concrete action I can take.",
    prompt:
      "I feel stuck and I am avoiding something. Please help me identify the blocker and choose one concrete next step that is small enough to start.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <circle cx="12" cy="6" r="2" />
        <path d="M12 8v13M5 16a7 7 0 0 0 14 0" />
        <path d="M3 16h4M17 16h4" />
      </svg>
    ),
  },
  {
    id: "selfcomp",
    label: "Self-compassion",
    description: "I'm being really hard on myself. Help me reframe.",
    prompt: "I'm being really hard on myself right now. Is there something we can do about that?",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10z" />
      </svg>
    ),
  },
  {
    id: "letgo",
    label: "Let go of a thought",
    description: "I want to try sitting with a thought instead of arguing with it.",
    prompt: "I want to try letting go of a thought instead of arguing with it. Can we do that?",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z" />
        <path d="M19 15l.7 2L22 18l-2.3.5L19 21l-.7-2.5L16 18l2.3-1z" />
      </svg>
    ),
  },
  {
    id: "patterns",
    label: "Reflect on patterns",
    description: "I've been noticing a pattern in how I react to stress.",
    prompt: "I've been noticing a pattern in how I react to stress. Can we explore it?",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M3 7c2 0 2 2 4 2s2-2 4-2 2 2 4 2 2-2 4-2" />
        <path d="M3 13c2 0 2 2 4 2s2-2 4-2 2 2 4 2 2-2 4-2" />
        <path d="M3 19c2 0 2 0 4 0s2 0 4 0 2 0 4 0 2 0 4 0" />
      </svg>
    ),
  },
  {
    id: "memrecall",
    label: "What do you know?",
    description: "What do you remember about me from past conversations?",
    prompt: "What do you remember about me from our past conversations?",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3" />
        <path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3" />
        <path d="M9 4v16M15 4v16" />
      </svg>
    ),
  },
];

const IconClock = ({ size = 12 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);

const IconSend = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M4 12l16-8-6 16-3-6-7-2z" />
  </svg>
);

function friendlyThreadName(threadId: string): string {
  if (!threadId) return "thread";
  if (threadId.length <= 24 && !threadId.includes(":")) return threadId;
  if (threadId.includes(":")) {
    const parts = threadId.split(":");
    return parts[parts.length - 1].slice(-14);
  }
  return threadId.slice(0, 16) + "…";
}

export default function TextChatPage() {
  const router = useRouter();
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const setThreadId = useSessionStore((s) => s.setThreadId);
  const messages = useSessionStore((s) => s.messages);
  const setMessages = useSessionStore((s) => s.setMessages);
  const addMessage = useSessionStore((s) => s.addMessage);
  const clearMessages = useSessionStore((s) => s.clearMessages);
  const isLoading = useSessionStore((s) => s.chatLoading);
  const chatStreamingStarted = useSessionStore((s) => s.chatStreamingStarted);
  const stages = useSessionStore((s) => s.chatStages);
  const notice = useSessionStore((s) => s.chatNotice);
  const setNotice = useSessionStore((s) => s.setChatNotice);
  const setMemoryFacts = useSessionStore((s) => s.setMemoryFacts);
  const memoryRefreshVersion = useSessionStore((s) => s.memoryRefreshVersion);
  const lastEndedSession = useSessionStore((s) => s.lastEndedSession);
  const clearLastEndedSession = useSessionStore((s) => s.clearLastEndedSession);
  const responseModelTier = useSessionStore((s) => s.responseModelTier);
  const {
    runAction,
    startNewSession,
    endCurrentSession,
    canEndSession,
    endingSession,
    isBusy,
  } = useCommandActions();
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const wasLoadingRef = useRef(isLoading);

  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [recentThreads, setRecentThreads] = useState<ThreadSummary[]>([]);
  const [originThread, setOriginThread] = useState<string | null>(null);

  useEffect(() => {
    if (sessionMode === "incognito") return;
    if (messages.length > 0) return;
    let cancelled = false;
    getHistory(threadId)
      .then((history) => {
        const state = useSessionStore.getState();
        if (
          !cancelled &&
          state.threadId === threadId &&
          state.messages.length === 0 &&
          history.length > 0
        ) {
          setMessages(
            history.map((m: Message) => ({
              role: m.role,
              content: m.content,
              responseStyle: m.response_style,
            }))
          );
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNotice(
            "Could not load chat history. Check that the backend is running."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [threadId, sessionMode, messages.length, setMessages, setNotice]);

  useEffect(() => {
    if (sessionMode === "incognito") return;
    getMemoryStatus(threadId, userId)
      .then(setMemoryStatus)
      .catch(() => {
        setNotice("Could not load memory status.");
      });
    getMemoryFacts(threadId, userId || undefined)
      .then((facts) => setMemoryFacts(facts))
      .catch(() => {
        setNotice("Could not load memory facts.");
      });
  }, [
    threadId,
    userId,
    sessionMode,
    setMemoryFacts,
    memoryRefreshVersion,
    setNotice,
  ]);

  useEffect(() => {
    if (sessionMode === "incognito") return;
    getThreads(5)
      .then(setRecentThreads)
      .catch(() => {
        setNotice("Could not load recent sessions.");
      });
  }, [sessionMode, threadId, setNotice]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, stages]);

  useEffect(() => {
    const wasLoading = wasLoadingRef.current;
    wasLoadingRef.current = isLoading;

    if (wasLoading && !isLoading) {
      window.setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  }, [isLoading]);

  const sendMessage = useCallback(
    (text?: string) => {
      const msg = (text || input).trim();
      if (!msg) return;
      const slashCommand = resolveSlashCommand(msg);
      if (!slashCommand && isLoading) return;

      clearLastEndedSession();
      setNotice(null);
      setInput("");
      setOriginThread(null);

      if (inputRef.current) {
        inputRef.current.style.height = "auto";
      }

      if (slashCommand) {
        if (slashCommand.kind === "unsupported") {
          addMessage({ role: "assistant", content: slashCommand.message });
          inputRef.current?.focus();
          return;
        }

        void runAction(slashCommand.actionId)
          .then((handled) => {
            if (!handled && slashCommand.disabledMessage) {
              addMessage({
                role: "assistant",
                content: slashCommand.disabledMessage,
              });
            }
            inputRef.current?.focus();
          })
          .catch(() => {
            addMessage({
              role: "assistant",
              content: "Could not run that shortcut.",
            });
            inputRef.current?.focus();
          });
        return;
      }

      const started = startTextChatStream({
        message: msg,
        threadId,
        userId,
        sessionMode,
        responseModelTier,
      });
      if (started) {
        inputRef.current?.focus();
      }
    },
    [
      input,
      isLoading,
      threadId,
      userId,
      responseModelTier,
      sessionMode,
      addMessage,
      clearLastEndedSession,
      setNotice,
      runAction,
    ]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  const turnCount = messages.filter((m) => m.role === "user").length;
  const totalFacts = memoryStatus
    ? Object.values(memoryStatus.counts).reduce((a, b) => a + b, 0)
    : 0;

  const otherThreads = recentThreads.filter(
    (t) => t.thread_id !== threadId && t.turn_count > 0
  );

  const showWelcome = messages.length === 0 && !isLoading;
  const isPersistent = sessionMode === "persistent";
  const memoryCountText =
    isPersistent && totalFacts > 0
      ? `I remember ${totalFacts} thing${totalFacts === 1 ? "" : "s"} from our past time together. Start anywhere.`
      : isPersistent
        ? "Fresh thread — start anywhere."
        : "Incognito · nothing saved this session.";

  const greetingFirstName =
    isPersistent && userId ? userId.split(/[\s_-]/)[0] : null;

  const handleNewSession = useCallback(() => {
    void startNewSession();
  }, [startNewSession]);

  const handleContinueInThread = useCallback(() => {
    clearLastEndedSession();
    window.setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  }, [clearLastEndedSession]);

  const handleReviewMemory = useCallback(() => {
    clearLastEndedSession();
    router.push("/memory");
  }, [clearLastEndedSession, router]);

  return (
    <>
      {/* Desktop top bar — wrapper controls breakpoint visibility */}
      <div className="oc-app-top-wrap">
      <header className="oc-app-top">
        <div className="flex items-center gap-3">
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-app-top-title">Chat</h2>
          {turnCount > 0 && (
            <span className="text-[12px] font-mono text-oc-text-dim">
              {turnCount} turn{turnCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2.5">
          {isLoading && (
            <span className="text-[12px] font-mono text-oc-cta">
              {chatStreamingStarted ? "replying…" : "thinking…"}
            </span>
          )}
          {messages.length > 0 && !isLoading && (
            <button
              type="button"
              onClick={() => clearMessages()}
              className="oc-tab-chip"
              title="Clear chat"
            >
              clear
            </button>
          )}
          <MemoryToggleButton />
          <SessionPill />
        </div>
      </header>
      </div>

      {/* Mobile top bar — wrapper controls breakpoint visibility */}
      <div className="oc-mobile-top-wrap">
      <header className="oc-mobile-top">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-mobile-top-title">Chat</h2>
        </div>
        <SessionPill />
      </header>
      </div>

      {/* Notice banner */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {notice ?? ""}
      </div>
      {notice && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[13px] text-amber-800 md:mx-6">
          <div className="flex items-start justify-between gap-3">
            <span>{notice}</span>
            <button
              type="button"
              onClick={() => setNotice(null)}
              aria-label="Dismiss notification"
              className="shrink-0 text-amber-700 hover:text-amber-900"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Origin-thread back button */}
      {originThread && originThread !== threadId && (
        <div className="mx-4 mt-3 md:mx-6 animate-fadeIn">
          <button
            onClick={() => {
              setThreadId(originThread);
              setOriginThread(null);
            }}
            className="flex items-center gap-1.5 text-[12px] font-mono text-oc-teal-700 hover:text-oc-teal-600 px-2.5 py-1.5 rounded-lg border border-oc-teal-200 bg-oc-teal-50 hover:bg-oc-teal-100 transition-all"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            back to current session
          </button>
        </div>
      )}

      {/* Scrollable content */}
      <div
        ref={scrollRef}
        className={`oc-chat-scroll${showWelcome ? " oc-chat-scroll--welcome" : ""}`}
      >
        {showWelcome ? (
          <div
            className="oc-chat-inner oc-chat-inner--narrow animate-fadeIn"
            style={{ paddingTop: 28 }}
          >
            <div style={{ textAlign: "center", marginBottom: 32 }}>
              <div className="oc-welcome-eyebrow">
                ⌘ {isPersistent ? (greetingFirstName ? `welcome back, ${greetingFirstName}` : "welcome back") : "private session"}
              </div>
              <h1 className="oc-welcome-title">
                What&rsquo;s <em>here</em>
                {greetingFirstName ? <>, {greetingFirstName}</> : null}?
              </h1>
              <p className="oc-welcome-sub">
                {isPersistent && totalFacts > 0 ? (
                  <>
                    I remember <b>{totalFacts} thing{totalFacts === 1 ? "" : "s"}</b>{" "}
                    from our past time together. Start anywhere.
                  </>
                ) : (
                  memoryCountText
                )}
              </p>
            </div>

            <div className="oc-prompts-grid">
              {PROMPT_CARDS.slice(0, 4).map((card) => (
                <button
                  key={card.id}
                  type="button"
                  className="oc-prompt-card"
                  onClick={() => sendMessage(card.prompt)}
                >
                  <span className="oc-prompt-icon">{card.icon}</span>
                  <span className="oc-prompt-body">
                    <span className="oc-prompt-title">{card.label}</span>
                    <span className="oc-prompt-desc">{card.description}</span>
                  </span>
                </button>
              ))}
            </div>

            {isPersistent && otherThreads.length > 0 && (
              <div className="oc-recents">
                <div className="oc-recents-eyebrow">Recent threads</div>
                {otherThreads.slice(0, 3).map((t) => (
                  <button
                    key={t.thread_id}
                    type="button"
                    className="oc-recents-row"
                    onClick={() => {
                      if (!originThread) setOriginThread(threadId);
                      setThreadId(t.thread_id);
                    }}
                  >
                    <IconClock size={12} />
                    <span className="name">{friendlyThreadName(t.thread_id)}</span>
                    <span className="preview">
                      {t.message_count} message{t.message_count === 1 ? "" : "s"}
                    </span>
                    <span className="turns">
                      {t.turn_count} turn{t.turn_count !== 1 ? "s" : ""}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="oc-chat-inner">
            <div className="space-y-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className="animate-slideUp"
                  style={{ animationDelay: `${Math.min(i * 30, 200)}ms` }}
                >
                  {msg.role === "user" ? (
                    <div className="oc-bubble oc-bubble--user">
                      <div className="oc-bubble-body">{msg.content}</div>
                    </div>
                  ) : (
                    <div className="oc-bubble">
                      <div className="oc-bubble-mark">
                        <CouchLogo variant="mono" className="w-3.5 h-3.5" />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="oc-bubble-body">{msg.content}</div>
                        {msg.responseStyle && <StateStrip msg={msg} />}
                        {isPersistent &&
                          lastEndedSession?.threadId !== threadId &&
                          msg.sessionAction === "suggest_end_session" && (
                            <SessionClosureAction
                              disabled={!canEndSession || isBusy}
                              ending={endingSession}
                              onEndSession={() => void endCurrentSession()}
                            />
                          )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isLoading && !chatStreamingStarted && (
                <div
                  className="oc-bubble animate-fadeIn"
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  <div className="oc-bubble-mark">
                    <CouchLogo variant="mono" className="w-3.5 h-3.5" />
                  </div>
                  <div
                    className="oc-bubble-body"
                    style={{ minWidth: 180 }}
                  >
                    {stages.length > 0 && (
                      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 6 }}>
                        {stages.map((s, i) => (
                          <div
                            key={i}
                            className="animate-fadeIn"
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              fontFamily: "var(--font-mono)",
                              fontSize: 12,
                              color: "var(--color-oc-muted)",
                            }}
                          >
                            <svg
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2.4"
                              className="w-3.5 h-3.5"
                              style={{ color: "var(--color-oc-green)" }}
                            >
                              <polyline points="20 6 9 17 4 12" />
                            </svg>
                            <span>{s}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        fontFamily: "var(--font-mono)",
                        fontSize: 12,
                        color: "var(--color-oc-cta)",
                      }}
                    >
                      <span className="relative flex h-3.5 w-3.5 items-center justify-center shrink-0">
                        <span className="animate-ping absolute inline-flex h-2.5 w-2.5 rounded-full bg-oc-cta opacity-50" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-cta" />
                      </span>
                      <span>{stages.length === 0 ? "starting…" : "processing…"}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="oc-composer-bar">
        <div style={{ width: "100%", maxWidth: 680 }}>
          {isPersistent &&
          lastEndedSession?.threadId === threadId ? (
            <SessionEndedCard
              session={lastEndedSession}
              onNewSession={handleNewSession}
              onContinueInThread={handleContinueInThread}
              onReviewMemory={handleReviewMemory}
              newSessionDisabled={isBusy}
            />
          ) : null}
          <div className="oc-composer">
            <label htmlFor="chat-input" className="sr-only">
              Message
            </label>
            <textarea
              id="chat-input"
              ref={inputRef}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Say what's on your mind…"
              disabled={isLoading}
              autoFocus
              rows={1}
              className="oc-composer-input"
            />
            <button
              type="button"
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
              className="oc-composer-send"
            >
              <IconSend size={16} />
            </button>
          </div>
        </div>
      </div>

      <MemoryPanel />
    </>
  );
}

function SessionEndedCard({
  session,
  onNewSession,
  onContinueInThread,
  onReviewMemory,
  newSessionDisabled,
}: {
  session: EndedSessionResult;
  onNewSession: () => void;
  onContinueInThread: () => void;
  onReviewMemory: () => void;
  newSessionDisabled: boolean;
}) {
  const hasSummary = Boolean(session.summary);

  return (
    <div className="mb-3 rounded-2xl border border-oc-teal-200 bg-oc-teal-50/70 px-4 py-3 animate-fadeIn">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-teal-700">
          Session ended
        </span>
        <span className="h-1 w-1 rounded-full bg-oc-teal-300" />
        <span className="text-[12px] font-mono text-oc-text-dim">
          {hasSummary ? "memory committed" : "no summary produced"}
        </span>
      </div>
      <p className="mt-2 text-[14px] leading-relaxed text-oc-text">
        {session.summary || session.detail || "The session has been closed."}
      </p>
      {hasSummary && session.themes && session.themes.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {session.themes.map((theme) => (
            <span
              key={theme}
              className="rounded-md border border-oc-teal-200/70 bg-white/70 px-2 py-0.5 text-[11px] font-mono text-oc-teal-700"
            >
              {theme}
            </span>
          ))}
        </div>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onNewSession}
          disabled={newSessionDisabled}
          className="rounded-lg bg-oc-teal-700 px-3 py-1.5 text-[12px] font-medium text-white transition-colors hover:bg-oc-teal-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          Start new session
        </button>
        <button
          type="button"
          onClick={onContinueInThread}
          className="rounded-lg border border-oc-teal-200 bg-white/80 px-3 py-1.5 text-[12px] font-medium text-oc-teal-800 transition-colors hover:bg-white"
        >
          Continue in this thread
        </button>
        <button
          type="button"
          onClick={onReviewMemory}
          className="rounded-lg border border-oc-teal-200 bg-white/60 px-3 py-1.5 text-[12px] font-medium text-oc-text transition-colors hover:bg-white"
        >
          Review memory
        </button>
      </div>
    </div>
  );
}

function SessionClosureAction({
  disabled,
  ending,
  onEndSession,
}: {
  disabled: boolean;
  ending: boolean;
  onEndSession: () => void;
}) {
  return (
    <div className="mt-2 ml-0.5 flex flex-wrap items-center gap-2 text-[12px] text-oc-text-dim">
      <span>Ready to close this session?</span>
      <button
        type="button"
        onClick={onEndSession}
        disabled={disabled}
        className="rounded-lg border border-oc-teal-200 bg-oc-teal-50 px-2.5 py-1 text-[12px] font-medium text-oc-teal-800 transition-colors hover:bg-oc-teal-100 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {ending ? "Ending..." : "End session"}
      </button>
    </div>
  );
}

/* ── State Strip ── diagnostic pills under assistant messages ── */

function StateStrip({ msg }: { msg: ChatMessage }) {
  const [expanded, setExpanded] = useState(false);
  const crisis = msg.crisis;
  const diag = msg.diagnostics || {};

  const isCrisis = crisis?.needs_crisis_response;
  const safetyLabel = isCrisis
    ? "crisis"
    : crisis?.needs_clarification
      ? "check"
      : (crisis?.level ?? 0) >= 1
        ? "distress"
        : "safe";

  const totalMs = diag.turn_total_ms != null ? Number(diag.turn_total_ms) : null;

  return (
    <div className="mt-2 ml-0.5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex flex-wrap items-center gap-2 group text-left"
      >
        <Pill variant="teal">{msg.responseStyle}</Pill>
        {msg.therapeuticApproach && msg.therapeuticApproach !== "none" && (
          <Pill variant="teal">{msg.therapeuticApproach}</Pill>
        )}
        {msg.responseStyle === "grounded_lookup" ? (
          <Pill variant="muted">grounded</Pill>
        ) : null}
        {msg.responseStyle === "memory_control" ? (
          <Pill variant="muted">memory</Pill>
        ) : null}
        <Pill
          variant={
            safetyLabel === "safe"
              ? "green"
              : safetyLabel === "crisis"
                ? "red"
                : "amber"
          }
        >
          {safetyLabel}
        </Pill>
        {diag.resource_lookup_status != null &&
        String(diag.resource_lookup_status) !== "not_attempted" ? (
          <Pill variant="muted">resources {String(diag.resource_lookup_status)}</Pill>
        ) : null}
        {diag.retrieval_path != null ? (
          <Pill variant="muted">{String(diag.retrieval_path)}</Pill>
        ) : null}
        {totalMs != null && (
          <span className="text-[11px] font-mono text-oc-text-dim">
            {totalMs.toFixed(0)}ms
          </span>
        )}
        <span className="text-[11px] text-oc-text-dim group-hover:text-oc-text-muted transition-colors ml-0.5">
          {expanded ? "▾" : "▸"}
        </span>
      </button>

      {expanded && (
        <div className="mt-2.5 bg-oc-bg-state rounded-xl p-5 space-y-3.5 animate-slideUp shadow-lg border border-oc-warm-800/50">
          <div className="flex items-center gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0">
              route
            </span>
            <div className="flex items-center gap-1.5 font-mono text-[13px]">
              <span className="text-oc-teal-300">crisis_gate</span>
              <span className="text-oc-warm-600">→</span>
              <span
                className={
                  safetyLabel === "safe" ? "text-emerald-400" : "text-red-400"
                }
              >
                {safetyLabel}
              </span>
              <span className="text-oc-warm-600">→</span>
              <span className="text-oc-teal-300">
                {msg.responseType === "crisis"
                  ? "crisis_response"
                  : msg.responseStyle === "memory_control"
                    ? "memory_control"
                    : msg.responseStyle === "grounded_lookup"
                      ? "grounded_lookup"
                      : "therapeutic"}
              </span>
              <span className="text-oc-warm-600">→</span>
              <span className="text-amber-300">{msg.responseStyle}</span>
              {msg.therapeuticApproach && msg.therapeuticApproach !== "none" && (
                <>
                  <span className="text-oc-warm-600">·</span>
                  <span className="text-purple-300">{msg.therapeuticApproach}</span>
                </>
              )}
            </div>
          </div>

          {crisis?.reason && (
            <div className="flex items-start gap-3">
              <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
                reason
              </span>
              <span className="text-oc-warm-400 italic text-[13px]">
                {crisis.reason}
              </span>
            </div>
          )}

          <div className="flex items-start gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
              timing
            </span>
            <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 flex-1 font-mono">
              <TimingRow label="load_memory" ms={diag.load_memory_ms} />
              <TimingRow label="crisis_gate" ms={diag.crisis_gate_ms} />
              <TimingRow label="turn_dispatch" ms={diag.turn_dispatch_ms} />
              <TimingRow label="memory_control" ms={diag.memory_control_ms} />
              <TimingRow label="grounded_lookup" ms={diag.grounded_lookup_ms} />
              <TimingRow
                label="crisis_resources"
                ms={diag.crisis_resource_lookup_ms}
              />
              <TimingRow
                label="extract_facts"
                ms={diag.extract_facts_ms}
                extra={diag.semantic_writes}
              />
              <TimingRow
                label="extract_rules"
                ms={diag.extract_procedural_ms}
                extra={diag.procedural_writes}
              />
              <TimingRow label="total" ms={diag.turn_total_ms} bold />
            </div>
          </div>

          <div className="flex items-start gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
              memory
            </span>
            <div className="flex flex-wrap gap-4 font-mono text-[12px] text-oc-warm-400">
              <span>
                sem:{" "}
                <span className="text-oc-teal-300">
                  {String(diag.semantic_hits ?? 0)}
                </span>
                /{String(diag.semantic_store_size ?? 0)}
              </span>
              <span>
                epi:{" "}
                <span className="text-oc-teal-300">
                  {String(diag.episodic_hits ?? 0)}
                </span>
                /{String(diag.episodic_store_size ?? 0)}
              </span>
              <span>
                proc:{" "}
                <span className="text-oc-teal-300">
                  {String(diag.procedural_count ?? 0)}
                </span>
              </span>
              <span>
                recall:{" "}
                <span
                  className={
                    diag.proactive_recall ? "text-emerald-400" : "text-oc-warm-600"
                  }
                >
                  {diag.proactive_recall ? "on" : "off"}
                </span>
              </span>
            </div>
          </div>

          {(diag.extract_facts_reason != null ||
            diag.extract_procedural_reason != null) && (
            <div className="flex items-start gap-3">
              <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
                notes
              </span>
              <div className="space-y-1 text-[12px] font-mono text-oc-warm-500">
                {diag.extract_facts_reason != null && (
                  <div>
                    <span className="text-oc-teal-400">facts:</span>{" "}
                    {String(diag.extract_facts_reason)}
                  </div>
                )}
                {diag.extract_procedural_reason != null && (
                  <div>
                    <span className="text-oc-teal-400">rules:</span>{" "}
                    {String(diag.extract_procedural_reason)}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TimingRow({
  label,
  ms,
  extra,
  bold,
}: {
  label: string;
  ms: unknown;
  extra?: unknown;
  bold?: boolean;
}) {
  if (ms == null && extra == null && !bold) return null;

  const formatted = ms != null ? `${Number(ms).toFixed(0)}ms` : "—";
  const writes = extra != null ? ` (${String(extra)}w)` : "";
  return (
    <div
      className={`flex justify-between text-[12px] ${
        bold ? "text-oc-warm-200" : "text-oc-warm-500"
      }`}
    >
      <span>{label}</span>
      <span className="tabular-nums">
        {formatted}
        {writes && <span className="text-oc-warm-600">{writes}</span>}
      </span>
    </div>
  );
}

function Pill({
  children,
  variant,
}: {
  children: React.ReactNode;
  variant: "teal" | "muted" | "green" | "red" | "amber";
}) {
  const styles = {
    teal: "bg-oc-teal-50 text-oc-teal-700 border-oc-teal-200/60",
    muted: "bg-oc-warm-100 text-oc-warm-600 border-oc-warm-200",
    green: "bg-emerald-50 text-emerald-700 border-emerald-200/60",
    red: "bg-red-50 text-red-700 border-red-200/60",
    amber: "bg-amber-50 text-amber-700 border-amber-200/60",
  };

  return (
    <span
      className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded-md border ${styles[variant]}`}
    >
      {children}
    </span>
  );
}
