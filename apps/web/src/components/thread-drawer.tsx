"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { getThreads, type ThreadSummary } from "@/lib/api";
import { useCommandActions } from "@/lib/command-actions";
import { useSessionStore } from "@/lib/session";

export function ThreadDrawer() {
  const {
    closeThreadDrawer,
    finalizeCurrentPersistentSession,
    isBusy,
    threadDrawerOpen,
  } = useCommandActions();
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const threadId = useSessionStore((s) => s.threadId);
  const setThreadId = useSessionStore((s) => s.setThreadId);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadThreads = useCallback(async () => {
    if (sessionMode === "incognito") {
      setThreads([]);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await getThreads(30, sessionMode);
      setThreads(result.filter((thread) => thread.turn_count > 0));
    } catch {
      setError("Could not load previous sessions.");
    } finally {
      setLoading(false);
    }
  }, [sessionMode]);

  useEffect(() => {
    if (!threadDrawerOpen) return;
    void loadThreads();
  }, [loadThreads, threadDrawerOpen]);

  useEffect(() => {
    if (!threadDrawerOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeThreadDrawer();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [closeThreadDrawer, threadDrawerOpen]);

  const handleSelectThread = useCallback(
    async (nextThreadId: string) => {
      if (isBusy) return;
      if (nextThreadId === threadId) {
        closeThreadDrawer();
        return;
      }

      await finalizeCurrentPersistentSession({ captureResult: false });
      setThreadId(nextThreadId);
      closeThreadDrawer();
    },
    [
      closeThreadDrawer,
      finalizeCurrentPersistentSession,
      isBusy,
      setThreadId,
      threadId,
    ]
  );

  if (!threadDrawerOpen) return null;

  return (
    <div
      className="fixed inset-0 z-40 bg-oc-teal-950/20 backdrop-blur-[2px] flex justify-end animate-fadeIn"
      onMouseDown={closeThreadDrawer}
    >
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="thread-drawer-title"
        className="h-full w-full max-w-md bg-oc-bg-card border-l border-oc-border-strong shadow-2xl flex flex-col"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="px-5 py-4 border-b border-oc-border flex items-center justify-between">
          <div>
            <h2
              id="thread-drawer-title"
              className="font-display text-lg text-oc-teal-900"
            >
              Previous sessions
            </h2>
            <p className="mt-0.5 text-[12px] text-oc-text-muted">
              Resume a persistent thread by selecting it below.
            </p>
          </div>
          <button
            type="button"
            onClick={closeThreadDrawer}
            className="text-[13px] font-mono text-oc-text-muted hover:text-oc-text transition-colors"
          >
            close
          </button>
        </header>

        <div className="px-5 py-3 border-b border-oc-border flex items-center justify-between">
          <span className="text-[12px] font-mono text-oc-text-dim">
            current: {threadId}
          </span>
          <button
            type="button"
            onClick={() => void loadThreads()}
            disabled={loading}
            className="text-[12px] font-mono text-oc-teal-600 hover:text-oc-teal-500 disabled:opacity-50"
          >
            {loading ? "loading..." : "refresh"}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {sessionMode === "incognito" ? (
            <EmptyState message="Previous sessions are available only in persistent mode." />
          ) : error ? (
            <div className="rounded-xl border border-oc-red/20 bg-oc-red-subtle px-4 py-3 text-[14px] text-oc-red">
              {error}
            </div>
          ) : loading ? (
            <div className="flex items-center gap-2 text-oc-text-muted text-sm font-mono">
              <div className="dot-pulse"><span /><span /><span /></div>
              loading
            </div>
          ) : threads.length === 0 ? (
            <EmptyState message="No previous persistent sessions yet." />
          ) : (
            <div className="space-y-2">
              {threads.map((thread) => {
                const isCurrent = thread.thread_id === threadId;
                return (
                  <button
                    key={thread.thread_id}
                    type="button"
                    disabled={isBusy}
                    onClick={() => void handleSelectThread(thread.thread_id)}
                    className={`w-full text-left rounded-xl border px-4 py-3 transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                      isCurrent
                        ? "border-oc-teal-200 bg-oc-teal-50"
                        : "border-oc-border bg-oc-bg hover:border-oc-border-strong hover:bg-oc-warm-50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <span className="font-mono text-[13px] text-oc-text-secondary truncate">
                        {thread.thread_id}
                      </span>
                      {isCurrent ? (
                        <span className="shrink-0 text-[11px] font-mono text-oc-teal-700">
                          current
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <ThreadTag>{thread.turn_count} turns</ThreadTag>
                      <ThreadTag>{thread.message_count} messages</ThreadTag>
                      {thread.has_context ? <ThreadTag>context</ThreadTag> : null}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {isBusy ? (
          <div className="px-5 py-3 border-t border-oc-border bg-oc-cta-subtle text-[12px] text-oc-cta">
            Wait for the current response or voice session to finish before
            switching threads.
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-oc-border bg-oc-bg px-4 py-8 text-center text-[14px] text-oc-text-muted">
      {message}
    </div>
  );
}

function ThreadTag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-md border border-oc-border bg-oc-bg-card px-2 py-1 text-[11px] font-mono text-oc-text-muted">
      {children}
    </span>
  );
}
