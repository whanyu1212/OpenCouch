"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  endSession,
  getMemoryFacts,
  getThreadSessionStatus,
  updateMemoryRecall,
  type ResponseModelTier,
} from "@/lib/api";
import { buildEndedSessionResult, useSessionStore } from "@/lib/session";

export type CommandActionId =
  | "show_help"
  | "open_chat"
  | "open_memory"
  | "open_state"
  | "open_threads"
  | "memory_recall_on"
  | "memory_recall_off"
  | "new_session"
  | "end_session"
  | "response_fast"
  | "response_quality";

export interface CommandAction {
  id: CommandActionId;
  label: string;
  description: string;
  group: "Session" | "Memory" | "Navigation" | "Preferences";
  disabled?: boolean;
  run: () => void | Promise<void>;
}

interface FinalizeSessionOptions {
  captureResult: boolean;
}

interface CommandActionsContextValue {
  actions: CommandAction[];
  canEndSession: boolean;
  commandPaletteOpen: boolean;
  endError: string | null;
  endingSession: boolean;
  hasActiveSession: boolean;
  identityLocked: boolean;
  isBusy: boolean;
  showTextResponseTier: boolean;
  threadDrawerOpen: boolean;
  closeCommandPalette: () => void;
  closeThreadDrawer: () => void;
  endCurrentSession: () => Promise<void>;
  finalizeCurrentPersistentSession: (
    options: FinalizeSessionOptions
  ) => Promise<unknown>;
  openCommandPalette: () => void;
  openThreadDrawer: () => void;
  runAction: (id: CommandActionId) => Promise<boolean>;
  setResponseTier: (tier: ResponseModelTier) => void;
  startNewSession: () => Promise<void>;
}

const CommandActionsContext =
  createContext<CommandActionsContextValue | null>(null);

export function CommandActionsProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const responseModelTier = useSessionStore((s) => s.responseModelTier);
  const setResponseModelTier = useSessionStore((s) => s.setResponseModelTier);
  const newSession = useSessionStore((s) => s.newSession);
  const chatLoading = useSessionStore((s) => s.chatLoading);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const messages = useSessionStore((s) => s.messages);
  const memoryFacts = useSessionStore((s) => s.memoryFacts);
  const setMemoryFacts = useSessionStore((s) => s.setMemoryFacts);
  const addUnseenMemories = useSessionStore((s) => s.addUnseenMemories);
  const bumpMemoryRefreshVersion = useSessionStore((s) => s.bumpMemoryRefreshVersion);
  const lastEndedSession = useSessionStore((s) => s.lastEndedSession);
  const setLastEndedSession = useSessionStore((s) => s.setLastEndedSession);
  const clearLastEndedSession = useSessionStore((s) => s.clearLastEndedSession);
  const isIncognito = sessionMode === "incognito";
  const hasSessionTurns = messages.some((message) => message.role === "user");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [threadDrawerOpen, setThreadDrawerOpen] = useState(false);
  const [endingSession, setEndingSession] = useState(false);
  const [endError, setEndError] = useState<string | null>(null);
  const [hasActiveSession, setHasActiveSession] = useState(false);
  const isBusy = chatLoading || endingSession || voiceConnected;
  const identityLocked =
    isBusy || (sessionMode === "persistent" && hasActiveSession);
  const canEndSession =
    sessionMode === "persistent" &&
    !voiceConnected &&
    hasActiveSession &&
    !isBusy;
  const showTextResponseTier = pathname !== "/voice";
  const openCommandPalette = useCallback(() => setCommandPaletteOpen(true), []);
  const closeCommandPalette = useCallback(() => setCommandPaletteOpen(false), []);
  const openThreadDrawer = useCallback(() => setThreadDrawerOpen(true), []);
  const closeThreadDrawer = useCallback(() => setThreadDrawerOpen(false), []);

  useEffect(() => {
    if (sessionMode !== "persistent" || voiceConnected) {
      setHasActiveSession(false);
      return;
    }

    let cancelled = false;
    const refreshStatus = () => {
      getThreadSessionStatus(threadId, sessionMode)
        .then((status) => {
          if (!cancelled) {
            setHasActiveSession(status.has_active_session);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setHasActiveSession(false);
          }
        });
    };

    refreshStatus();
    const intervalId = window.setInterval(refreshStatus, 30000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [threadId, sessionMode, voiceConnected, lastEndedSession]);

  useEffect(() => {
    if (
      sessionMode === "persistent" &&
      !voiceConnected &&
      chatLoading &&
      hasSessionTurns
    ) {
      setHasActiveSession(true);
    }
  }, [chatLoading, hasSessionTurns, sessionMode, voiceConnected]);

  const refreshSemanticFacts = useCallback(async () => {
    if (isIncognito) {
      setMemoryFacts([]);
      return;
    }

    const facts = await getMemoryFacts(threadId, userId || undefined, sessionMode);
    setMemoryFacts(facts);
    addUnseenMemories(Math.max(0, facts.length - memoryFacts.length));
  }, [
    addUnseenMemories,
    isIncognito,
    memoryFacts.length,
    setMemoryFacts,
    threadId,
    userId,
  ]);

  const finalizeCurrentPersistentSession = useCallback(
    async ({ captureResult }: FinalizeSessionOptions) => {
      if (
        sessionMode !== "persistent" ||
        voiceConnected ||
        !hasActiveSession
      ) {
        if (!captureResult) clearLastEndedSession();
        setEndError(null);
        return null;
      }

      setEndingSession(true);
      setEndError(null);

      try {
        const result = await endSession(threadId, undefined, sessionMode);
        try {
          await refreshSemanticFacts();
        } catch {
          // Ending a session should still succeed if the memory panel refresh fails.
        }
        bumpMemoryRefreshVersion();

        if (captureResult) {
          setLastEndedSession(
            buildEndedSessionResult({
              threadId,
              result,
              modality: "text",
            })
          );
        } else {
          clearLastEndedSession();
        }
        setHasActiveSession(false);
        return result;
      } catch {
        setEndError(
          captureResult
            ? "Could not end the current session."
            : "Could not finalize the previous session before switching. It will still close on timeout or shutdown."
        );
        return null;
      } finally {
        setEndingSession(false);
      }
    },
    [
      bumpMemoryRefreshVersion,
      clearLastEndedSession,
      hasActiveSession,
      refreshSemanticFacts,
      sessionMode,
      setLastEndedSession,
      threadId,
      voiceConnected,
    ]
  );

  const endCurrentSession = useCallback(async () => {
    if (!canEndSession) return;
    await finalizeCurrentPersistentSession({ captureResult: true });
  }, [canEndSession, finalizeCurrentPersistentSession]);

  const startNewSession = useCallback(async () => {
    if (isBusy) return;
    await finalizeCurrentPersistentSession({ captureResult: false });
    newSession();
  }, [finalizeCurrentPersistentSession, isBusy, newSession]);

  const setResponseTier = useCallback(
    (tier: ResponseModelTier) => {
      setResponseModelTier(tier);
    },
    [setResponseModelTier]
  );

  const setMemoryRecall = useCallback(
    async (enabled: boolean) => {
      if (isIncognito) return;
      await updateMemoryRecall(enabled, threadId, userId || undefined, sessionMode);
      bumpMemoryRefreshVersion();
    },
    [bumpMemoryRefreshVersion, isIncognito, threadId, userId]
  );

  const actions = useMemo<CommandAction[]>(
    () => [
      {
        id: "show_help",
        label: "Open actions",
        description: "Open the available web actions.",
        group: "Navigation",
        run: openCommandPalette,
      },
      {
        id: "open_chat",
        label: "Open chat",
        description: "Return to the text chat.",
        group: "Navigation",
        run: () => router.push("/"),
      },
      {
        id: "open_memory",
        label: "Open memory",
        description: "Review facts, sessions, and style rules.",
        group: "Memory",
        run: () => router.push("/memory"),
      },
      {
        id: "memory_recall_on",
        label: "Turn proactive recall on",
        description: "Allow relevant past sessions to be mentioned unprompted.",
        group: "Memory",
        disabled: isIncognito,
        run: () => setMemoryRecall(true),
      },
      {
        id: "memory_recall_off",
        label: "Turn proactive recall off",
        description: "Only reference saved memories when asked or required.",
        group: "Memory",
        disabled: isIncognito,
        run: () => setMemoryRecall(false),
      },
      {
        id: "open_state",
        label: "Open state",
        description: "Inspect the current graph state.",
        group: "Navigation",
        run: () => router.push("/state"),
      },
      {
        id: "open_threads",
        label: "Open threads",
        description: "Browse and resume previous persistent sessions.",
        group: "Session",
        run: openThreadDrawer,
      },
      {
        id: "new_session",
        label: "Return home",
        description: "Go back to the landing page to choose memory mode, user, and thread.",
        group: "Session",
        disabled: isBusy,
        run: startNewSession,
      },
      {
        id: "end_session",
        label: "End session",
        description: "Summarize and close the active persistent session.",
        group: "Session",
        disabled: !canEndSession,
        run: endCurrentSession,
      },
      {
        id: "response_fast",
        label: "Use fast replies",
        description: "Prefer the faster text response model.",
        group: "Preferences",
        disabled: chatLoading || responseModelTier === "fast",
        run: () => setResponseTier("fast"),
      },
      {
        id: "response_quality",
        label: "Use higher quality replies",
        description: "Prefer the stronger text response model.",
        group: "Preferences",
        disabled: chatLoading || responseModelTier === "quality",
        run: () => setResponseTier("quality"),
      },
    ],
    [
      canEndSession,
      chatLoading,
      endCurrentSession,
      isBusy,
      isIncognito,
      openCommandPalette,
      openThreadDrawer,
      responseModelTier,
      router,
      setMemoryRecall,
      setResponseTier,
      startNewSession,
    ]
  );

  const runAction = useCallback(
    async (id: CommandActionId) => {
      const action = actions.find((candidate) => candidate.id === id);
      if (!action || action.disabled) return false;
      await action.run();
      return true;
    },
    [actions]
  );

  const value = useMemo<CommandActionsContextValue>(
    () => ({
      actions,
      canEndSession,
      closeThreadDrawer,
      commandPaletteOpen,
      closeCommandPalette,
      endError,
      endingSession,
      hasActiveSession,
      identityLocked,
      isBusy,
      showTextResponseTier,
      threadDrawerOpen,
      endCurrentSession,
      finalizeCurrentPersistentSession,
      openCommandPalette,
      openThreadDrawer,
      runAction,
      setResponseTier,
      startNewSession,
    }),
    [
      actions,
      canEndSession,
      closeCommandPalette,
      closeThreadDrawer,
      commandPaletteOpen,
      endCurrentSession,
      endError,
      endingSession,
      finalizeCurrentPersistentSession,
      hasActiveSession,
      identityLocked,
      isBusy,
      openCommandPalette,
      openThreadDrawer,
      runAction,
      setResponseTier,
      showTextResponseTier,
      startNewSession,
      threadDrawerOpen,
    ]
  );

  return (
    <CommandActionsContext.Provider value={value}>
      {children}
    </CommandActionsContext.Provider>
  );
}

export function useCommandActions() {
  const value = useContext(CommandActionsContext);
  if (!value) {
    throw new Error("useCommandActions must be used inside CommandActionsProvider");
  }
  return value;
}
