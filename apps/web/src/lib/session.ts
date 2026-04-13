"use client";

import { create } from "zustand";

/**
 * Shared session state across all pages (text, voice, memory, state).
 *
 * The session has two phases:
 * 1. Setup — user picks persistent or incognito mode
 * 2. Active — chat/voice/memory pages are usable
 *
 * In "persistent" mode, the userId and threadId carry over from the
 * sidebar inputs and history/memory is loaded. In "incognito" mode,
 * a random thread ID is generated and no user_id is sent to the
 * backend, which means the memory store has nothing to retrieve.
 */

export type SessionMode = "persistent" | "incognito";

interface SessionState {
  /** Whether the user has completed setup for this session */
  isSetup: boolean;
  /** Memory mode selected during setup */
  sessionMode: SessionMode;
  userId: string;
  threadId: string;
  setUserId: (id: string) => void;
  setThreadId: (id: string) => void;
  /** Start a session with the given mode */
  startSession: (mode: SessionMode, userId: string, threadId: string) => void;
  /** Reset to setup screen with a new thread */
  newSession: () => void;
}

function generateThreadId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `web-${rand}`;
}

export const useSessionStore = create<SessionState>((set) => ({
  isSetup: false,
  sessionMode: "persistent",
  userId: "",
  threadId: "",
  setUserId: (id: string) => set({ userId: id }),
  setThreadId: (id: string) => set({ threadId: id }),
  startSession: (mode: SessionMode, userId: string, threadId: string) =>
    set({
      isSetup: true,
      sessionMode: mode,
      // Incognito: no user_id, random thread. Persistent: fall back
      // to sensible defaults if the user left fields empty.
      userId: mode === "incognito" ? "" : (userId.trim() || "web-user"),
      threadId: mode === "incognito" ? generateThreadId() : (threadId.trim() || generateThreadId()),
    }),
  newSession: () =>
    set({
      isSetup: false,
      sessionMode: "persistent",
      threadId: generateThreadId(),
    }),
}));
