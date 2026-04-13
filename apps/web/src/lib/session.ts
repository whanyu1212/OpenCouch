"use client";

import { create } from "zustand";

/**
 * Shared session state across all pages (text, voice, memory).
 *
 * userId and threadId are editable from the sidebar. Both text
 * and voice modes read from this store so memory is shared.
 */

interface SessionState {
  userId: string;
  threadId: string;
  setUserId: (id: string) => void;
  setThreadId: (id: string) => void;
  newSession: () => void;
}

function generateThreadId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `web-${rand}`;
}

export const useSessionStore = create<SessionState>((set) => ({
  userId: "web-user",
  threadId: "web-default",
  setUserId: (id: string) => set({ userId: id }),
  setThreadId: (id: string) => set({ threadId: id }),
  newSession: () => set({ threadId: generateThreadId() }),
}));
