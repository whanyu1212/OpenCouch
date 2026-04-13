"use client";

import { create } from "zustand";

/**
 * Shared session state across all pages (text, voice, memory, state).
 *
 * Chat messages are stored here (not in page-local useState) so they
 * survive tab navigation. When the user switches from Chat → Memory
 * → back to Chat, the messages are still there.
 */

export type SessionMode = "persistent" | "incognito";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  mode?: string | null;
  modeSource?: string | null;
  modality?: string | null;
  responseType?: string | null;
  crisis?: {
    level: number;
    confidence: string;
    reason: string;
    needs_crisis_response: boolean;
    needs_clarification: boolean;
  } | null;
  diagnostics?: Record<string, unknown> | null;
}

interface SessionState {
  /** Whether the user has completed setup for this session */
  isSetup: boolean;
  /** Memory mode selected during setup */
  sessionMode: SessionMode;
  userId: string;
  threadId: string;

  /** Chat messages — persisted across tab switches */
  messages: ChatMessage[];
  /** Whether the agent is currently generating a response */
  chatLoading: boolean;

  setUserId: (id: string) => void;
  setThreadId: (id: string) => void;
  setMessages: (msgs: ChatMessage[]) => void;
  addMessage: (msg: ChatMessage) => void;
  clearMessages: () => void;
  setChatLoading: (loading: boolean) => void;
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
  messages: [],
  chatLoading: false,

  setUserId: (id: string) => set({ userId: id }),
  setThreadId: (id: string) => set({ threadId: id }),
  setMessages: (msgs: ChatMessage[]) => set({ messages: msgs }),
  addMessage: (msg: ChatMessage) =>
    set((state) => ({ messages: [...state.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  setChatLoading: (loading: boolean) => set({ chatLoading: loading }),

  startSession: (mode: SessionMode, userId: string, threadId: string) =>
    set({
      isSetup: true,
      sessionMode: mode,
      userId: mode === "incognito" ? "" : (userId.trim() || "web-user"),
      threadId: mode === "incognito" ? generateThreadId() : (threadId.trim() || generateThreadId()),
      messages: [],
      chatLoading: false,
    }),
  newSession: () =>
    set({
      isSetup: false,
      sessionMode: "persistent",
      threadId: generateThreadId(),
      messages: [],
      chatLoading: false,
    }),
}));
