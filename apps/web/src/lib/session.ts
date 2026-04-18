"use client";

import { create } from "zustand";
import type { MemoryFact } from "./api";

/**
 * Shared session state across all pages (text, voice, memory, state).
 *
 * Chat messages are stored here (not in page-local useState) so they
 * survive tab navigation. When the user switches from Chat → Memory
 * → back to Chat, the messages are still there.
 *
 * Voice connection state also lives here so a voice session stays alive
 * when the user navigates to another in-app tab (Chat, Memory, State).
 * Non-serializable refs (WebSocket, AudioContext, etc.) live in module-
 * level variables to avoid triggering Zustand subscriber re-renders.
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

export interface VoiceTranscript {
  role: "user" | "assistant";
  text: string;
}

// ── Module-level voice refs (non-reactive, never trigger re-renders) ──
// These are singleton — only one voice session exists at a time.
let _voiceWs: WebSocket | null = null;
let _voiceAudioCtx: AudioContext | null = null;
let _voiceMediaStream: MediaStream | null = null;
let _voiceProcessor: ScriptProcessorNode | null = null;
let _voiceNextPlayTime = 0;
let _voiceGeneration = 0;

/** Read-only access to voice refs from outside the store */
export function getVoiceRefs() {
  return {
    ws: _voiceWs,
    audioCtx: _voiceAudioCtx,
    mediaStream: _voiceMediaStream,
    processor: _voiceProcessor,
    nextPlayTime: _voiceNextPlayTime,
    generation: _voiceGeneration,
  };
}

export function setVoiceNextPlayTime(t: number) {
  _voiceNextPlayTime = t;
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

  /** Live memory panel state */
  memoryFacts: MemoryFact[];
  memoryPanelOpen: boolean;
  /** Count of facts added since panel was last viewed */
  memoryUnseenCount: number;

  // ── Voice session (reactive UI state only) ────────────────────────
  voiceConnected: boolean;
  voiceAgentSpeaking: boolean;
  voiceTranscripts: VoiceTranscript[];
  voiceError: string | null;

  setUserId: (id: string) => void;
  setThreadId: (id: string) => void;
  setMessages: (msgs: ChatMessage[]) => void;
  addMessage: (msg: ChatMessage) => void;
  /** Append text to the last message (for token streaming) */
  appendToLastMessage: (text: string) => void;
  /** Patch the last message with metadata (for DoneEvent reconciliation) */
  updateLastMessage: (updates: Partial<ChatMessage>) => void;
  clearMessages: () => void;
  setChatLoading: (loading: boolean) => void;
  /** Start a session with the given mode */
  startSession: (mode: SessionMode, userId: string, threadId: string) => void;
  /** Reset to setup screen with a new thread */
  newSession: () => void;
  setMemoryFacts: (facts: MemoryFact[]) => void;
  setMemoryPanelOpen: (open: boolean) => void;
  addUnseenMemories: (count: number) => void;
  clearUnseenMemories: () => void;

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected: boolean) => void;
  setVoiceAgentSpeaking: (speaking: boolean) => void;
  addVoiceTranscript: (t: VoiceTranscript) => void;
  setVoiceError: (error: string | null) => void;
  /**
   * Store voice refs in module-level vars and bump generation.
   * Returns the generation so callers can scope their onclose.
   */
  voiceSetRefs: (refs: {
    ws?: WebSocket | null;
    audioCtx?: AudioContext | null;
    mediaStream?: MediaStream | null;
    processor?: ScriptProcessorNode | null;
  }) => void;
  /** Tear down voice resources — only if generation matches */
  voiceCleanup: (generation: number) => void;
  /** Full disconnect: close WS + cleanup (always runs) */
  voiceDisconnect: () => void;
  /** Bump generation, returning the new value (used by connect) */
  voiceNewGeneration: () => number;
  /** Clear transcripts (e.g. on fresh connect) */
  clearVoiceTranscripts: () => void;
}

function generateThreadId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `web-${rand}`;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  isSetup: false,
  sessionMode: "persistent",
  userId: "",
  threadId: "",
  messages: [],
  chatLoading: false,
  memoryFacts: [],
  memoryPanelOpen: false,
  memoryUnseenCount: 0,

  // Voice defaults (reactive only)
  voiceConnected: false,
  voiceAgentSpeaking: false,
  voiceTranscripts: [],
  voiceError: null,

  setUserId: (id: string) => set({ userId: id }),
  setThreadId: (id: string) => set({ threadId: id }),
  setMessages: (msgs: ChatMessage[]) => set({ messages: msgs }),
  addMessage: (msg: ChatMessage) =>
    set((state) => ({ messages: [...state.messages, msg] })),
  appendToLastMessage: (text: string) =>
    set((state) => {
      const msgs = [...state.messages];
      if (msgs.length > 0) {
        const last = { ...msgs[msgs.length - 1] };
        last.content = (last.content || "") + text;
        msgs[msgs.length - 1] = last;
      }
      return { messages: msgs };
    }),
  updateLastMessage: (updates: Partial<ChatMessage>) =>
    set((state) => {
      const msgs = [...state.messages];
      if (msgs.length > 0) {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...updates };
      }
      return { messages: msgs };
    }),
  clearMessages: () => set({ messages: [] }),
  setChatLoading: (loading: boolean) => set({ chatLoading: loading }),

  setMemoryFacts: (facts: MemoryFact[]) => set({ memoryFacts: facts }),
  setMemoryPanelOpen: (open: boolean) =>
    set((state) => ({
      memoryPanelOpen: open,
      memoryUnseenCount: open ? 0 : state.memoryUnseenCount,
    })),
  addUnseenMemories: (count: number) =>
    set((state) => ({
      memoryUnseenCount: state.memoryPanelOpen ? 0 : state.memoryUnseenCount + count,
    })),
  clearUnseenMemories: () => set({ memoryUnseenCount: 0 }),

  startSession: (mode: SessionMode, userId: string, threadId: string) =>
    set({
      isSetup: true,
      sessionMode: mode,
      userId: mode === "incognito" ? "" : (userId.trim() || "web-user"),
      threadId: mode === "incognito" ? generateThreadId() : (threadId.trim() || generateThreadId()),
      messages: [],
      chatLoading: false,
      memoryFacts: [],
      memoryUnseenCount: 0,
    }),
  newSession: () => {
    // Tear down any active voice session when resetting
    get().voiceDisconnect();
    set({
      isSetup: false,
      sessionMode: "persistent",
      threadId: generateThreadId(),
      messages: [],
      chatLoading: false,
      memoryFacts: [],
      memoryUnseenCount: 0,
    });
  },

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected) => set({ voiceConnected: connected }),
  setVoiceAgentSpeaking: (speaking) => set({ voiceAgentSpeaking: speaking }),
  addVoiceTranscript: (t) =>
    set((state) => ({ voiceTranscripts: [...state.voiceTranscripts, t] })),
  setVoiceError: (error) => set({ voiceError: error }),
  clearVoiceTranscripts: () => set({ voiceTranscripts: [] }),

  voiceNewGeneration: () => {
    _voiceGeneration += 1;
    return _voiceGeneration;
  },

  voiceSetRefs: (refs) => {
    if (refs.ws !== undefined) _voiceWs = refs.ws;
    if (refs.audioCtx !== undefined) _voiceAudioCtx = refs.audioCtx;
    if (refs.mediaStream !== undefined) _voiceMediaStream = refs.mediaStream;
    if (refs.processor !== undefined) _voiceProcessor = refs.processor;
  },

  voiceCleanup: (generation: number) => {
    // Only clean up if this generation is still the active one
    if (generation !== _voiceGeneration) return;
    _voiceProcessor?.disconnect();
    _voiceMediaStream?.getTracks().forEach((t) => t.stop());
    _voiceAudioCtx?.close();
    _voiceProcessor = null;
    _voiceMediaStream = null;
    _voiceAudioCtx = null;
    _voiceNextPlayTime = 0;
  },

  voiceDisconnect: () => {
    // Force-bump generation so any pending onclose becomes a no-op
    _voiceGeneration += 1;
    _voiceWs?.close();
    _voiceProcessor?.disconnect();
    _voiceMediaStream?.getTracks().forEach((t) => t.stop());
    _voiceAudioCtx?.close();
    _voiceWs = null;
    _voiceProcessor = null;
    _voiceMediaStream = null;
    _voiceAudioCtx = null;
    _voiceNextPlayTime = 0;
    set({
      voiceConnected: false,
      voiceAgentSpeaking: false,
    });
  },
}));
