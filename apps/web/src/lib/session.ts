"use client";

import { useSyncExternalStore } from "react";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type {
  EndSessionResponse,
  MemoryFact,
  ResponseModelTier,
  TranscriptionLanguageOption,
} from "./api";

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
  responseStyle?: string | null;
  responseStyleSource?: string | null;
  therapeuticApproach?: string | null;
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
  role: "user" | "assistant" | "system";
  text: string;
  itemId?: string;
}

export type VoiceActivityName =
  | "memory_saved"
  | "memory_recall_updated"
  | "memory_delete_pending"
  | "memory_deleted"
  | "factual_lookup"
  | "crisis_resources_lookup"
  | "exercise";

export type VoiceActivityStatus =
  | "started"
  | "completed"
  | "failed"
  | "pending"
  | "cancelled";

export interface VoiceActivityEvent {
  id: string;
  activity: VoiceActivityName;
  status: VoiceActivityStatus;
  label: string;
  detail: string;
  timestamp: string;
}

export interface VoiceFinalizationState {
  threadId: string | null;
  status: "idle" | "in_progress" | "completed" | "failed";
  detail: string | null;
  updatedAt: string | null;
}

export interface VoiceSessionInfo {
  roomName: string;
  identity: string;
  memoryMode: string;
  serverUrl: string;
  connectedAt: string;
}

export interface EndedSessionResult extends EndSessionResponse {
  threadId: string;
}

type VoiceRoomHandle = {
  disconnect: (stopTracks?: boolean) => Promise<void> | void;
};

const IDLE_VOICE_FINALIZATION_STATE: VoiceFinalizationState = {
  threadId: null,
  status: "idle",
  detail: null,
  updatedAt: null,
};

// LiveKit Room is non-reactive and kept outside Zustand state to avoid
// re-rendering subscribers on SDK object mutation.
let _voiceRoom: VoiceRoomHandle | null = null;

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
  /** Bumped when memory surfaces should re-fetch from the backend */
  memoryRefreshVersion: number;
  /** Most recent explicit session-end result for the current thread */
  lastEndedSession: EndedSessionResult | null;
  /** User-facing text response preference */
  responseModelTier: ResponseModelTier;

  // ── Voice session (reactive UI state only) ────────────────────────
  voiceConnected: boolean;
  voiceAgentSpeaking: boolean;
  voiceReadyToSpeak: boolean;
  transcriptionLanguageSelected: TranscriptionLanguageOption;
  voiceTranscripts: VoiceTranscript[];
  voiceActivities: VoiceActivityEvent[];
  voiceFinalization: VoiceFinalizationState;
  voiceSessionInfo: VoiceSessionInfo | null;
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
  bumpMemoryRefreshVersion: () => void;
  setLastEndedSession: (session: EndedSessionResult | null) => void;
  clearLastEndedSession: () => void;
  setResponseModelTier: (tier: ResponseModelTier) => void;

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected: boolean) => void;
  setVoiceAgentSpeaking: (speaking: boolean) => void;
  setVoiceReadyToSpeak: (ready: boolean) => void;
  setTranscriptionLanguageSelected: (language: TranscriptionLanguageOption) => void;
  addVoiceTranscript: (t: VoiceTranscript) => void;
  addVoiceActivity: (event: VoiceActivityEvent) => void;
  clearVoiceActivities: () => void;
  setVoiceFinalization: (state: VoiceFinalizationState) => void;
  clearVoiceFinalization: () => void;
  setVoiceSessionInfo: (info: VoiceSessionInfo | null) => void;
  setVoiceError: (error: string | null) => void;
  /** Store the LiveKit room handle outside reactive Zustand state. */
  voiceSetRefs: (refs: { room?: VoiceRoomHandle | null }) => void;
  /** Disconnect active voice resources and mark memory finalization pending. */
  voiceDisconnect: () => void;
  /** Clear transcripts (e.g. on fresh connect) */
  clearVoiceTranscripts: () => void;
}

function generateThreadId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `web-${rand}`;
}

// Note: `isSetup` and `threadId` are intentionally NOT persisted.
// - `isSetup`: every fresh page load returns to the landing screen so the
//   user makes an explicit choice about identity and memory mode.
// - `threadId`: per-session ephemeral; persisting it caused stale thread
//   IDs to bleed across mode switches (e.g., incognito → persistent
//   carrying over the ID). The form auto-generates a fresh one if blank.
// The remembered fields below act as prefill defaults for the setup form,
// not as a way to skip past it.
type PersistedSessionState = Pick<
  SessionState,
  | "sessionMode"
  | "userId"
  | "responseModelTier"
  | "transcriptionLanguageSelected"
>;

type SessionStorePersistApi = {
  persist?: {
    hasHydrated: () => boolean;
    onHydrate: (callback: () => void) => () => void;
    onFinishHydration: (callback: () => void) => () => void;
  };
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
  isSetup: false,
  sessionMode: "persistent",
  userId: "",
  threadId: "",
  messages: [],
  chatLoading: false,
  memoryFacts: [],
  memoryPanelOpen: false,
  memoryUnseenCount: 0,
  memoryRefreshVersion: 0,
  lastEndedSession: null,
  responseModelTier: "fast",

  // Voice defaults (reactive only)
  voiceConnected: false,
  voiceAgentSpeaking: false,
  voiceReadyToSpeak: false,
  transcriptionLanguageSelected: "en",
  voiceTranscripts: [],
  voiceActivities: [],
  voiceFinalization: IDLE_VOICE_FINALIZATION_STATE,
  voiceSessionInfo: null,
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
  bumpMemoryRefreshVersion: () =>
    set((state) => ({ memoryRefreshVersion: state.memoryRefreshVersion + 1 })),
  setLastEndedSession: (session) => set({ lastEndedSession: session }),
  clearLastEndedSession: () => set({ lastEndedSession: null }),
  setResponseModelTier: (tier) => set({ responseModelTier: tier }),

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
      lastEndedSession: null,
      voiceSessionInfo: null,
      voiceActivities: [],
    }),
  newSession: () => {
    // Tear down any active voice session when resetting
    get().voiceDisconnect();
    set({
      isSetup: false,
      sessionMode: "persistent",
      // Clear the threadId so the setup form opens with an empty input.
      // startSession() generates a fresh one if the user leaves it blank.
      threadId: "",
      messages: [],
      chatLoading: false,
      memoryFacts: [],
      memoryUnseenCount: 0,
      lastEndedSession: null,
      voiceSessionInfo: null,
      voiceActivities: [],
    });
  },

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected) => set({ voiceConnected: connected }),
  setVoiceAgentSpeaking: (speaking) => set({ voiceAgentSpeaking: speaking }),
  setVoiceReadyToSpeak: (ready) => set({ voiceReadyToSpeak: ready }),
  setTranscriptionLanguageSelected: (language) =>
    set({ transcriptionLanguageSelected: language }),
  addVoiceTranscript: (t) =>
    set((state) => {
      if (t.itemId) {
        const existingIndex = state.voiceTranscripts.findIndex(
          (candidate) => candidate.role === t.role && candidate.itemId === t.itemId
        );
        if (existingIndex !== -1) {
          const transcripts = [...state.voiceTranscripts];
          transcripts[existingIndex] = { ...transcripts[existingIndex], text: t.text };
          return { voiceTranscripts: transcripts };
        }
      }
      return { voiceTranscripts: [...state.voiceTranscripts, t] };
    }),
  addVoiceActivity: (event) =>
    set((state) => ({
      voiceActivities: [...state.voiceActivities, event].slice(-12),
    })),
  clearVoiceActivities: () => set({ voiceActivities: [] }),
  setVoiceFinalization: (voiceFinalization) => set({ voiceFinalization }),
  clearVoiceFinalization: () =>
    set({ voiceFinalization: IDLE_VOICE_FINALIZATION_STATE }),
  setVoiceSessionInfo: (voiceSessionInfo) => set({ voiceSessionInfo }),
  setVoiceError: (error) => set({ voiceError: error }),
  clearVoiceTranscripts: () => set({ voiceTranscripts: [] }),

  voiceSetRefs: (refs) => {
    if (refs.room !== undefined) _voiceRoom = refs.room;
  },

  voiceDisconnect: () => {
    const shouldTrackFinalization = get().voiceConnected || _voiceRoom !== null;
    const threadId = get().threadId;

    _voiceRoom?.disconnect();
    _voiceRoom = null;
    set({
      voiceConnected: false,
      voiceAgentSpeaking: false,
      voiceReadyToSpeak: false,
      voiceFinalization:
        shouldTrackFinalization && threadId
          ? {
              threadId,
              status: "in_progress",
              detail: "Saving session memory...",
              updatedAt: new Date().toISOString(),
            }
          : get().voiceFinalization,
    });
  },
    }),
    {
      name: "opencouch-web-session",
      storage: createJSONStorage(() => localStorage),
      partialize: (state): PersistedSessionState => ({
        sessionMode: state.sessionMode,
        userId: state.userId,
        responseModelTier: state.responseModelTier,
        transcriptionLanguageSelected: state.transcriptionLanguageSelected,
      }),
      merge: (persisted, current): SessionState => {
        const saved = persisted as Partial<PersistedSessionState> | undefined;
        return {
          ...current,
          ...saved,
          // Always start at the landing screen — even if an older deploy
          // stored isSetup: true in localStorage, drop it on rehydration.
          isSetup: false,
          // Drop any persisted threadId from older deploys; threadId is
          // ephemeral and should never carry across reloads/mode switches.
          threadId: "",
          messages: [],
          chatLoading: false,
          memoryFacts: [],
          memoryPanelOpen: false,
          memoryUnseenCount: 0,
          memoryRefreshVersion: 0,
          lastEndedSession: null,
          voiceConnected: false,
          voiceAgentSpeaking: false,
          voiceReadyToSpeak: false,
          voiceTranscripts: [],
          voiceActivities: [],
          voiceFinalization: IDLE_VOICE_FINALIZATION_STATE,
          voiceSessionInfo: null,
          voiceError: null,
        };
      },
    }
  )
);

function getPersistApi() {
  return (useSessionStore as typeof useSessionStore & SessionStorePersistApi)
    .persist;
}

export function useSessionStoreHydrated(): boolean {
  return useSyncExternalStore(
    (onStoreChange) => {
      const persistApi = getPersistApi();
      if (!persistApi) return () => {};

      const unsubscribeHydrate = persistApi.onHydrate(onStoreChange);
      const unsubscribeFinish = persistApi.onFinishHydration(onStoreChange);
      return () => {
        unsubscribeHydrate();
        unsubscribeFinish();
      };
    },
    () => getPersistApi()?.hasHydrated() ?? true,
    () => false
  );
}
