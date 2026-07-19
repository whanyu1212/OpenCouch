"use client";

import { useSyncExternalStore } from "react";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  createChatStream,
  getMemoryFacts,
  type AssistantVoiceOption,
  type EndSessionResponse,
  type MemoryFact,
  type RealtimeVoiceSafetyResource,
  type RealtimeVoiceSafetyResourceStatus,
  type RealtimeVoiceSafetyResourcesResponse,
  type RealtimeVoiceSafetySupport,
  type ResponseModelTier,
  type SessionAction,
  type SessionFeedbackModality,
  type StreamEvent,
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
  therapeuticApproach?: string | null;
  sessionAction?: SessionAction | null;
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
  responseId?: string;
}

export interface VoiceSafetyOverlayState {
  clientTurnId: string;
  open: boolean;
  riskLevel: 2 | 3 | null;
  headline: string;
  validation: string;
  immediateStep: string;
  resourceStatus: "loading" | RealtimeVoiceSafetyResourceStatus;
  inferredLocation: string;
  resources: RealtimeVoiceSafetyResource[];
  message: string;
}

export type VoiceActivityName =
  | "memory_saved"
  | "memory_recall_updated"
  | "memory_delete_pending"
  | "memory_deleted"
  | "factual_lookup"
  | "crisis_resources_lookup"
  | "exercise"
  | "therapeutic_skill";

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
  blocksTextTurns: boolean;
  detail: string | null;
  updatedAt: string | null;
}

export interface VoiceSessionInfo {
  roomName: string;
  identity: string;
  memoryMode: string;
  assistantVoice: string;
  serverUrl: string;
  connectedAt: string;
}

export interface EndedSessionResult extends EndSessionResponse {
  threadId: string;
  modality: SessionFeedbackModality;
}

export function buildEndedSessionResult({
  threadId,
  result,
  modality,
}: {
  threadId: string;
  result: EndSessionResponse;
  modality: SessionFeedbackModality;
}): EndedSessionResult {
  return { threadId, modality, ...result };
}

type VoiceConnectionHandle = {
  disconnect: (options?: { finalize?: boolean }) => Promise<void> | void;
};

const IDLE_VOICE_FINALIZATION_STATE: VoiceFinalizationState = {
  threadId: null,
  status: "idle",
  blocksTextTurns: false,
  detail: null,
  updatedAt: null,
};

export function voiceFinalizationBlocksTextTurns(
  finalization: VoiceFinalizationState,
  threadId: string
): boolean {
  return (
    finalization.threadId === threadId &&
    finalization.blocksTextTurns &&
    (finalization.status === "in_progress" || finalization.status === "failed")
  );
}

const DEFAULT_VOICE_SAFETY_SUPPORT: RealtimeVoiceSafetySupport = {
  headline: "You deserve immediate support right now.",
  validation: "What you shared sounds serious, and you do not have to handle it alone.",
  immediate_step:
    "Move away from anything you could use to hurt yourself and contact emergency services or a trusted person nearby now.",
};

// Voice transport handles are non-reactive and kept outside Zustand state to
// avoid re-rendering subscribers on SDK or WebRTC object mutation.
let _voiceConnection: VoiceConnectionHandle | null = null;
let _chatSocket: WebSocket | null = null;
let _chatStreamId = 0;

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
  /** Whether at least one assistant chunk has arrived for the active text turn */
  chatStreamingStarted: boolean;
  /** Current streamed pipeline stages for the active text turn */
  chatStages: string[];
  /** User-facing chat notice/error message */
  chatNotice: string | null;

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
  voiceConnectionPending: boolean;
  voiceAgentSpeaking: boolean;
  voiceReadyToSpeak: boolean;
  assistantVoiceSelected: AssistantVoiceOption;
  voiceTranscripts: VoiceTranscript[];
  voiceActivities: VoiceActivityEvent[];
  voiceFinalization: VoiceFinalizationState;
  voiceSessionInfo: VoiceSessionInfo | null;
  voiceError: string | null;
  voiceSafetyOverlay: VoiceSafetyOverlayState | null;
  voiceSafetyResourceWorkActive: boolean;
  voiceSuppressedAssistantResponseIds: string[];
  voiceSuppressedAssistantItemIds: string[];

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
  setChatStreamingStarted: (started: boolean) => void;
  setChatStages: (stages: string[]) => void;
  setChatNotice: (notice: string | null) => void;
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
  setVoiceConnectionPending: (pending: boolean) => void;
  setVoiceAgentSpeaking: (speaking: boolean) => void;
  setVoiceReadyToSpeak: (ready: boolean) => void;
  setAssistantVoiceSelected: (voice: AssistantVoiceOption) => void;
  addVoiceTranscript: (t: VoiceTranscript) => void;
  addVoiceActivity: (event: VoiceActivityEvent) => void;
  clearVoiceActivities: () => void;
  setVoiceFinalization: (state: VoiceFinalizationState) => void;
  clearVoiceFinalization: () => void;
  setVoiceSessionInfo: (info: VoiceSessionInfo | null) => void;
  setVoiceError: (error: string | null) => void;
  setVoiceSafetyOverlay: (input: {
    clientTurnId: string;
    riskLevel: 2 | 3 | null;
    support: RealtimeVoiceSafetySupport | null;
  }) => void;
  updateVoiceSafetyResources: (
    clientTurnId: string,
    response: RealtimeVoiceSafetyResourcesResponse
  ) => void;
  dismissVoiceSafetyOverlay: () => void;
  setVoiceSafetyResourceWorkActive: (active: boolean) => void;
  suppressVoiceAssistantTranscripts: (input: {
    responseIds: string[];
    itemIds: string[];
  }) => void;
  /** Store the voice connection handle outside reactive Zustand state. */
  voiceSetRefs: (refs: { connection?: VoiceConnectionHandle | null }) => void;
  /** Disconnect active voice resources and mark memory finalization pending. */
  voiceDisconnect: () => void;
  /** Clear transcripts (e.g. on fresh connect) */
  clearVoiceTranscripts: () => void;
}

type ClearedVoiceSessionUiState = Pick<
  SessionState,
  | "voiceConnected"
  | "voiceConnectionPending"
  | "voiceAgentSpeaking"
  | "voiceReadyToSpeak"
  | "voiceTranscripts"
  | "voiceActivities"
  | "voiceFinalization"
  | "voiceSessionInfo"
  | "voiceError"
  | "voiceSafetyOverlay"
  | "voiceSafetyResourceWorkActive"
  | "voiceSuppressedAssistantResponseIds"
  | "voiceSuppressedAssistantItemIds"
>;

function clearedVoiceSessionUiState(): ClearedVoiceSessionUiState {
  return {
    voiceConnected: false,
    voiceConnectionPending: false,
    voiceAgentSpeaking: false,
    voiceReadyToSpeak: false,
    voiceTranscripts: [],
    voiceActivities: [],
    voiceFinalization: IDLE_VOICE_FINALIZATION_STATE,
    voiceSessionInfo: null,
    voiceError: null,
    voiceSafetyOverlay: null,
    voiceSafetyResourceWorkActive: false,
    voiceSuppressedAssistantResponseIds: [],
    voiceSuppressedAssistantItemIds: [],
  };
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
  | "assistantVoiceSelected"
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
  chatStreamingStarted: false,
  chatStages: [],
  chatNotice: null,
  memoryFacts: [],
  memoryPanelOpen: false,
  memoryUnseenCount: 0,
  memoryRefreshVersion: 0,
  lastEndedSession: null,
  responseModelTier: "fast",

  // Voice defaults (reactive only)
  voiceConnected: false,
  voiceConnectionPending: false,
  voiceAgentSpeaking: false,
  voiceReadyToSpeak: false,
  assistantVoiceSelected: "marin",
  voiceTranscripts: [],
  voiceActivities: [],
  voiceFinalization: IDLE_VOICE_FINALIZATION_STATE,
  voiceSessionInfo: null,
  voiceError: null,
  voiceSafetyOverlay: null,
  voiceSafetyResourceWorkActive: false,
  voiceSuppressedAssistantResponseIds: [],
  voiceSuppressedAssistantItemIds: [],

  setUserId: (id: string) => set({ userId: id }),
  setThreadId: (id: string) => {
    cancelActiveChatStream({ resetLoading: true });
    set({
      threadId: id,
      messages: [],
      chatStages: [],
      chatNotice: null,
      chatStreamingStarted: false,
      memoryFacts: [],
      memoryUnseenCount: 0,
      lastEndedSession: null,
      ...clearedVoiceSessionUiState(),
    });
  },
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
  setChatStreamingStarted: (chatStreamingStarted: boolean) =>
    set({ chatStreamingStarted }),
  setChatStages: (chatStages: string[]) => set({ chatStages }),
  setChatNotice: (chatNotice: string | null) => set({ chatNotice }),

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

  startSession: (mode: SessionMode, userId: string, threadId: string) => {
    cancelActiveChatStream({ resetLoading: true });
    set({
      isSetup: true,
      sessionMode: mode,
      userId: mode === "incognito" ? "" : (userId.trim() || "web-user"),
      threadId:
        mode === "incognito"
          ? generateThreadId()
          : (threadId.trim() || generateThreadId()),
      messages: [],
      chatLoading: false,
      chatStreamingStarted: false,
      chatStages: [],
      chatNotice: null,
      memoryFacts: [],
      memoryUnseenCount: 0,
      lastEndedSession: null,
      ...clearedVoiceSessionUiState(),
    });
  },
  newSession: () => {
    cancelActiveChatStream({ resetLoading: true });
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
      chatStreamingStarted: false,
      chatStages: [],
      chatNotice: null,
      memoryFacts: [],
      memoryUnseenCount: 0,
      lastEndedSession: null,
      ...clearedVoiceSessionUiState(),
    });
  },

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected) => set({ voiceConnected: connected }),
  setVoiceConnectionPending: (voiceConnectionPending) =>
    set({ voiceConnectionPending }),
  setVoiceAgentSpeaking: (speaking) => set({ voiceAgentSpeaking: speaking }),
  setVoiceReadyToSpeak: (ready) => set({ voiceReadyToSpeak: ready }),
  setAssistantVoiceSelected: (voice) => set({ assistantVoiceSelected: voice }),
  addVoiceTranscript: (t) =>
    set((state) => {
      if (
        t.role === "assistant" &&
        ((t.responseId &&
          state.voiceSuppressedAssistantResponseIds.includes(t.responseId)) ||
          (t.itemId && state.voiceSuppressedAssistantItemIds.includes(t.itemId)))
      ) {
        return state;
      }
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
  setVoiceSafetyOverlay: ({ clientTurnId, riskLevel, support }) => {
    const copy = support ?? DEFAULT_VOICE_SAFETY_SUPPORT;
    set({
      voiceSafetyOverlay: {
        clientTurnId,
        open: true,
        riskLevel,
        headline: copy.headline,
        validation: copy.validation,
        immediateStep: copy.immediate_step,
        resourceStatus: "loading",
        inferredLocation: "",
        resources: [],
        message: "Looking for verified support resources...",
      },
    });
  },
  updateVoiceSafetyResources: (clientTurnId, response) =>
    set((state) => {
      const overlay = state.voiceSafetyOverlay;
      if (
        !overlay?.open ||
        overlay.clientTurnId !== clientTurnId ||
        response.client_turn_id !== clientTurnId
      ) {
        return state;
      }
      return {
        voiceSafetyOverlay: {
          ...overlay,
          resourceStatus: response.status,
          inferredLocation: response.inferred_location,
          resources: response.resources,
          message: response.message,
        },
      };
    }),
  dismissVoiceSafetyOverlay: () => set({ voiceSafetyOverlay: null }),
  setVoiceSafetyResourceWorkActive: (voiceSafetyResourceWorkActive) =>
    set({ voiceSafetyResourceWorkActive }),
  suppressVoiceAssistantTranscripts: ({ responseIds, itemIds }) =>
    set((state) => {
      const suppressedResponseIds = new Set([
        ...state.voiceSuppressedAssistantResponseIds,
        ...responseIds,
      ]);
      const suppressedItemIds = new Set([
        ...state.voiceSuppressedAssistantItemIds,
        ...itemIds,
      ]);
      return {
        voiceTranscripts: state.voiceTranscripts.filter(
          (transcript) =>
            transcript.role !== "assistant" ||
            !(
              (transcript.responseId &&
                suppressedResponseIds.has(transcript.responseId)) ||
              (transcript.itemId && suppressedItemIds.has(transcript.itemId))
            )
        ),
        voiceSuppressedAssistantResponseIds: [...suppressedResponseIds],
        voiceSuppressedAssistantItemIds: [...suppressedItemIds],
      };
    }),
  clearVoiceTranscripts: () =>
    set({
      voiceTranscripts: [],
      voiceSuppressedAssistantResponseIds: [],
      voiceSuppressedAssistantItemIds: [],
    }),

  voiceSetRefs: (refs) => {
    if (refs.connection !== undefined) _voiceConnection = refs.connection;
  },

  voiceDisconnect: () => {
    const shouldTrackFinalization =
      get().voiceConnected || _voiceConnection !== null;
    const threadId = get().threadId;

    void _voiceConnection?.disconnect({ finalize: true });
    _voiceConnection = null;
    set({
      voiceConnected: false,
      voiceConnectionPending: false,
      voiceAgentSpeaking: false,
      voiceReadyToSpeak: false,
      voiceFinalization:
        shouldTrackFinalization && threadId
          ? {
              threadId,
              status: "in_progress",
              blocksTextTurns: false,
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
        assistantVoiceSelected: state.assistantVoiceSelected,
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
          chatStreamingStarted: false,
          chatStages: [],
          chatNotice: null,
          memoryFacts: [],
          memoryPanelOpen: false,
          memoryUnseenCount: 0,
          memoryRefreshVersion: 0,
          lastEndedSession: null,
          voiceConnected: false,
          voiceConnectionPending: false,
          voiceAgentSpeaking: false,
          voiceReadyToSpeak: false,
          voiceTranscripts: [],
          voiceActivities: [],
          voiceFinalization: IDLE_VOICE_FINALIZATION_STATE,
          voiceSessionInfo: null,
           voiceError: null,
           voiceSafetyOverlay: null,
           voiceSafetyResourceWorkActive: false,
           voiceSuppressedAssistantResponseIds: [],
           voiceSuppressedAssistantItemIds: [],
         };
      },
    }
  )
);

interface CancelActiveChatStreamOptions {
  resetLoading?: boolean;
}

export function cancelActiveChatStream({
  resetLoading = false,
}: CancelActiveChatStreamOptions = {}): void {
  _chatStreamId += 1;
  _chatSocket?.close(1000, "client_cancelled");
  _chatSocket = null;

  if (resetLoading) {
    useSessionStore.setState({
      chatLoading: false,
      chatStreamingStarted: false,
      chatStages: [],
    });
  }
}

interface StartTextChatStreamOptions {
  message: string;
  threadId: string;
  userId: string;
  sessionMode: SessionMode;
  responseModelTier: ResponseModelTier;
}

export function startTextChatStream({
  message,
  threadId,
  userId,
  sessionMode,
  responseModelTier,
}: StartTextChatStreamOptions): boolean {
  const msg = message.trim();
  if (!msg || useSessionStore.getState().chatLoading) {
    return false;
  }

  cancelActiveChatStream();
  const streamId = _chatStreamId + 1;
  _chatStreamId = streamId;
  const isCurrentStream = () => _chatStreamId === streamId;

  let done = false;
  let streamingStarted = false;

  useSessionStore.setState((state) => ({
    lastEndedSession: null,
    chatNotice: null,
    chatStages: [],
    chatLoading: true,
    chatStreamingStarted: false,
    messages: [...state.messages, { role: "user", content: msg }],
  }));

  const ws = createChatStream({
    message: msg,
    threadId,
    userId,
    memoryMode: sessionMode,
    responseModelTier,
    onEvent: (event: StreamEvent) => {
      if (!isCurrentStream()) return;
      if (event.type === "status") {
        useSessionStore.setState((state) => ({
          chatStages: [...state.chatStages, event.stage],
        }));
        return;
      }

      if (event.type === "chunk") {
        if (!streamingStarted) {
          streamingStarted = true;
          useSessionStore.setState((state) => ({
            chatStreamingStarted: true,
            chatStages: [],
            messages: [
              ...state.messages,
              { role: "assistant", content: event.text },
            ],
          }));
        } else {
          useSessionStore.getState().appendToLastMessage(event.text);
        }
        return;
      }

      if (event.type === "error") {
        done = true;
        useSessionStore.setState({
          chatLoading: false,
          chatStreamingStarted: false,
          chatStages: [],
          chatNotice: event.message,
        });
        ws.close();
        return;
      }

      if (event.type !== "done") return;

      done = true;
      const resp = event.response;
      if (streamingStarted) {
        useSessionStore.getState().updateLastMessage({
          content: resp.response_text,
          responseStyle: resp.response_style,
          therapeuticApproach: resp.therapeutic_approach,
          sessionAction: resp.session_action,
          responseType: resp.response_type,
          crisis: resp.crisis,
          diagnostics: resp.diagnostics,
        });
      } else {
        useSessionStore.getState().addMessage({
          role: "assistant",
          content: resp.response_text,
          responseStyle: resp.response_style,
          therapeuticApproach: resp.therapeutic_approach,
          sessionAction: resp.session_action,
          responseType: resp.response_type,
          crisis: resp.crisis,
          diagnostics: resp.diagnostics,
        });
      }

      useSessionStore.setState({
        chatLoading: false,
        chatStreamingStarted: false,
        chatStages: [],
      });

      const semanticWrites = Number(resp.diagnostics?.semantic_writes ?? 0);
      const proceduralWrites = Number(resp.diagnostics?.procedural_writes ?? 0);
      const memoryControlTurn =
        resp.response_style === "memory_control" ||
        resp.diagnostics?.memory_control_ms != null;
      if (
        sessionMode === "persistent" &&
        (semanticWrites > 0 || proceduralWrites > 0 || memoryControlTurn)
      ) {
        useSessionStore.getState().bumpMemoryRefreshVersion();
      }
      if (
        sessionMode === "persistent" &&
        (semanticWrites > 0 || memoryControlTurn)
      ) {
        void getMemoryFacts(threadId, userId || undefined, sessionMode)
          .then((facts) => {
            if (!isCurrentStream()) return;
            useSessionStore.getState().setMemoryFacts(facts);
            const currentFactCount = useSessionStore.getState().memoryFacts.length;
            useSessionStore
              .getState()
              .addUnseenMemories(Math.max(0, facts.length - currentFactCount));
          })
          .catch(() => {
            if (!isCurrentStream()) return;
            useSessionStore
              .getState()
              .setChatNotice("Reply completed, but memory refresh failed.");
          });
      }

      ws.close();
    },
    onProtocolError: () => {
      if (!isCurrentStream()) return;
      done = true;
      useSessionStore.setState({
        chatStages: [],
        chatLoading: false,
        chatStreamingStarted: false,
        chatNotice:
          "The chat stream sent an unreadable response. Please try again.",
      });
    },
  });

  _chatSocket = ws;

  ws.onerror = () => {
    if (!isCurrentStream() || done) return;
    done = true;
    useSessionStore.setState({
      chatStages: [],
      chatLoading: false,
      chatStreamingStarted: false,
      chatNotice:
        "Connection error. Check that the backend is running on the configured API URL.",
    });
  };

  ws.onclose = () => {
    if (!isCurrentStream()) return;
    _chatSocket = null;
    if (!done && useSessionStore.getState().chatLoading) {
      useSessionStore.setState({
        chatStages: [],
        chatLoading: false,
        chatStreamingStarted: false,
        chatNotice: "The chat connection closed before the reply finished.",
      });
    }
  };

  return true;
}

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
