"use client";

import { create } from "zustand";
import type {
  EndSessionResponse,
  MemoryFact,
  ResponseModelTier,
  RealtimeVoiceOption,
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
  itemId?: string;
}

export interface EndedSessionResult extends EndSessionResponse {
  threadId: string;
}

// ── Module-level voice refs (non-reactive, never trigger re-renders) ──
// These are singleton — only one voice session exists at a time.
let _voiceWs: WebSocket | null = null;
let _voiceAudioCtx: AudioContext | null = null;
let _voiceMediaStream: MediaStream | null = null;
let _voiceProcessor: ScriptProcessorNode | null = null;
let _voiceGainNode: GainNode | null = null;
let _voiceNextPlayTime = 0;
let _voiceGeneration = 0;
let _voicePlaybackEpoch = 0;

// Active AudioBufferSourceNodes for flush-on-interrupt
let _voiceActiveSources: AudioBufferSourceNode[] = [];

// Per-item playback tracking for truncation reporting
let _voiceCurrentItemId: string | null = null;
let _voiceCurrentContentIndex = 0;
let _voiceItemPlaybackStartTime = 0; // AudioContext.currentTime when first chunk started
let _voiceItemScheduledEndTime = 0;  // last scheduled end time for current item
let _voiceLocalDuckActive = false;
let _voiceLocalDuckUntilMs = 0;

/** Read-only access to voice refs from outside the store */
export function getVoiceRefs() {
  return {
    ws: _voiceWs,
    audioCtx: _voiceAudioCtx,
    mediaStream: _voiceMediaStream,
    processor: _voiceProcessor,
    gainNode: _voiceGainNode,
    nextPlayTime: _voiceNextPlayTime,
    generation: _voiceGeneration,
    playbackEpoch: _voicePlaybackEpoch,
    currentItemId: _voiceCurrentItemId,
    currentContentIndex: _voiceCurrentContentIndex,
    itemPlaybackStartTime: _voiceItemPlaybackStartTime,
    itemScheduledEndTime: _voiceItemScheduledEndTime,
  };
}

export function setVoiceNextPlayTime(t: number) {
  _voiceNextPlayTime = t;
}

/** Set the current playback item (called on first audio chunk of a new item) */
export function setVoiceCurrentItem(
  itemId: string,
  contentIndex: number,
  startTime: number
) {
  _voiceCurrentItemId = itemId;
  _voiceCurrentContentIndex = contentIndex;
  _voiceItemPlaybackStartTime = startTime;
  _voiceItemScheduledEndTime = startTime;
}

/** Update the scheduled end time as new chunks are queued */
export function setVoiceItemScheduledEnd(t: number) {
  _voiceItemScheduledEndTime = t;
}

/** Register an active source for interrupt flushing */
export function addVoiceActiveSource(src: AudioBufferSourceNode) {
  _voiceActiveSources.push(src);
}

/** Remove a source that ended naturally (prevents leak during long sessions) */
export function removeVoiceActiveSource(src: AudioBufferSourceNode) {
  const idx = _voiceActiveSources.indexOf(src);
  if (idx !== -1) _voiceActiveSources.splice(idx, 1);
}

/** Ramp the assistant playback gain to a target value. */
export function setVoiceGain(target: number, rampMs = 12) {
  const ctx = _voiceAudioCtx;
  const gain = _voiceGainNode;
  if (!ctx || !gain) return;

  const now = ctx.currentTime;
  gain.gain.cancelScheduledValues(now);
  gain.gain.setValueAtTime(gain.gain.value, now);
  gain.gain.linearRampToValueAtTime(target, now + (rampMs / 1000));
}

/** Immediately duck assistant playback based on local mic activity. */
export function activateVoiceLocalDuck(targetGain: number, holdMs: number) {
  _voiceLocalDuckUntilMs = performance.now() + holdMs;
  if (_voiceLocalDuckActive) return;

  _voiceLocalDuckActive = true;
  setVoiceGain(targetGain, 10);
}

/** Restore playback gain after local ducking if the hold window elapsed. */
export function releaseVoiceLocalDuckIfReady(nowMs: number) {
  if (!_voiceLocalDuckActive || nowMs < _voiceLocalDuckUntilMs) return false;

  _voiceLocalDuckActive = false;
  setVoiceGain(1, 20);
  return true;
}

/** Reset local ducking state without touching playback scheduling. */
export function clearVoiceLocalDuck() {
  _voiceLocalDuckActive = false;
  _voiceLocalDuckUntilMs = 0;
}

/** Flush all queued audio — mute gain, stop sources, reset timing */
export function flushVoicePlayback() {
  const ctx = _voiceAudioCtx;
  const gain = _voiceGainNode;
  clearVoiceLocalDuck();

  // Mute with a tiny ramp to avoid click
  if (gain && ctx) {
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.005);
  }

  // Stop all scheduled sources
  for (const src of _voiceActiveSources) {
    try {
      src.stop(0);
      src.disconnect();
    } catch {
      // Already stopped — ignore
    }
  }
  _voiceActiveSources = [];

  // Restore gain for next response
  if (gain && ctx) {
    gain.gain.setValueAtTime(1, ctx.currentTime + 0.01);
  }

  // Reset playback timing
  if (ctx) {
    _voiceNextPlayTime = ctx.currentTime;
  } else {
    _voiceNextPlayTime = 0;
  }

  // Bump playback epoch so stale onended callbacks are ignored
  _voicePlaybackEpoch += 1;

  // Clear current item tracking
  _voiceCurrentItemId = null;
  _voiceCurrentContentIndex = 0;
  _voiceItemPlaybackStartTime = 0;
  _voiceItemScheduledEndTime = 0;
}

/** Compute how many ms of the current item were actually played */
export function computePlayedMs(): number {
  const ctx = _voiceAudioCtx;
  if (!ctx || !_voiceItemPlaybackStartTime) return 0;
  const played = ctx.currentTime - _voiceItemPlaybackStartTime;
  const scheduled = _voiceItemScheduledEndTime - _voiceItemPlaybackStartTime;
  const clamped = Math.max(0, Math.min(played, scheduled));
  return Math.floor(clamped * 1000);
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
  /** Bumped when memory surfaces should re-fetch from the backend */
  memoryRefreshVersion: number;
  /** Most recent explicit session-end result for the current thread */
  lastEndedSession: EndedSessionResult | null;
  /** User-facing text response preference */
  responseModelTier: ResponseModelTier;

  // ── Voice session (reactive UI state only) ────────────────────────
  voiceConnected: boolean;
  voiceAgentSpeaking: boolean;
  voiceSelected: RealtimeVoiceOption;
  transcriptionLanguageSelected: TranscriptionLanguageOption;
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
  bumpMemoryRefreshVersion: () => void;
  setLastEndedSession: (session: EndedSessionResult | null) => void;
  clearLastEndedSession: () => void;
  setResponseModelTier: (tier: ResponseModelTier) => void;

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected: boolean) => void;
  setVoiceAgentSpeaking: (speaking: boolean) => void;
  setVoiceSelected: (voice: RealtimeVoiceOption) => void;
  setTranscriptionLanguageSelected: (language: TranscriptionLanguageOption) => void;
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
    gainNode?: GainNode | null;
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
  memoryRefreshVersion: 0,
  lastEndedSession: null,
  responseModelTier: "fast",

  // Voice defaults (reactive only)
  voiceConnected: false,
  voiceAgentSpeaking: false,
  voiceSelected: "cedar",
  transcriptionLanguageSelected: "en",
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
      lastEndedSession: null,
    });
  },

  // ── Voice actions ─────────────────────────────────────────────────
  setVoiceConnected: (connected) => set({ voiceConnected: connected }),
  setVoiceAgentSpeaking: (speaking) => set({ voiceAgentSpeaking: speaking }),
  setVoiceSelected: (voice) => set({ voiceSelected: voice }),
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
    if (refs.gainNode !== undefined) _voiceGainNode = refs.gainNode;
  },

  voiceCleanup: (generation: number) => {
    // Only clean up if this generation is still the active one
    if (generation !== _voiceGeneration) return;
    flushVoicePlayback();
    _voiceProcessor?.disconnect();
    _voiceGainNode?.disconnect();
    _voiceMediaStream?.getTracks().forEach((t) => t.stop());
    _voiceAudioCtx?.close();
    _voiceProcessor = null;
    _voiceGainNode = null;
    _voiceMediaStream = null;
    _voiceAudioCtx = null;
    _voiceNextPlayTime = 0;
  },

  voiceDisconnect: () => {
    // Force-bump generation so any pending onclose becomes a no-op
    _voiceGeneration += 1;
    flushVoicePlayback();
    _voiceWs?.close();
    _voiceProcessor?.disconnect();
    _voiceGainNode?.disconnect();
    _voiceMediaStream?.getTracks().forEach((t) => t.stop());
    _voiceAudioCtx?.close();
    _voiceWs = null;
    _voiceProcessor = null;
    _voiceGainNode = null;
    _voiceMediaStream = null;
    _voiceAudioCtx = null;
    _voiceNextPlayTime = 0;
    set({
      voiceConnected: false,
      voiceAgentSpeaking: false,
    });
  },
}));
