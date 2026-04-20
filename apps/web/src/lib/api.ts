/**
 * API client for the OpenCouch backend.
 *
 * All endpoints are relative to the backend server running at
 * NEXT_PUBLIC_API_URL (default: http://localhost:8000/api).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const WS_BASE = API_BASE.replace(/^http/, "ws");

export const REALTIME_VOICE_OPTIONS = [
  "alloy",
  "ash",
  "ballad",
  "coral",
  "echo",
  "sage",
  "shimmer",
  "verse",
  "marin",
  "cedar",
] as const;

export type RealtimeVoiceOption = (typeof REALTIME_VOICE_OPTIONS)[number];

export const TRANSCRIPTION_LANGUAGE_OPTIONS = [
  { value: "en", label: "english" },
  { value: "", label: "auto detect" },
  { value: "es", label: "spanish" },
  { value: "fr", label: "french" },
  { value: "de", label: "german" },
  { value: "it", label: "italian" },
  { value: "pt", label: "portuguese" },
  { value: "ja", label: "japanese" },
  { value: "ko", label: "korean" },
  { value: "zh", label: "chinese" },
] as const;

export type TranscriptionLanguageOption =
  (typeof TRANSCRIPTION_LANGUAGE_OPTIONS)[number]["value"];

// ── Types ────────────────────────────────────────────────────────────

export interface CrisisInfo {
  level: number;
  confidence: string;
  reason: string;
  needs_crisis_response: boolean;
  needs_clarification: boolean;
}

export interface ChatResponse {
  response_text: string;
  response_type: string;
  mode: string | null;
  mode_source: string | null;
  modality: string | null;
  crisis: CrisisInfo;
  diagnostics: Record<string, unknown>;
}

export interface ThreadSummary {
  thread_id: string;
  turn_count: number;
  message_count: number;
  has_context: boolean;
}

export interface ThreadSessionStatus {
  has_active_session: boolean;
}

export type ResponseModelTier = "fast" | "quality";

export interface Message {
  role: "user" | "assistant";
  content: string;
  mode: string | null;
}

export interface MemoryStatus {
  memory_mode: string;
  owner_id: string;
  counts: Record<string, number>;
  crisis_log_count: number;
  session_feedback_count: number;
  proactive_recall_enabled: boolean;
}

export interface MemoryFact {
  index: number;
  key: string;
  category: string;
  predicate: string;
  subject: string;
  object: string;
  evidence_quote: string;
  confidence: string;
  created_at: string;
}

export interface MemorySession {
  index: number;
  key: string;
  summary: string;
  themes: string[];
  mood_opened: string;
  mood_closed: string;
  turn_count: number;
  ended_at: string;
}

export interface MemoryRule {
  index: number;
  rule: string;
  evidence: string[];
  confidence: string;
  added_at?: string;
}

export interface EndSessionResponse {
  summary: string | null;
  detail?: string;
  themes?: string[];
  mood_opened?: string;
  mood_closed?: string;
  turn_count?: number;
  open_loops?: string[];
  resolved_threads?: string[];
}

// ── Stream event types ───────────────────────────────────────────────

export interface StreamStatusEvent {
  type: "status";
  stage: string;
  detail: string;
}

export interface StreamChunkEvent {
  type: "chunk";
  text: string;
}

export interface StreamDoneEvent {
  type: "done";
  response: ChatResponse;
}

export type StreamEvent = StreamStatusEvent | StreamChunkEvent | StreamDoneEvent;

// ── REST helpers ─────────────────────────────────────────────────────

export async function postChat(
  message: string,
  threadId: string,
  userId?: string,
  responseModelTier?: ResponseModelTier
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId,
      user_id: userId || undefined,
      response_model_tier: responseModelTier || undefined,
    }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

export async function getThreads(limit = 20): Promise<ThreadSummary[]> {
  const res = await fetch(`${API_BASE}/threads?limit=${limit}`);
  if (!res.ok) throw new Error(`Threads failed: ${res.status}`);
  return res.json();
}

export async function getHistory(threadId: string): Promise<Message[]> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/history`);
  if (!res.ok) throw new Error(`History failed: ${res.status}`);
  return res.json();
}

export async function getThreadSessionStatus(
  threadId: string
): Promise<ThreadSessionStatus> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/session-status`);
  if (!res.ok) throw new Error(`Session status failed: ${res.status}`);
  return res.json();
}

export async function endSession(threadId: string): Promise<EndSessionResponse> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/end`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`End session failed: ${res.status}`);
  return res.json();
}

export async function getMemoryStatus(
  threadId: string,
  userId?: string
): Promise<MemoryStatus> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/status?${params}`);
  if (!res.ok) throw new Error(`Memory status failed: ${res.status}`);
  return res.json();
}

export async function getMemoryFacts(
  threadId: string,
  userId?: string
): Promise<MemoryFact[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/facts?${params}`);
  if (!res.ok) throw new Error(`Memory facts failed: ${res.status}`);
  return res.json();
}

export async function getMemorySessions(
  threadId: string,
  userId?: string
): Promise<MemorySession[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/sessions?${params}`);
  if (!res.ok) throw new Error(`Memory sessions failed: ${res.status}`);
  return res.json();
}

export async function getMemoryRules(
  threadId: string,
  userId?: string
): Promise<MemoryRule[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/rules?${params}`);
  if (!res.ok) throw new Error(`Memory rules failed: ${res.status}`);
  return res.json();
}

// ── Memory deletion ───────────────────────────────────────────────────

export interface DeleteMemoryResponse {
  deleted: boolean;
  detail: string;
}

export async function deleteMemoryFact(
  index: number,
  threadId: string,
  userId?: string
): Promise<DeleteMemoryResponse> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/facts/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete fact failed: ${res.status}`);
  return res.json();
}

export async function deleteMemorySession(
  index: number,
  threadId: string,
  userId?: string
): Promise<DeleteMemoryResponse> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/sessions/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete session failed: ${res.status}`);
  return res.json();
}

export async function deleteMemoryRule(
  index: number,
  threadId: string,
  userId?: string
): Promise<DeleteMemoryResponse> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/rules/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete rule failed: ${res.status}`);
  return res.json();
}

// ── Thread state (raw agent state dict) ─────────────────────────────

export async function getThreadState(
  threadId: string
): Promise<Record<string, unknown> | null> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/state`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Thread state failed: ${res.status}`);
  return res.json();
}

// ── WebSocket stream for text chat ───────────────────────────────────

export function createChatStream(
  message: string,
  threadId: string,
  userId?: string,
  responseModelTier?: ResponseModelTier,
  onEvent?: (event: StreamEvent) => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/chat/stream`);

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        message,
        thread_id: threadId,
        user_id: userId || undefined,
        response_model_tier: responseModelTier || undefined,
      })
    );
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onEvent?.(data);
  };

  return ws;
}

// ── WebSocket for voice ──────────────────────────────────────────────

export function createVoiceSession(
  userId: string,
  threadId: string,
  voice: RealtimeVoiceOption,
  transcriptionLanguage: TranscriptionLanguageOption,
  callbacks: {
    onReady?: () => void;
    onAudio?: (audioBytes: Uint8Array, itemId: string, contentIndex: number) => void;
    onCaption?: (
      role: "user" | "assistant",
      text: string,
      itemId: string,
      status: "partial" | "final" | "cleared"
    ) => void;
    onTranscript?: (role: "user" | "assistant", text: string, itemId: string) => void;
    onInterrupted?: () => void;
    onError?: (message: string) => void;
  }
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/voice/session`);

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "start",
        user_id: userId,
        thread_id: threadId,
        voice,
        transcription_language: transcriptionLanguage,
      })
    );
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "ready" && callbacks.onReady) {
      callbacks.onReady();
    } else if (data.type === "audio" && callbacks.onAudio) {
      const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
      callbacks.onAudio(bytes, data.item_id || "", data.content_index ?? 0);
    } else if (data.type === "caption" && callbacks.onCaption) {
      callbacks.onCaption(
        data.role,
        data.text || "",
        data.item_id || "",
        data.status || "partial"
      );
    } else if (data.type === "transcript" && callbacks.onTranscript) {
      callbacks.onTranscript(data.role, data.text, data.item_id || "");
    } else if (data.type === "interrupted" && callbacks.onInterrupted) {
      callbacks.onInterrupted();
    } else if (data.type === "error" && callbacks.onError) {
      callbacks.onError(data.message);
    }
  };

  return ws;
}

/** Send a truncation report back to the server */
export function sendVoiceTruncate(
  ws: WebSocket,
  itemId: string,
  contentIndex: number,
  audioEndMs: number
): void {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(
    JSON.stringify({
      type: "truncate",
      item_id: itemId,
      content_index: contentIndex,
      audio_end_ms: audioEndMs,
    })
  );
}
