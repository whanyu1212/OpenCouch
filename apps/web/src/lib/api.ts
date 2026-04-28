/**
 * API client for the OpenCouch backend.
 *
 * All endpoints are relative to the backend server running at
 * NEXT_PUBLIC_API_URL (default: http://localhost:8000/api).
 */

const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"
).replace(/\/$/, "");

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
export type VoiceMemoryMode = "persistent" | "incognito";

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
  response_style: string | null;
  response_style_source: string | null;
  therapeutic_approach: string | null;
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
  response_style: string | null;
}

export interface MemoryStatus {
  memory_mode: string;
  owner_id: string;
  counts: Record<string, number>;
  crisis_log_count: number;
  session_feedback_count: number;
  proactive_recall_enabled: boolean;
}

export interface MemoryRecallUpdateResponse {
  owner_id: string;
  proactive_recall_enabled: boolean;
  detail: string;
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

export interface RuntimeInfo {
  model: string;
  version: string;
}

export interface LiveKitVoiceTokenResponse {
  server_url: string;
  participant_token: string;
  room_name: string;
  identity: string;
  memory_mode: string;
}

export interface LiveKitVoiceFinalizationStatusResponse {
  status: "in_progress" | "completed" | "failed";
  detail: string | null;
  updated_at: string;
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

export interface ChatStreamOptions {
  message: string;
  threadId: string;
  userId?: string;
  responseModelTier?: ResponseModelTier;
  onEvent?: (event: StreamEvent) => void;
  onProtocolError?: (error: Error) => void;
}

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

export async function getInfo(): Promise<RuntimeInfo> {
  const res = await fetch(`${API_BASE}/info`);
  if (!res.ok) throw new Error(`Info failed: ${res.status}`);
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

export type SessionFeedbackLabel = "positive" | "negative" | "skip";

export async function endSession(
  threadId: string,
  feedback?: SessionFeedbackLabel
): Promise<EndSessionResponse> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback: feedback ?? null }),
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

export async function updateMemoryRecall(
  enabled: boolean,
  threadId: string,
  userId?: string
): Promise<MemoryRecallUpdateResponse> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/recall?${params}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(`Memory recall update failed: ${res.status}`);
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

function createWebSocketUrl(path: string): string {
  if (API_BASE.startsWith("http://")) {
    return `${API_BASE.replace(/^http:\/\//, "ws://")}${path}`;
  }
  if (API_BASE.startsWith("https://")) {
    return `${API_BASE.replace(/^https:\/\//, "wss://")}${path}`;
  }
  if (API_BASE.startsWith("//")) {
    const protocol =
      typeof window !== "undefined" && window.location.protocol === "http:"
        ? "ws:"
        : "wss:";
    return `${protocol}${API_BASE}${path}`;
  }

  if (typeof window === "undefined") {
    return `${API_BASE}${path}`;
  }

  const base = API_BASE.startsWith("/") ? API_BASE : `/${API_BASE}`;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${base}${path}`;
}

function parseStreamEvent(raw: unknown): StreamEvent {
  if (typeof raw !== "string") {
    throw new Error("Chat stream frame was not text.");
  }

  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || !("type" in parsed)) {
    throw new Error("Chat stream frame was not a valid event.");
  }

  const event = parsed as Partial<StreamEvent>;
  if (event.type !== "status" && event.type !== "chunk" && event.type !== "done") {
    throw new Error("Chat stream frame had an unknown event type.");
  }

  return parsed as StreamEvent;
}

export function createChatStream({
  message,
  threadId,
  userId,
  responseModelTier,
  onEvent,
  onProtocolError,
}: ChatStreamOptions): WebSocket {
  const ws = new WebSocket(createWebSocketUrl("/chat/stream"));

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
    try {
      onEvent?.(parseStreamEvent(event.data));
    } catch (error) {
      onProtocolError?.(
        error instanceof Error
          ? error
          : new Error("Could not parse chat stream frame.")
      );
      ws.close();
    }
  };

  return ws;
}

// ── LiveKit voice session helpers ───────────────────────────────────

export async function createLiveKitVoiceToken(
  userId: string,
  threadId: string,
  transcriptionLanguage: TranscriptionLanguageOption,
  memoryMode: VoiceMemoryMode
): Promise<LiveKitVoiceTokenResponse> {
  const res = await fetch(`${API_BASE}/voice/livekit/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      thread_id: threadId,
      transcription_language: transcriptionLanguage,
      memory_mode: memoryMode,
      dispatch_agent: true,
    }),
  });
  if (!res.ok) {
    throw new Error(`LiveKit token failed: ${res.status}`);
  }
  return res.json();
}

export async function getLiveKitVoiceFinalizationStatus(
  threadId: string
): Promise<LiveKitVoiceFinalizationStatusResponse | null> {
  const res = await fetch(
    `${API_BASE}/voice/livekit/finalization-status/${threadId}`,
    { cache: "no-store" }
  );
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    throw new Error(`LiveKit finalization status failed: ${res.status}`);
  }
  return res.json();
}
