/**
 * API client for the OpenCouch backend.
 *
 * All endpoints are relative to the backend server running at
 * NEXT_PUBLIC_API_URL (default: http://localhost:8000/api).
 */

const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"
).replace(/\/$/, "");

export const ASSISTANT_VOICE_OPTIONS = [
  { value: "marin", label: "marin" },
  { value: "cedar", label: "cedar" },
  { value: "sage", label: "sage" },
  { value: "verse", label: "verse" },
  { value: "alloy", label: "alloy" },
  { value: "ash", label: "ash" },
  { value: "ballad", label: "ballad" },
  { value: "coral", label: "coral" },
  { value: "echo", label: "echo" },
  { value: "shimmer", label: "shimmer" },
] as const;

export type AssistantVoiceOption =
  (typeof ASSISTANT_VOICE_OPTIONS)[number]["value"];

export type ApiMemoryMode = "persistent" | "incognito";
export type VoiceMemoryMode = ApiMemoryMode;
export type SessionAction = "none" | "suggest_end_session";

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
  therapeutic_approach: string | null;
  session_action: SessionAction;
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
  session_id: string;
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

export interface RealtimeVoiceSessionResponse {
  client_secret: string;
  thread_id: string;
  user_id: string | null;
  memory_mode: VoiceMemoryMode;
  session_config: Record<string, unknown>;
}

export interface RealtimeVoiceToolCallResponse {
  output: Record<string, unknown>;
}

export interface RealtimeVoiceTurnRecordResponse {
  recorded: boolean;
  thread_id: string;
  message_count: number;
}

export interface RealtimeVoiceRecordedToolCall {
  tool_name: string;
  status: "started" | "completed" | "failed";
  output?: Record<string, unknown>;
  error?: string;
}

export interface RealtimeVoiceEndSessionResponse {
  finalized: boolean;
  summary: string | null;
  detail: string;
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
  memoryMode?: ApiMemoryMode;
  responseModelTier?: ResponseModelTier;
  onEvent?: (event: StreamEvent) => void;
  onProtocolError?: (error: Error) => void;
}

// ── REST helpers ─────────────────────────────────────────────────────

function setOptionalParam(
  params: URLSearchParams,
  key: string,
  value?: string | number
): void {
  if (value !== undefined && value !== null && String(value) !== "") {
    params.set(key, String(value));
  }
}

function querySuffix(params: URLSearchParams): string {
  const query = params.toString();
  return query ? `?${query}` : "";
}

function threadScopedParams({
  threadId,
  userId,
  memoryMode,
}: {
  threadId: string;
  userId?: string;
  memoryMode?: ApiMemoryMode;
}): URLSearchParams {
  const params = new URLSearchParams({ thread_id: threadId });
  setOptionalParam(params, "user_id", userId);
  setOptionalParam(params, "memory_mode", memoryMode);
  return params;
}

function memoryModePayload(memoryMode?: ApiMemoryMode): {
  memory_mode?: ApiMemoryMode;
} {
  return memoryMode ? { memory_mode: memoryMode } : {};
}

export async function postChat(
  message: string,
  threadId: string,
  userId?: string,
  responseModelTier?: ResponseModelTier,
  memoryMode?: ApiMemoryMode
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId,
      user_id: userId || undefined,
      response_model_tier: responseModelTier || undefined,
      ...memoryModePayload(memoryMode),
    }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

export async function getThreads(
  limit = 20,
  memoryMode?: ApiMemoryMode
): Promise<ThreadSummary[]> {
  const params = new URLSearchParams();
  setOptionalParam(params, "limit", limit);
  setOptionalParam(params, "memory_mode", memoryMode);
  const res = await fetch(`${API_BASE}/threads${querySuffix(params)}`);
  if (!res.ok) throw new Error(`Threads failed: ${res.status}`);
  return res.json();
}

export async function getHistory(
  threadId: string,
  memoryMode?: ApiMemoryMode
): Promise<Message[]> {
  const params = new URLSearchParams();
  setOptionalParam(params, "memory_mode", memoryMode);
  const res = await fetch(
    `${API_BASE}/threads/${threadId}/history${querySuffix(params)}`
  );
  if (!res.ok) throw new Error(`History failed: ${res.status}`);
  return res.json();
}

export async function getThreadSessionStatus(
  threadId: string,
  memoryMode?: ApiMemoryMode
): Promise<ThreadSessionStatus> {
  const params = new URLSearchParams();
  setOptionalParam(params, "memory_mode", memoryMode);
  const res = await fetch(
    `${API_BASE}/threads/${threadId}/session-status${querySuffix(params)}`
  );
  if (!res.ok) throw new Error(`Session status failed: ${res.status}`);
  return res.json();
}

export type SessionFeedbackLabel = "positive" | "negative" | "skip";
export type SessionFeedbackModality = "text" | "voice";

export interface SessionFeedbackResponse {
  recorded: boolean;
}

export async function submitSessionFeedback(
  threadId: string,
  feedback: SessionFeedbackLabel,
  memoryMode?: ApiMemoryMode,
  modality: SessionFeedbackModality = "text"
): Promise<SessionFeedbackResponse> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feedback,
      ...memoryModePayload(memoryMode),
      modality,
    }),
  });
  if (!res.ok) throw new Error(`Session feedback failed: ${res.status}`);
  return res.json();
}

export async function endSession(
  threadId: string,
  feedback?: SessionFeedbackLabel,
  memoryMode?: ApiMemoryMode
): Promise<EndSessionResponse> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feedback: feedback ?? null,
      ...memoryModePayload(memoryMode),
    }),
  });
  if (!res.ok) throw new Error(`End session failed: ${res.status}`);
  return res.json();
}

export async function getMemoryStatus(
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<MemoryStatus> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/status?${params}`);
  if (!res.ok) throw new Error(`Memory status failed: ${res.status}`);
  return res.json();
}

export async function updateMemoryRecall(
  enabled: boolean,
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<MemoryRecallUpdateResponse> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
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
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<MemoryFact[]> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/facts?${params}`);
  if (!res.ok) throw new Error(`Memory facts failed: ${res.status}`);
  return res.json();
}

export async function getMemorySessions(
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<MemorySession[]> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/sessions?${params}`);
  if (!res.ok) throw new Error(`Memory sessions failed: ${res.status}`);
  return res.json();
}

export async function getMemoryRules(
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<MemoryRule[]> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
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
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<DeleteMemoryResponse> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/facts/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete fact failed: ${res.status}`);
  return res.json();
}

export async function deleteMemorySession(
  index: number,
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<DeleteMemoryResponse> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/sessions/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete session failed: ${res.status}`);
  return res.json();
}

export async function deleteMemoryRule(
  index: number,
  threadId: string,
  userId?: string,
  memoryMode?: ApiMemoryMode
): Promise<DeleteMemoryResponse> {
  const params = threadScopedParams({ threadId, userId, memoryMode });
  const res = await fetch(`${API_BASE}/memory/rules/${index}?${params}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete rule failed: ${res.status}`);
  return res.json();
}

// ── Thread state (raw agent state dict) ─────────────────────────────

export async function getThreadState(
  threadId: string,
  memoryMode?: ApiMemoryMode
): Promise<Record<string, unknown> | null> {
  const params = new URLSearchParams();
  setOptionalParam(params, "memory_mode", memoryMode);
  const res = await fetch(
    `${API_BASE}/threads/${threadId}/state${querySuffix(params)}`
  );
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
  memoryMode,
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
        ...memoryModePayload(memoryMode),
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

// ── OpenAI Realtime voice session helpers ───────────────────────────

export async function createRealtimeVoiceSession({
  threadId,
  userId,
  memoryMode,
  assistantVoice,
}: {
  threadId: string;
  userId?: string;
  memoryMode: VoiceMemoryMode;
  assistantVoice?: AssistantVoiceOption;
}): Promise<RealtimeVoiceSessionResponse> {
  const res = await fetch(`${API_BASE}/voice/realtime/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      user_id: userId || undefined,
      memory_mode: memoryMode,
      assistant_voice: assistantVoice || undefined,
    }),
  });
  if (!res.ok) {
    throw new Error(`Realtime voice session failed: ${res.status}`);
  }
  return res.json();
}

export async function executeRealtimeVoiceTool({
  threadId,
  userId,
  currentUserMessage,
  transcript,
  memoryMode,
  toolName,
  arguments: args,
}: {
  threadId: string;
  userId?: string;
  currentUserMessage?: string;
  transcript?: Record<string, unknown>[];
  memoryMode: VoiceMemoryMode;
  toolName: string;
  arguments?: Record<string, unknown>;
}): Promise<RealtimeVoiceToolCallResponse> {
  const res = await fetch(`${API_BASE}/voice/realtime/tools`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      user_id: userId || undefined,
      current_user_message: currentUserMessage || "",
      transcript: transcript || [],
      memory_mode: memoryMode,
      tool_name: toolName,
      arguments: args || {},
    }),
  });
  if (!res.ok) {
    throw new Error(`Realtime voice tool failed: ${res.status}`);
  }
  return res.json();
}

export async function recordRealtimeVoiceTurn({
  threadId,
  userId,
  userText,
  assistantText,
  memoryMode,
  toolCalls,
}: {
  threadId: string;
  userId?: string;
  userText: string;
  assistantText: string;
  memoryMode: VoiceMemoryMode;
  toolCalls?: RealtimeVoiceRecordedToolCall[];
}): Promise<RealtimeVoiceTurnRecordResponse> {
  const res = await fetch(`${API_BASE}/voice/realtime/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      user_id: userId || undefined,
      user_text: userText,
      assistant_text: assistantText,
      memory_mode: memoryMode,
      tool_calls: toolCalls || [],
    }),
  });
  if (!res.ok) {
    throw new Error(`Realtime voice turn record failed: ${res.status}`);
  }
  return res.json();
}

export async function endRealtimeVoiceSession(
  threadId: string,
  memoryMode: VoiceMemoryMode
): Promise<RealtimeVoiceEndSessionResponse> {
  const res = await fetch(`${API_BASE}/voice/realtime/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, memory_mode: memoryMode }),
  });
  if (!res.ok) {
    throw new Error(`Realtime voice end failed: ${res.status}`);
  }
  return res.json();
}
