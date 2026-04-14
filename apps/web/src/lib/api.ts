/**
 * API client for the OpenCouch backend.
 *
 * All endpoints are relative to the backend server running at
 * NEXT_PUBLIC_API_URL (default: http://localhost:8000/api).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const WS_BASE = API_BASE.replace(/^http/, "ws");

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
  proactive_recall_enabled: boolean;
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
  userId?: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId,
      user_id: userId || undefined,
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

export async function endSession(threadId: string): Promise<unknown> {
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
): Promise<Record<string, unknown>[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/facts?${params}`);
  if (!res.ok) throw new Error(`Memory facts failed: ${res.status}`);
  return res.json();
}

export async function getMemorySessions(
  threadId: string,
  userId?: string
): Promise<Record<string, unknown>[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/sessions?${params}`);
  if (!res.ok) throw new Error(`Memory sessions failed: ${res.status}`);
  return res.json();
}

export async function getMemoryRules(
  threadId: string,
  userId?: string
): Promise<Record<string, unknown>[]> {
  const params = new URLSearchParams({ thread_id: threadId });
  if (userId) params.set("user_id", userId);
  const res = await fetch(`${API_BASE}/memory/rules?${params}`);
  if (!res.ok) throw new Error(`Memory rules failed: ${res.status}`);
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
  onEvent?: (event: StreamEvent) => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/chat/stream`);

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        message,
        thread_id: threadId,
        user_id: userId || undefined,
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
  callbacks: {
    onAudio?: (audioBytes: Uint8Array) => void;
    onTranscript?: (role: "user" | "assistant", text: string) => void;
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
      })
    );
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "audio" && callbacks.onAudio) {
      const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
      callbacks.onAudio(bytes);
    } else if (data.type === "transcript" && callbacks.onTranscript) {
      callbacks.onTranscript(data.role, data.text);
    } else if (data.type === "error" && callbacks.onError) {
      callbacks.onError(data.message);
    }
  };

  return ws;
}
