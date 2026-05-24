"use client";

import { useEffect, useRef, useState, type ReactElement } from "react";
import Link from "next/link";
import {
  type RealtimeVoiceSessionResponse,
  type RealtimeVoiceEndSessionResponse,
  type RealtimeVoiceTurnPolicyResponse,
  type VoiceMemoryMode,
} from "@/lib/api";
import {
  connectRealtimeVoiceSession,
  type RealtimeVoiceConnectionStatus,
  type RealtimeVoiceSessionHandle,
  type RealtimeVoiceToolEvent,
} from "@/lib/realtime-voice-session";
import { type RealtimeTranscriptUpdate } from "@/lib/realtime-voice-events";
import { useSessionStore } from "@/lib/session";
import { CouchLogo } from "@/components/logo";
import { SessionPill } from "@/components/conversation-shell";

type LogLevel = "info" | "event" | "tool" | "error";

interface DevLog {
  id: string;
  timestamp: string;
  level: LogLevel;
  label: string;
  detail?: string;
  payload?: Record<string, unknown>;
}

interface TranscriptRow {
  id: string;
  role: "user" | "assistant";
  text: string;
  final: boolean;
}

interface ToolRow {
  callId: string;
  name: string;
  status: "started" | "completed" | "failed";
  detail?: string;
  output?: Record<string, unknown>;
}

const STATUS_LABELS: Record<RealtimeVoiceConnectionStatus, string> = {
  requesting_session: "requesting session",
  requesting_microphone: "requesting mic",
  connecting: "connecting",
  connected: "connected",
  finalizing: "finalizing",
  disconnected: "disconnected",
};

const IconMic = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
);

const IconStop = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

const IconRefresh = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M20 12a8 8 0 1 1-2.3-5.7" />
    <path d="M20 4v6h-6" />
  </svg>
);

export default function RealtimeVoiceDogfoodPage(): ReactElement {
  const threadId = useSessionStore((s) => s.threadId);
  const userId = useSessionStore((s) => s.userId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const setVoiceConnected = useSessionStore((s) => s.setVoiceConnected);
  const setVoiceAgentSpeaking = useSessionStore((s) => s.setVoiceAgentSpeaking);
  const setVoiceReadyToSpeak = useSessionStore((s) => s.setVoiceReadyToSpeak);
  const setVoiceError = useSessionStore((s) => s.setVoiceError);
  const addVoiceTranscript = useSessionStore((s) => s.addVoiceTranscript);
  const clearVoiceTranscripts = useSessionStore((s) => s.clearVoiceTranscripts);
  const bumpMemoryRefreshVersion = useSessionStore(
    (s) => s.bumpMemoryRefreshVersion
  );

  const audioRef = useRef<HTMLAudioElement>(null);
  const handleRef = useRef<RealtimeVoiceSessionHandle | null>(null);
  const [memoryMode, setMemoryMode] = useState<VoiceMemoryMode>(sessionMode);
  const [status, setStatus] =
    useState<RealtimeVoiceConnectionStatus>("disconnected");
  const [session, setSession] = useState<RealtimeVoiceSessionResponse | null>(
    null
  );
  const [endedSession, setEndedSession] =
    useState<RealtimeVoiceEndSessionResponse | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptRow[]>([]);
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [latestPolicy, setLatestPolicy] =
    useState<RealtimeVoiceTurnPolicyResponse | null>(null);
  const [logs, setLogs] = useState<DevLog[]>([]);
  const [showRawEvents, setShowRawEvents] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connected = status === "connected";
  const busy =
    status === "requesting_session" ||
    status === "requesting_microphone" ||
    status === "connecting" ||
    status === "finalizing";

  useEffect(() => {
    return () => {
      void handleRef.current?.disconnect({ finalize: false });
      setVoiceConnected(false);
      setVoiceAgentSpeaking(false);
      setVoiceReadyToSpeak(false);
    };
  }, [setVoiceAgentSpeaking, setVoiceConnected, setVoiceReadyToSpeak]);

  const pushLog = (
    level: LogLevel,
    label: string,
    detail?: string,
    payload?: Record<string, unknown>
  ) => {
    setLogs((current) =>
      [
        {
          id: crypto.randomUUID(),
          timestamp: new Date().toLocaleTimeString(),
          level,
          label,
          detail,
          payload,
        },
        ...current,
      ].slice(0, 80)
    );
  };

  const handleConnect = async () => {
    if (!audioRef.current || !threadId || busy || connected) return;

    setError(null);
    setEndedSession(null);
    setSession(null);
    setTranscripts([]);
    setTools([]);
    setLatestPolicy(null);
    setLogs([]);
    clearVoiceTranscripts();
    setVoiceError(null);
    setVoiceConnected(false);
    setVoiceAgentSpeaking(false);
    setVoiceReadyToSpeak(false);

    try {
      handleRef.current = await connectRealtimeVoiceSession({
        threadId,
        userId: memoryMode === "persistent" ? userId || undefined : undefined,
        memoryMode,
        audioElement: audioRef.current,
        onStatus: (nextStatus) => {
          setStatus(nextStatus);
          pushLog("info", "status", STATUS_LABELS[nextStatus]);
          setVoiceConnected(nextStatus === "connected");
        },
        onSession: (nextSession) => {
          setSession(nextSession);
          pushLog(
            "info",
            "session",
            `${nextSession.session_config.model ?? "realtime"} / ${nextSession.memory_mode}`
          );
        },
        onRawEvent: (event) => {
          pushLog(
            "event",
            String(event.type ?? "server event"),
            undefined,
            event
          );
        },
        onTranscript: (update) => {
          mergeTranscript(update);
          if (update.final) {
            addVoiceTranscript({
              role: update.role,
              text: update.text,
              itemId: update.itemId,
            });
          }
        },
        onToolEvent: (toolEvent) => {
          mergeToolEvent(toolEvent);
          pushLog(
            toolEvent.status === "failed" ? "error" : "tool",
            toolEvent.name,
            toolEvent.status,
            toolEvent.output
          );
        },
        onTurnPolicy: (policy) => {
          setLatestPolicy(policy);
          pushLog(
            "info",
            "turn policy",
            `${policy.route} / ${policy.response_style}`,
            {
              required_tool_name: policy.required_tool_name,
              required_tool_arguments: policy.required_tool_arguments,
            }
          );
        },
        onTurnRecorded: (response) => {
          pushLog(
            "info",
            "turn recorded",
            `${response.message_count} messages in backend state`
          );
          if (memoryMode === "persistent") bumpMemoryRefreshVersion();
        },
        onEnded: (response) => {
          setEndedSession(response);
          pushLog(
            "info",
            "session finalized",
            response.summary || response.detail
          );
          if (memoryMode === "persistent") bumpMemoryRefreshVersion();
        },
        onAgentSpeaking: setVoiceAgentSpeaking,
        onReadyToSpeak: setVoiceReadyToSpeak,
        onError: (err) => {
          setError(err.message);
          setVoiceError(err.message);
          pushLog("error", "error", err.message);
        },
      });
    } catch {
      handleRef.current = null;
      setVoiceConnected(false);
    }
  };

  const handleDisconnect = async () => {
    if (!handleRef.current || busy) return;
    const handle = handleRef.current;
    handleRef.current = null;
    try {
      await handle.disconnect({ finalize: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Could not disconnect voice session.";
      setError(message);
      setVoiceError(message);
      pushLog("error", "disconnect", message);
    } finally {
      setVoiceConnected(false);
      setVoiceAgentSpeaking(false);
      setVoiceReadyToSpeak(false);
    }
  };

  const handleReset = () => {
    setError(null);
    setEndedSession(null);
    setSession(null);
    setTranscripts([]);
    setTools([]);
    setLatestPolicy(null);
    setLogs([]);
    clearVoiceTranscripts();
  };

  const mergeTranscript = (update: RealtimeTranscriptUpdate) => {
    const id = update.itemId || `${update.role}-active`;
    setTranscripts((current) => {
      const index = current.findIndex((row) => row.id === id);
      if (index === -1) {
        return [
          ...current,
          {
            id,
            role: update.role,
            text: update.text,
            final: update.final,
          },
        ];
      }

      const next = [...current];
      const existing = next[index];
      next[index] = {
        ...existing,
        text: update.final ? update.text : existing.text + update.text,
        final: update.final,
      };
      return next;
    });
  };

  const mergeToolEvent = (event: RealtimeVoiceToolEvent) => {
    setTools((current) => {
      const index = current.findIndex((row) => row.callId === event.callId);
      if (index === -1) return [{ ...event }, ...current].slice(0, 12);

      const next = [...current];
      next[index] = { ...next[index], ...event };
      return next;
    });
  };

  return (
    <>
      <div className="oc-app-top-wrap">
        <header className="oc-app-top">
          <div className="flex items-center gap-3">
            <span className="oc-mobile-mark">
              <CouchLogo className="h-4 w-4" />
            </span>
            <div>
              <h2 className="oc-app-top-title">Realtime voice dogfood</h2>
              <p className="hidden text-[12px] text-oc-text-muted md:block">
                Temporary OpenAI WebRTC test page
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2.5">
            <StatusPill status={status} />
            <SessionPill />
          </div>
        </header>
      </div>

      <div className="oc-mobile-top-wrap">
        <header className="oc-mobile-top">
          <div className="flex items-center gap-2.5">
            <span className="oc-mobile-mark">
              <CouchLogo className="h-4 w-4" />
            </span>
            <h2 className="oc-mobile-top-title">Realtime voice</h2>
          </div>
          <StatusPill status={status} compact />
        </header>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5 md:px-6">
        <div className="mx-auto grid w-full max-w-[1280px] gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
          <section className="space-y-4">
            <div className="rounded-[8px] border border-oc-border bg-oc-bg-card p-4">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <h3 className="text-[15px] font-semibold text-oc-text">
                    Connection
                  </h3>
                  <p className="mt-1 text-[12px] text-oc-text-muted">
                    {threadId || "Start an app session first"}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-[8px] border border-oc-border bg-white text-oc-text-secondary hover:text-oc-text"
                  title="Clear dogfood logs"
                  aria-label="Clear dogfood logs"
                  disabled={busy || connected}
                >
                  <IconRefresh size={15} />
                </button>
              </div>

              <div className="mb-4 grid grid-cols-2 rounded-[8px] border border-oc-border bg-white p-1">
                {(["persistent", "incognito"] as VoiceMemoryMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setMemoryMode(mode)}
                    disabled={busy || connected}
                    className={`rounded-[6px] px-3 py-2 text-[12px] font-medium transition ${
                      memoryMode === mode
                        ? "bg-oc-teal-600 text-white"
                        : "text-oc-text-secondary hover:bg-oc-accent-subtle"
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={handleConnect}
                  disabled={!threadId || connected || busy}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-[8px] bg-oc-teal-600 px-3 text-[13px] font-semibold text-white transition hover:bg-oc-teal-700 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <IconMic />
                  Connect
                </button>
                <button
                  type="button"
                  onClick={handleDisconnect}
                  disabled={!connected || busy}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-[8px] border border-oc-border bg-white px-3 text-[13px] font-semibold text-oc-text transition hover:border-oc-border-strong disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <IconStop />
                  Disconnect
                </button>
              </div>

              {error && (
                <div className="mt-4 rounded-[8px] border border-oc-red/20 bg-oc-red-subtle px-3 py-2 text-[12.5px] text-oc-red">
                  {error}
                </div>
              )}

              <audio ref={audioRef} autoPlay className="hidden" />
            </div>

            <div className="rounded-[8px] border border-oc-border bg-white p-4">
              <h3 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-oc-text-muted">
                Session
              </h3>
              <dl className="mt-3 space-y-2 text-[13px]">
                <InfoRow label="mode" value={memoryMode} />
                <InfoRow label="user" value={memoryMode === "persistent" ? userId || "none" : "anonymous"} />
                <InfoRow label="thread" value={threadId || "none"} />
                <InfoRow label="route" value={latestPolicy?.route || "none"} />
                <InfoRow
                  label="style"
                  value={latestPolicy?.response_style || "none"}
                />
                <InfoRow
                  label="required tool"
                  value={latestPolicy?.required_tool_name || "none"}
                />
                <InfoRow
                  label="model"
                  value={String(session?.session_config.model ?? "not connected")}
                />
              </dl>
              {endedSession && (
                <div className="mt-4 rounded-[8px] bg-oc-accent-subtle px-3 py-2 text-[12.5px] leading-5 text-oc-text-secondary">
                  {endedSession.summary || endedSession.detail}
                </div>
              )}
              <Link
                href="/voice"
                className="mt-4 inline-flex text-[12px] font-medium text-oc-teal-700 hover:text-oc-teal-900"
              >
                Back to current voice page
              </Link>
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(320px,0.75fr)]">
            <div className="rounded-[8px] border border-oc-border bg-white p-4">
              <div className="mb-4 flex items-center justify-between gap-3">
                <h3 className="text-[15px] font-semibold text-oc-text">
                  Transcript
                </h3>
                <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-oc-text-muted">
                  {transcripts.length} items
                </span>
              </div>

              <div className="space-y-3">
                {transcripts.length === 0 && (
                  <EmptyState text="Connect, allow the microphone, then speak." />
                )}
                {transcripts.map((row) => (
                  <div
                    key={row.id}
                    className={`rounded-[8px] border px-3 py-2 ${
                      row.role === "assistant"
                        ? "border-oc-teal-200/70 bg-oc-teal-50"
                        : "border-oc-line bg-oc-bg-card"
                    }`}
                  >
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <span className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-oc-text-muted">
                        {row.role}
                      </span>
                      <span className="text-[11px] text-oc-text-muted">
                        {row.final ? "final" : "streaming"}
                      </span>
                    </div>
                    <p className="text-[14px] leading-6 text-oc-text-secondary">
                      {row.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-[8px] border border-oc-border bg-white p-4">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <h3 className="text-[15px] font-semibold text-oc-text">
                    Tool calls
                  </h3>
                  <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-oc-text-muted">
                    {tools.length}
                  </span>
                </div>
                <div className="space-y-2">
                  {tools.length === 0 && (
                    <EmptyState text="Ask about memory or guided exercises to trigger tools." />
                  )}
                  {tools.map((tool) => (
                    <div
                      key={tool.callId}
                      className="rounded-[8px] border border-oc-border bg-oc-bg-card px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-[12px] text-oc-text">
                          {tool.name}
                        </span>
                        <span
                          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.06em] ${
                            tool.status === "completed"
                              ? "bg-oc-green-subtle text-oc-green"
                              : tool.status === "failed"
                                ? "bg-oc-red-subtle text-oc-red"
                                : "bg-oc-orange-subtle text-oc-orange"
                          }`}
                        >
                          {tool.status}
                        </span>
                      </div>
                      {tool.detail && (
                        <p className="mt-1 text-[12px] text-oc-text-muted">
                          {tool.detail}
                        </p>
                      )}
                      {tool.output && (
                        <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-[6px] bg-white p-2 text-[11px] leading-5 text-oc-text-muted">
                          {JSON.stringify(tool.output, null, 2)}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[8px] border border-oc-border bg-white p-4">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <h3 className="text-[15px] font-semibold text-oc-text">
                    Event log
                  </h3>
                  <label className="flex items-center gap-2 text-[12px] text-oc-text-muted">
                    <input
                      type="checkbox"
                      checked={showRawEvents}
                      onChange={(event) => setShowRawEvents(event.target.checked)}
                    />
                    raw
                  </label>
                </div>
                <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
                  {logs.length === 0 && (
                    <EmptyState text="Realtime server events will appear here." />
                  )}
                  {logs.map((log) => (
                    <div
                      key={log.id}
                      className="rounded-[8px] border border-oc-line bg-oc-bg-code px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={`font-mono text-[11px] uppercase tracking-[0.06em] ${
                            log.level === "error"
                              ? "text-oc-red"
                              : log.level === "tool"
                                ? "text-oc-teal-700"
                                : "text-oc-text-muted"
                          }`}
                        >
                          {log.label}
                        </span>
                        <span className="font-mono text-[10px] text-oc-text-muted">
                          {log.timestamp}
                        </span>
                      </div>
                      {log.detail && (
                        <p className="mt-1 text-[12px] leading-5 text-oc-text-secondary">
                          {log.detail}
                        </p>
                      )}
                      {showRawEvents && log.payload && (
                        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-[6px] bg-white p-2 text-[11px] leading-5 text-oc-text-muted">
                          {JSON.stringify(log.payload, null, 2)}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

function StatusPill({
  status,
  compact = false,
}: {
  status: RealtimeVoiceConnectionStatus;
  compact?: boolean;
}) {
  const active = status === "connected";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[10.5px] uppercase tracking-[0.08em] ${
        active
          ? "border-oc-teal-200 bg-oc-teal-50 text-oc-teal-700"
          : "border-oc-border bg-white text-oc-text-muted"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          active ? "bg-oc-pulse" : "bg-oc-text-dim"
        }`}
      />
      {compact ? status : STATUS_LABELS[status]}
    </span>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <dt className="font-mono text-[11px] uppercase tracking-[0.08em] text-oc-text-muted">
        {label}
      </dt>
      <dd className="max-w-[220px] break-words text-right text-[13px] text-oc-text-secondary">
        {value}
      </dd>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-[8px] border border-dashed border-oc-border bg-oc-bg-card px-3 py-4 text-center text-[12.5px] text-oc-text-muted">
      {text}
    </div>
  );
}
