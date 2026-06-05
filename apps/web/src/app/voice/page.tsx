"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ASSISTANT_VOICE_OPTIONS,
  type ApiMemoryMode,
  type AssistantVoiceOption,
} from "@/lib/api";
import { useCommandActions } from "@/lib/command-actions";
import { useSessionStore, type EndedSessionResult } from "@/lib/session";
import { useRealtimeVoiceSession } from "@/components/realtime-voice-session-provider";
import { CouchLogo } from "@/components/logo";
import { SessionPill } from "@/components/conversation-shell";
import { SessionFeedback } from "@/components/session-feedback";

const ENABLE_VOICE_DEBUG = process.env.NEXT_PUBLIC_ENABLE_VOICE_DEBUG === "true";

type VoiceEndOptionsStatus = "idle" | "in_progress" | "completed" | "failed";

function formatVoiceTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "not yet";
  }

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return "unknown";
  }

  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function clipVoiceText(text: string, limit = 180): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1).trim()}…`;
}

function safeServerHost(value: string | null | undefined): string {
  if (!value) {
    return "none";
  }

  try {
    return new URL(value).host || "configured";
  } catch {
    return "configured";
  }
}

function assistantVoiceLabel(value: string): string {
  const match = ASSISTANT_VOICE_OPTIONS.find((option) => option.value === value);
  return match?.label ?? (value || "default");
}

function VoiceDebugRow({
  label,
  value,
}: {
  label: string;
  value: boolean | number | string | null | undefined;
}) {
  const displayValue =
    value === null || value === undefined || value === "" ? "none" : String(value);

  return (
    <div className="rounded-xl border border-oc-border-subtle bg-white/60 px-3 py-2">
      <p className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim">
        {label}
      </p>
      <p className="mt-1 break-words text-[12px] font-mono text-oc-text-secondary">
        {displayValue}
      </p>
    </div>
  );
}

function VoiceEndOptionsDialog({
  threadId,
  memoryMode,
  finalizationStatus,
  detail,
  endedSession,
  isPersistent,
  onContinueInChat,
  onReviewMemory,
  onStartNewSession,
  onStayOnVoice,
}: {
  threadId: string;
  memoryMode: ApiMemoryMode;
  finalizationStatus: VoiceEndOptionsStatus;
  detail: string | null;
  endedSession: EndedSessionResult | null;
  isPersistent: boolean;
  onContinueInChat: () => void;
  onReviewMemory: () => void;
  onStartNewSession: () => void;
  onStayOnVoice: () => void;
}) {
  const hasSummary = Boolean(endedSession?.summary);
  const reviewDisabled = !isPersistent || finalizationStatus === "in_progress";
  const statusCopy = (() => {
    if (!isPersistent) {
      return "This was an incognito voice session, so no memory will be saved.";
    }
    if (finalizationStatus === "in_progress") {
      return "Session memory is saving in the background. You can move around while it finishes.";
    }
    if (finalizationStatus === "completed") {
      return hasSummary
        ? "The session summary is ready."
        : detail || "The voice session has finished saving.";
    }
    if (finalizationStatus === "failed") {
      return (
        detail ||
        "The voice session ended, but memory saving did not finish cleanly."
      );
    }
    return "The voice session has ended.";
  })();

  return (
    <div
      className="fixed inset-0 z-[85] flex items-end justify-center bg-[rgba(21,32,29,0.36)] px-0 py-0 md:items-center md:px-6 md:py-8"
      role="presentation"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="voice-ended-title"
        className="w-full max-w-[560px] rounded-t-[22px] border border-oc-line bg-white shadow-[0_28px_80px_-34px_rgba(21,32,29,0.58)] md:rounded-[22px]"
      >
        <div className="flex items-start justify-between gap-4 border-b border-oc-line-2 px-5 py-4 md:px-6 md:py-5">
          <div>
            <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-oc-primary">
              Voice session
            </p>
            <h2
              id="voice-ended-title"
              className="mt-1 font-display text-[24px] font-semibold leading-tight text-oc-ink"
            >
              Session ended
            </h2>
            <p className="mt-1 text-[13.5px] leading-[1.55] text-oc-muted">
              {statusCopy}
            </p>
          </div>
          <button
            type="button"
            onClick={onStayOnVoice}
            aria-label="Close voice session options"
            className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-oc-line bg-white text-oc-muted transition-[color,border-color,background] hover:border-oc-primary/30 hover:bg-oc-surface-tint hover:text-oc-primary"
          >
            x
          </button>
        </div>

        <div className="px-5 py-4 md:px-6 md:py-5">
          {hasSummary && (
            <div className="mb-4 rounded-xl border border-oc-teal-200 bg-oc-teal-50/70 px-4 py-3">
              <p className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-oc-teal-700">
                Summary
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-oc-text">
                {endedSession?.summary}
              </p>
              {endedSession?.themes && endedSession.themes.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {endedSession.themes.map((theme) => (
                    <span
                      key={theme}
                      className="rounded-md border border-oc-teal-200/70 bg-white/70 px-2 py-0.5 text-[11px] font-mono text-oc-teal-700"
                    >
                      {theme}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          <SessionFeedback
            threadId={threadId}
            memoryMode={memoryMode}
            modality="voice"
            className="mb-4"
          />

          <div className="grid gap-2">
            <button
              type="button"
              onClick={onContinueInChat}
              className="rounded-xl bg-oc-primary px-4 py-3 text-left text-[14px] font-medium text-[#F6F1E5] transition-colors hover:bg-oc-primary-2"
            >
              Continue in chat
              <span className="mt-0.5 block text-[12px] font-normal opacity-80">
                Return to text without starting over.
              </span>
            </button>
            <button
              type="button"
              onClick={onStartNewSession}
              className="rounded-xl border border-oc-line bg-white px-4 py-3 text-left text-[14px] font-medium text-oc-ink transition-colors hover:bg-oc-surface-tint"
            >
              Start a new session
              <span className="mt-0.5 block text-[12px] font-normal text-oc-muted">
                Go home and choose mode, user, and thread again.
              </span>
            </button>
            <button
              type="button"
              onClick={onReviewMemory}
              disabled={reviewDisabled}
              className="rounded-xl border border-oc-line bg-white px-4 py-3 text-left text-[14px] font-medium text-oc-ink transition-colors hover:bg-oc-surface-tint disabled:cursor-not-allowed disabled:opacity-50"
            >
              Review memory
              <span className="mt-0.5 block text-[12px] font-normal text-oc-muted">
                {!isPersistent
                  ? "Not available for incognito sessions."
                  : finalizationStatus === "in_progress"
                    ? "Available after memory saving finishes."
                    : "Open saved facts, summaries, and style rules."}
              </span>
            </button>
            <button
              type="button"
              onClick={onStayOnVoice}
              className="rounded-xl border border-transparent px-4 py-2.5 text-center text-[13px] font-medium text-oc-muted transition-colors hover:bg-oc-surface-tint hover:text-oc-primary"
            >
              Stay on Voice
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

const IconMic = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
);

function VoiceOrb({
  size = 280,
  active = false,
  waiting = false,
  listening = false,
}: {
  size?: number;
  active?: boolean;
  waiting?: boolean;
  listening?: boolean;
}) {
  const showBars = active || listening;
  return (
    <div
      className={`oc-orb-wrap${active ? " oc-orb-active" : ""}${
        waiting ? " oc-orb-waiting" : ""
      }${listening ? " oc-orb-listening" : ""}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <div className="oc-orb-ring" />
      <div className="oc-orb-ring oc-orb-ring-2" />
      <div className="oc-orb-core">
        {showBars && (
          <div className="oc-orb-bars">
            <i /><i /><i /><i /><i />
          </div>
        )}
        <IconMic size={size * 0.18} />
      </div>
    </div>
  );
}

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function VoicePage() {
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const chatLoading = useSessionStore((s) => s.chatLoading);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const voiceAgentSpeaking = useSessionStore((s) => s.voiceAgentSpeaking);
  const voiceReadyToSpeak = useSessionStore((s) => s.voiceReadyToSpeak);
  const assistantVoiceSelected = useSessionStore(
    (s) => s.assistantVoiceSelected
  );
  const voiceTranscripts = useSessionStore((s) => s.voiceTranscripts);
  const voiceActivities = useSessionStore((s) => s.voiceActivities);
  const voiceFinalization = useSessionStore((s) => s.voiceFinalization);
  const voiceSessionInfo = useSessionStore((s) => s.voiceSessionInfo);
  const lastEndedSession = useSessionStore((s) => s.lastEndedSession);
  const clearLastEndedSession = useSessionStore((s) => s.clearLastEndedSession);
  const voiceError = useSessionStore((s) => s.voiceError);
  const setVoiceError = useSessionStore((s) => s.setVoiceError);
  const setChatNotice = useSessionStore((s) => s.setChatNotice);
  const setAssistantVoiceSelected = useSessionStore(
    (s) => s.setAssistantVoiceSelected
  );
  const clearVoiceTranscripts = useSessionStore((s) => s.clearVoiceTranscripts);
  const clearVoiceActivities = useSessionStore((s) => s.clearVoiceActivities);
  const realtimeVoice = useRealtimeVoiceSession();
  const { startNewSession } = useCommandActions();
  const router = useRouter();
  const [voiceEndOptionsOpen, setVoiceEndOptionsOpen] = useState(false);
  const [endedVoiceThreadId, setEndedVoiceThreadId] = useState<string | null>(null);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  // Live call timer — derived from `connectedAt` so we don't need an effect
  // to reset state when the call ends. The interval ticks `now` purely to
  // re-render the formatted timer; React skips the update when the deps are
  // stable, so disconnect drops back to 00:00 with no cascade.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!voiceConnected) return;
    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => window.clearInterval(interval);
  }, [voiceConnected]);
  const connectedAtMs = voiceSessionInfo?.connectedAt
    ? Date.parse(voiceSessionInfo.connectedAt)
    : NaN;
  const callElapsedMs =
    voiceConnected && Number.isFinite(connectedAtMs)
      ? Math.max(0, now - connectedAtMs)
      : 0;

  useEffect(() => {
    transcriptScrollRef.current?.scrollTo({
      top: transcriptScrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [voiceTranscripts]);

  const connect = useCallback(async () => {
    if (voiceConnected || realtimeVoice.busy) {
      return;
    }

    if (chatLoading) {
      setVoiceError("Wait for the current text reply to finish before starting voice.");
      return;
    }

    setVoiceError(null);
    setVoiceEndOptionsOpen(false);
    clearVoiceTranscripts();
    clearVoiceActivities();

    try {
      await realtimeVoice.connect();
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to start Realtime voice session.";
      setVoiceError(message);
    }
  }, [
    clearVoiceActivities,
    clearVoiceTranscripts,
    chatLoading,
    realtimeVoice,
    setVoiceError,
    voiceConnected,
  ]);

  const disconnect = useCallback(async () => {
    const disconnectedThreadId = threadId;
    setVoiceError(null);
    setEndedVoiceThreadId(disconnectedThreadId);
    setVoiceEndOptionsOpen(true);
    await realtimeVoice.disconnect({ finalize: true });
  }, [realtimeVoice, setVoiceError, threadId]);

  const realtimeConnecting =
    realtimeVoice.status === "requesting_session" ||
    realtimeVoice.status === "requesting_microphone" ||
    realtimeVoice.status === "connecting";

  const realtimeFinalizing = realtimeVoice.status === "finalizing";

  const isSavingDisconnectedSession =
    (!voiceConnected || realtimeFinalizing) &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "in_progress";
  const finalizationFailedForCurrentThread =
    !voiceConnected &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "failed";
  const finalizationCompletedForCurrentThread =
    !voiceConnected &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "completed";
  const connectDisabled =
    chatLoading || realtimeConnecting || realtimeFinalizing || isSavingDisconnectedSession;
  const latestTranscript = voiceTranscripts[voiceTranscripts.length - 1] ?? null;
  const latestActivity = voiceActivities[voiceActivities.length - 1] ?? null;
  const effectiveVoiceMemoryMode = voiceSessionInfo?.memoryMode || sessionMode;
  const isPersistent = effectiveVoiceMemoryMode === "persistent";
  const endOptionsFinalizationStatus: VoiceEndOptionsStatus =
    endedVoiceThreadId && voiceFinalization.threadId === endedVoiceThreadId
      ? voiceFinalization.status
      : "idle";
  const endOptionsFinalizationDetail =
    endedVoiceThreadId && voiceFinalization.threadId === endedVoiceThreadId
      ? voiceFinalization.detail
      : null;
  const endedVoiceSession =
    endedVoiceThreadId && lastEndedSession?.threadId === endedVoiceThreadId
      ? lastEndedSession
      : null;
  const voiceCanAcceptSpeech =
    voiceConnected &&
    realtimeVoice.status === "connected" &&
    voiceReadyToSpeak &&
    !voiceAgentSpeaking;
  const voiceIsWarming = voiceConnected && !voiceCanAcceptSpeech;

  const statusText = (() => {
    if (realtimeVoice.status === "requesting_session") {
      return "creating a realtime session...";
    }
    if (realtimeVoice.status === "requesting_microphone") {
      return "waiting for microphone access...";
    }
    if (realtimeVoice.status === "connecting") {
      return "connecting to openai realtime...";
    }
    if (realtimeVoice.status === "finalizing") {
      return "ending voice session...";
    }
    if (!voiceConnected) {
      return "connect when you're ready";
    }
    if (voiceAgentSpeaking) {
      return "agent speaking...";
    }
    if (realtimeVoice.status === "connected") {
      return "connected - speak naturally";
    }
    return "ready - speak when you want";
  })();

  const readinessNotice = (() => {
    if (!voiceConnected || voiceCanAcceptSpeech) {
      return null;
    }
    if (realtimeVoice.status === "requesting_microphone") {
      return {
        title: "Allow microphone access",
        detail:
          "The browser needs microphone permission before the voice session can start.",
      };
    }
    if (realtimeVoice.status === "connecting") {
      return {
        title: "Connecting voice",
        detail: "Realtime audio is setting up. The mic opens once the session is ready.",
      };
    }
    return null;
  })();

  const continueInChat = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    if (endOptionsFinalizationStatus === "in_progress") {
      setChatNotice(
        "Voice session ended. Memory is saving in the background; you can keep using chat."
      );
    } else if (endOptionsFinalizationStatus === "failed") {
      setChatNotice(
        endOptionsFinalizationDetail ||
          "Voice session ended, but the latest memory may not be saved yet."
      );
    } else {
      setChatNotice(null);
    }
    router.push("/");
  }, [
    endOptionsFinalizationDetail,
    endOptionsFinalizationStatus,
    router,
    setChatNotice,
  ]);

  const reviewMemory = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    clearLastEndedSession();
    router.push("/memory");
  }, [clearLastEndedSession, router]);

  const startFreshSession = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    clearLastEndedSession();
    void startNewSession();
  }, [clearLastEndedSession, startNewSession]);

  const eyebrowState: "idle" | "warming" | "listening" | "speaking" = (() => {
    if (!voiceConnected) return "idle";
    if (voiceAgentSpeaking) return "speaking";
    if (voiceCanAcceptSpeech) return "listening";
    return "warming";
  })();

  const eyebrowLabel = (() => {
    if (eyebrowState === "idle") return "agent · idle";
    if (eyebrowState === "speaking") return "agent · speaking";
    if (eyebrowState === "listening") return "agent · listening";
    return "agent · wait";
  })();

  const mobileVoiceTitle = (() => {
    if (!voiceConnected) return "Voice";
    if (voiceAgentSpeaking) return "speaking";
    if (voiceCanAcceptSpeech) return "listening";
    return "warming up";
  })();

  const memoryConsolidationMessage = (() => {
    if (isSavingDisconnectedSession) {
      return "Voice memory is saving in the background. You can use chat or other tabs while it finishes; reconnecting voice unlocks when the latest session memory is ready.";
    }
    if (finalizationCompletedForCurrentThread) {
      return "Memory consolidation finished. The saved summary and extracted memories are now available for future persistent sessions.";
    }
    if (finalizationFailedForCurrentThread) {
      return "Memory consolidation did not finish cleanly. You can reconnect, but the latest voice session may not be saved yet.";
    }
    return null;
  })();

  const showMemoryConsolidationNote =
    isPersistent && memoryConsolidationMessage !== null;

  const consolidationToneClass = finalizationFailedForCurrentThread
    ? "border-oc-red/20 bg-oc-red-subtle text-oc-red"
    : finalizationCompletedForCurrentThread
      ? "border-oc-green/20 bg-green-50 text-oc-green"
      : "border-oc-teal-200 bg-oc-teal-50 text-oc-teal-800";

  const callTimerText = formatElapsed(callElapsedMs);

  return (
    <>
      {/* Desktop top bar — wrapper controls breakpoint visibility */}
      <div className="oc-app-top-wrap">
      <header className="oc-app-top">
        <div className="flex items-center gap-3">
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-app-top-title">Voice</h2>
          {voiceConnected && (
            <span
              className="text-[11px] font-mono"
              style={{
                color: "var(--color-oc-primary)",
                letterSpacing: "0.06em",
              }}
            >
              ● live · {callTimerText}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-[10px] font-mono uppercase tracking-[0.08em] text-oc-text-dim">
            openai realtime
          </span>
          <SessionPill />
        </div>
      </header>
      </div>

      {/* Mobile top bar — wrapper controls breakpoint visibility */}
      <div className="oc-mobile-top-wrap">
      <header className="oc-mobile-top">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2
            className="oc-mobile-top-title"
            style={voiceConnected ? { fontStyle: "italic", fontWeight: 500 } : undefined}
          >
            {mobileVoiceTitle}
          </h2>
        </div>
        {voiceConnected ? (
          <span
            className="text-[10px] font-mono"
            style={{
              color: "var(--color-oc-primary)",
              letterSpacing: "0.06em",
            }}
          >
            ● live · {callTimerText}
          </span>
        ) : (
          <span className="text-[10px] font-mono uppercase tracking-[0.08em] text-oc-text-dim">
            openai realtime
          </span>
        )}
      </header>
      </div>

      {/* Main Content Area: split layout on desktop */}
      <div className="flex flex-col md:flex-row flex-1 min-h-0">

        {/* Voice stage — left side on desktop */}
        <div className="oc-voice-stage md:w-3/5 md:border-r border-oc-line-2 flex-shrink-0 md:flex-shrink">
          <div
            className={`oc-voice-eyebrow oc-voice-eyebrow--${eyebrowState} ${
              eyebrowState !== "idle" ? "is-ready" : ""
            }`}
          >
            <span className="dot" />
            {eyebrowLabel}
          </div>

          <VoiceOrb
            size={280}
            active={voiceAgentSpeaking}
            waiting={voiceIsWarming}
            listening={voiceCanAcceptSpeech}
          />

          {!voiceConnected ? (
          <>
            <h1 className="oc-voice-title">
              Take a breath. <em>Begin when ready.</em>
            </h1>
            <p className="oc-voice-sub">
              {sessionMode === "incognito"
                ? "Incognito voice: nothing is saved this call."
                : "Persistent voice shares the same memory as text. We won’t speak until you do."}
            </p>
            {finalizationCompletedForCurrentThread && (
              <p className="text-[12px] font-mono text-oc-green mb-3 -mt-3">
                {voiceFinalization.detail || "Previous voice session memory saved."}
              </p>
            )}
            {finalizationFailedForCurrentThread && (
              <p className="text-[12px] font-mono text-oc-red mb-3 -mt-3">
                {voiceFinalization.detail ||
                  "The previous voice session may still be finishing its memory save. You can reconnect, but the latest memory may not be ready yet."}
              </p>
            )}
            {isSavingDisconnectedSession && (
              <p className="text-[12px] font-mono text-oc-teal-700 mb-3 -mt-3">
                Saving session memory from the previous call. You can use chat
                while this finishes.
              </p>
            )}
            {chatLoading && (
              <p className="text-[12px] font-mono text-oc-teal-700 mb-3 -mt-3">
                Text reply in progress. Voice unlocks when it finishes.
              </p>
            )}
            <button
              type="button"
              className="oc-voice-cta"
              onClick={() => void connect()}
              disabled={connectDisabled}
            >
              <IconMic size={16} />
              {isSavingDisconnectedSession
                ? "Saving memory…"
                : chatLoading
                  ? "Replying…"
                  : realtimeConnecting
                  ? "Connecting…"
                  : "Connect voice"}
            </button>
            {isSavingDisconnectedSession && (
              <button
                type="button"
                className="oc-voice-cta oc-voice-cta--secondary mt-3"
                onClick={() => router.push("/")}
              >
                Use chat
              </button>
            )}

            {/* Voice controls — quiet inline controls, not cards */}
            <div
              className="mt-5 flex flex-wrap items-center justify-center gap-3"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--color-oc-text-dim)",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              <label className="flex items-center gap-2">
                <span>voice</span>
                <select
                  value={assistantVoiceSelected}
                  onChange={(event) =>
                    setAssistantVoiceSelected(
                      event.target.value as AssistantVoiceOption
                    )
                  }
                  disabled={voiceConnected}
                  className="rounded-lg border border-oc-border bg-white px-2.5 py-1.5 text-[12px] font-mono normal-case tracking-normal text-oc-text-secondary disabled:opacity-60"
                >
                  {ASSISTANT_VOICE_OPTIONS.map((option) => (
                    <option key={option.label} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {/* Single meta line — replaces the four debug cards. */}
            <div className="oc-voice-meta">
              <span>
                voice <b>{assistantVoiceLabel(assistantVoiceSelected)}</b>
              </span>
              <span className="sep">·</span>
              <span>
                memory <b>{isPersistent ? "persistent" : "off"}</b>
              </span>
              {voiceSessionInfo?.roomName && (
                <>
                  <span className="sep">·</span>
                  <span>
                    thread <b>{voiceSessionInfo.roomName}</b>
                  </span>
                </>
              )}
            </div>
          </>
        ) : (
          <>
            {latestTranscript ? (
              <p
                style={{
                  fontFamily: "var(--font-display)",
                  fontStyle: "italic",
                  fontSize: 22,
                  color: "var(--color-oc-ink-2)",
                  lineHeight: 1.4,
                  maxWidth: 360,
                  margin: "0 auto",
                  textAlign: "center",
                }}
              >
                &ldquo;
                <span
                  style={{
                    color:
                      latestTranscript.role === "user"
                        ? "var(--color-oc-primary)"
                        : "var(--color-oc-ink-2)",
                  }}
                >
                  {clipVoiceText(latestTranscript.text, 140)}
                </span>
                &rdquo;
              </p>
            ) : (
              <p
                style={{
                  fontFamily: "var(--font-display)",
                  fontStyle: "italic",
                  fontSize: 22,
                  color: "var(--color-oc-muted)",
                  lineHeight: 1.4,
                  maxWidth: 320,
                  margin: "0 auto",
                  textAlign: "center",
                }}
              >
                {statusText}
              </p>
            )}
            {readinessNotice && (
              <div
                className="oc-voice-readiness"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <span className="oc-voice-readiness-icon">
                  <IconMic size={16} />
                </span>
                <span className="oc-voice-readiness-copy">
                  <b>{readinessNotice.title}</b>
                  <span>{readinessNotice.detail}</span>
                </span>
              </div>
            )}
            <div className="mt-6 flex items-center gap-3">
              <button
                type="button"
                className="oc-voice-cta oc-voice-cta--secondary text-oc-red"
                onClick={() => void disconnect()}
              >
                End session
              </button>
            </div>

            {ENABLE_VOICE_DEBUG && (
              <div className="oc-voice-meta" style={{ marginTop: 18 }}>
                {latestActivity && (
                  <>
                    <span>
                      {latestActivity.label} <b>{latestActivity.status}</b>
                    </span>
                    <span className="sep">·</span>
                  </>
                )}
                <span>
                  mic <b>{voiceCanAcceptSpeech ? "open" : "muted"}</b>
                </span>
                <span className="sep">·</span>
                <span>
                  voice{" "}
                  <b>
                    {assistantVoiceLabel(
                      voiceSessionInfo?.assistantVoice || assistantVoiceSelected
                    )}
                  </b>
                </span>
              </div>
            )}
          </>
        )}
        </div>

        {/* Right side on desktop: Transcript & notices */}
        <div className="flex flex-col flex-1 min-h-0 md:w-2/5 bg-white/40 backdrop-blur-sm">

      {/* Memory consolidation note — surfaced inline, not as a card. */}
      {showMemoryConsolidationNote && (
        <div className="px-6 pb-4 flex justify-center">
          <div
            className={`max-w-2xl rounded-2xl border px-4 py-3 text-[13px] leading-relaxed ${consolidationToneClass}`}
          >
            {memoryConsolidationMessage}
          </div>
        </div>
      )}

      {voiceError && (
        <div className="px-6 pb-4 flex justify-center">
          <div className="max-w-2xl rounded-2xl border border-oc-red/20 bg-oc-red-subtle px-4 py-3 text-[13px] text-oc-red">
            {voiceError}
          </div>
        </div>
      )}

      {voiceEndOptionsOpen && (
        <VoiceEndOptionsDialog
          threadId={endedVoiceThreadId || threadId}
          memoryMode={
            effectiveVoiceMemoryMode === "incognito" ? "incognito" : "persistent"
          }
          finalizationStatus={endOptionsFinalizationStatus}
          detail={endOptionsFinalizationDetail}
          endedSession={endedVoiceSession}
          isPersistent={isPersistent}
          onContinueInChat={continueInChat}
          onReviewMemory={reviewMemory}
          onStartNewSession={startFreshSession}
          onStayOnVoice={() => setVoiceEndOptionsOpen(false)}
        />
      )}

      {/* Transcript scroller */}
      {voiceTranscripts.length > 0 ? (
        <div className="flex flex-col flex-1 min-h-0 border-t md:border-t-0 border-oc-line-2">
          <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-oc-line-2 md:px-6 shrink-0 bg-oc-bg-card/50">
            <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim">
              Transcript
            </span>
            <span className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim/80">
              realtime
            </span>
          </div>
          <div
            ref={transcriptScrollRef}
            className="flex-1 overflow-y-auto px-4 py-4 space-y-3 md:px-6"
          >
            {voiceTranscripts.map((t, i) => {
              const variant =
                t.role === "user"
                  ? "user"
                  : t.role === "assistant"
                    ? "assistant"
                    : "system";
              return (
                <div
                  key={t.itemId ? `${t.role}:${t.itemId}` : `${t.role}:${i}`}
                  className={`oc-vt-row oc-vt-row--${variant} animate-fadeIn`}
                >
                  <span className="oc-vt-role">{t.role}</span>
                  <span className="oc-vt-bubble">{t.text}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="hidden md:flex flex-1 items-center justify-center text-oc-text-dim font-mono text-[12px] p-6 text-center">
          <p>Transcript will appear here once you connect and speak.</p>
        </div>
      )}

      {/* Verbose debug overlay — replaces the four always-visible status cards. */}
      {ENABLE_VOICE_DEBUG && (
        <details className="mx-4 mb-4 rounded-2xl border border-dashed border-oc-border bg-white/70 px-5 py-4 text-left shadow-sm md:mx-6">
          <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
            Voice debug
          </summary>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            <VoiceDebugRow label="connection" value={realtimeVoice.status} />
            <VoiceDebugRow
              label="provider connected"
              value={realtimeVoice.connected}
            />
            <VoiceDebugRow label="store connected" value={voiceConnected} />
            <VoiceDebugRow
              label="agent state"
              value={
                voiceAgentSpeaking
                  ? "speaking"
                  : voiceCanAcceptSpeech
                    ? "listening"
                    : "idle"
              }
            />
            <VoiceDebugRow label="ready to speak" value={voiceReadyToSpeak} />
            <VoiceDebugRow label="mic enabled" value={voiceCanAcceptSpeech} />
            <VoiceDebugRow label="agent speaking" value={voiceAgentSpeaking} />
            <VoiceDebugRow label="memory mode" value={effectiveVoiceMemoryMode} />
            <VoiceDebugRow label="thread" value={voiceSessionInfo?.roomName} />
            <VoiceDebugRow
              label="participant"
              value={voiceSessionInfo?.identity}
            />
            <VoiceDebugRow
              label="server host"
              value={safeServerHost(voiceSessionInfo?.serverUrl)}
            />
            <VoiceDebugRow
              label="connected at"
              value={formatVoiceTimestamp(voiceSessionInfo?.connectedAt)}
            />
            <VoiceDebugRow label="user id" value={userId || "none"} />
            <VoiceDebugRow label="thread id" value={threadId} />
            <VoiceDebugRow
              label="selected voice"
              value={assistantVoiceSelected}
            />
            <VoiceDebugRow
              label="turn response"
              value="realtime automatic"
            />
            <VoiceDebugRow
              label="finalization"
              value={voiceFinalization.status}
            />
            <VoiceDebugRow
              label="finalization detail"
              value={voiceFinalization.detail}
            />
            <VoiceDebugRow
              label="finalization updated"
              value={formatVoiceTimestamp(voiceFinalization.updatedAt)}
            />
            <VoiceDebugRow
              label="transcript count"
              value={voiceTranscripts.length}
            />
            <VoiceDebugRow
              label="activity count"
              value={voiceActivities.length}
            />
            <VoiceDebugRow
              label="latest activity"
              value={
                latestActivity
                  ? `${latestActivity.label}: ${latestActivity.status}`
                  : null
              }
            />
            <VoiceDebugRow
              label="latest transcript"
              value={
                latestTranscript
                  ? `${latestTranscript.role}: ${clipVoiceText(latestTranscript.text, 120)}`
                  : null
              }
            />
          </div>
          <p className="mt-3 text-[11px] font-mono text-oc-text-dim">
            Developer-only readout. Enable with{" "}
            <span className="text-oc-teal-700">
              NEXT_PUBLIC_ENABLE_VOICE_DEBUG=true
            </span>
            . Tokens and raw credentials are intentionally not shown.
          </p>
        </details>
      )}
        </div>
      </div>
    </>
  );
}
