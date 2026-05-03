"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  RoomAudioRenderer,
  SessionProvider,
  useAgent,
  useSession,
  useSessionMessages,
} from "@livekit/components-react";
import { RoomEvent } from "livekit-client";
import {
  useSessionStore,
  type VoiceActivityEvent,
  type VoiceActivityName,
  type VoiceActivityStatus,
} from "@/lib/session";
import {
  getLiveKitVoiceFinalizationStatus,
  getMemorySessions,
  type MemorySession,
} from "@/lib/api";
import {
  createOpenCouchVoiceRoom,
  createOpenCouchVoiceTokenSource,
} from "@/lib/livekit-session";

const VOICE_FINALIZATION_STALE_AFTER_MS = 120_000;
const VOICE_ACTIVITY_TOPIC = "opencouch.voice_activity";

const VOICE_ACTIVITY_NAMES = new Set<VoiceActivityName>([
  "memory_saved",
  "memory_recall_updated",
  "memory_delete_pending",
  "memory_deleted",
  "factual_lookup",
  "crisis_resources_lookup",
  "exercise",
]);

const VOICE_ACTIVITY_STATUSES = new Set<VoiceActivityStatus>([
  "started",
  "completed",
  "failed",
  "pending",
  "cancelled",
]);

type SessionMessage = {
  type?: string;
  message: string;
  id: string;
  from?: {
    identity?: string;
  };
};

function transcriptRoleForMessage(
  message: SessionMessage,
  localIdentity: string | undefined
): "user" | "assistant" | null {
  if (message.type === "userTranscript") {
    return "user";
  }
  if (message.type === "agentTranscript") {
    return "assistant";
  }
  if (message.type === "chat") {
    return message.from?.identity === localIdentity ? "user" : "assistant";
  }
  return null;
}

function isVoiceFinalizationStale(updatedAt: string | null): boolean {
  if (!updatedAt) {
    return false;
  }

  const timestamp = Date.parse(updatedAt);
  if (Number.isNaN(timestamp)) {
    return false;
  }

  return Date.now() - timestamp > VOICE_FINALIZATION_STALE_AFTER_MS;
}

function latestSessionForThread(
  sessions: MemorySession[],
  threadId: string
): MemorySession | null {
  let latest: MemorySession | null = null;

  for (const session of sessions) {
    if (session.session_id !== threadId || !session.summary) {
      continue;
    }
    if (!latest || session.ended_at > latest.ended_at) {
      latest = session;
    }
  }

  return latest;
}

function createVoiceActivityId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function normalizeVoiceActivityText(
  value: unknown,
  fallback: string,
  limit: number
): string {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return fallback;
  }
  return normalized.slice(0, limit);
}

function parseVoiceActivityPayload(payload: Uint8Array): VoiceActivityEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(payload));
  } catch {
    return null;
  }

  if (typeof value !== "object" || value === null) {
    return null;
  }

  const record = value as Record<string, unknown>;
  if (record.type !== "voice_activity") {
    return null;
  }

  const activity = record.activity;
  const status = record.status;
  if (
    typeof activity !== "string" ||
    !VOICE_ACTIVITY_NAMES.has(activity as VoiceActivityName) ||
    typeof status !== "string" ||
    !VOICE_ACTIVITY_STATUSES.has(status as VoiceActivityStatus)
  ) {
    return null;
  }

  return {
    id: createVoiceActivityId(),
    activity: activity as VoiceActivityName,
    status: status as VoiceActivityStatus,
    label: normalizeVoiceActivityText(record.label, "Voice activity", 60),
    detail: normalizeVoiceActivityText(record.detail, "", 180),
    timestamp:
      typeof record.timestamp === "string"
        ? record.timestamp
        : new Date().toISOString(),
  };
}

function shouldRefreshMemoryForActivity(event: VoiceActivityEvent): boolean {
  return (
    event.status === "completed" &&
    (event.activity === "memory_saved" ||
      event.activity === "memory_deleted" ||
      event.activity === "memory_recall_updated")
  );
}

function VoiceSessionSync({ children }: { children: React.ReactNode }) {
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const assistantVoiceSelected = useSessionStore((s) => s.assistantVoiceSelected);
  const transcriptionLanguageSelected = useSessionStore(
    (s) => s.transcriptionLanguageSelected
  );
  const voiceFinalization = useSessionStore((s) => s.voiceFinalization);
  const setVoiceConnected = useSessionStore((s) => s.setVoiceConnected);
  const setVoiceAgentSpeaking = useSessionStore((s) => s.setVoiceAgentSpeaking);
  const setVoiceReadyToSpeak = useSessionStore((s) => s.setVoiceReadyToSpeak);
  const addVoiceTranscript = useSessionStore((s) => s.addVoiceTranscript);
  const addVoiceActivity = useSessionStore((s) => s.addVoiceActivity);
  const setVoiceFinalization = useSessionStore((s) => s.setVoiceFinalization);
  const clearVoiceFinalization = useSessionStore((s) => s.clearVoiceFinalization);
  const setVoiceSessionInfo = useSessionStore((s) => s.setVoiceSessionInfo);
  const setLastEndedSession = useSessionStore((s) => s.setLastEndedSession);
  const voiceSetRefs = useSessionStore((s) => s.voiceSetRefs);
  const bumpMemoryRefreshVersion = useSessionStore((s) => s.bumpMemoryRefreshVersion);

  const room = useMemo(() => createOpenCouchVoiceRoom(), []);
  const tokenSource = useMemo(
    () =>
      createOpenCouchVoiceTokenSource(
        userId || "voice-user",
        threadId,
        transcriptionLanguageSelected,
        sessionMode,
        assistantVoiceSelected,
        (token) => {
          setVoiceSessionInfo({
            roomName: token.room_name,
            identity: token.identity,
            memoryMode: token.memory_mode,
            assistantVoice: token.assistant_voice,
            serverUrl: token.server_url,
            connectedAt: new Date().toISOString(),
          });
        }
      ),
    [
      sessionMode,
      assistantVoiceSelected,
      setVoiceSessionInfo,
      threadId,
      transcriptionLanguageSelected,
      userId,
    ]
  );
  const session = useSession(tokenSource, { room });
  const agent = useAgent(session);
  const waitUntilCouldBeListening = agent.waitUntilCouldBeListening;
  const { messages } = useSessionMessages(session);
  const wasConnectedRef = useRef(session.isConnected);

  useEffect(() => {
    setVoiceConnected(session.isConnected);
  }, [session.isConnected, setVoiceConnected]);

  useEffect(() => {
    const wasConnected = wasConnectedRef.current;

    if (session.isConnected) {
      clearVoiceFinalization();
    } else if (wasConnected) {
      setVoiceFinalization({
        threadId,
        status: "in_progress",
        detail: "Saving session memory...",
        updatedAt: new Date().toISOString(),
      });
    }

    wasConnectedRef.current = session.isConnected;
  }, [
    clearVoiceFinalization,
    session.isConnected,
    setVoiceFinalization,
    threadId,
  ]);

  useEffect(() => {
    setVoiceAgentSpeaking(agent.state === "speaking");
  }, [agent.state, setVoiceAgentSpeaking]);

  useEffect(() => {
    if (!session.isConnected) {
      setVoiceReadyToSpeak(false);
      return;
    }

    const abortController = new AbortController();
    let settleTimer: number | null = null;

    const waitForReadiness = async () => {
      try {
        await waitUntilCouldBeListening(abortController.signal);
        if (abortController.signal.aborted) {
          return;
        }
        settleTimer = window.setTimeout(() => {
          if (!abortController.signal.aborted) {
            setVoiceReadyToSpeak(true);
          }
        }, 250);
      } catch {
        // Ignore aborted readiness waits during disconnects and reconnects.
      }
    };

    setVoiceReadyToSpeak(false);
    void waitForReadiness();

    return () => {
      abortController.abort();
      if (settleTimer !== null) {
        window.clearTimeout(settleTimer);
      }
    };
  }, [session.isConnected, setVoiceReadyToSpeak, waitUntilCouldBeListening]);

  useEffect(() => {
    voiceSetRefs({ room: session.room });
    return () => {
      voiceSetRefs({ room: null });
    };
  }, [session.room, voiceSetRefs]);

  useEffect(() => {
    const handleDataReceived = (
      payload: Uint8Array,
      _participant?: unknown,
      _kind?: unknown,
      topic?: string
    ) => {
      if (topic !== VOICE_ACTIVITY_TOPIC) {
        return;
      }

      const event = parseVoiceActivityPayload(payload);
      if (!event) {
        return;
      }

      addVoiceActivity(event);
      if (shouldRefreshMemoryForActivity(event)) {
        bumpMemoryRefreshVersion();
      }
    };

    session.room.on(RoomEvent.DataReceived, handleDataReceived);
    return () => {
      session.room.off(RoomEvent.DataReceived, handleDataReceived);
    };
  }, [addVoiceActivity, bumpMemoryRefreshVersion, session.room]);

  useEffect(() => {
    if (!session.isConnected) {
      return;
    }
    const localIdentity = session.room.localParticipant.identity;
    for (const message of messages) {
      const role = transcriptRoleForMessage(message, localIdentity);
      if (!role) continue;
      const text = message.message.trim();
      if (!text) continue;
      addVoiceTranscript({
        role,
        text,
        itemId: message.id,
      });
    }
  }, [addVoiceTranscript, messages, session.isConnected, session.room.localParticipant.identity]);

  useEffect(() => {
    const finalizationThreadId = voiceFinalization.threadId;

    if (
      session.isConnected ||
      voiceFinalization.status !== "in_progress" ||
      !finalizationThreadId ||
      finalizationThreadId !== threadId
    ) {
      return;
    }

    let cancelled = false;
    let nextPollId: number | null = null;

    const markFailed = (detail: string) => {
      setVoiceFinalization({
        threadId: finalizationThreadId,
        status: "failed",
        detail,
        updatedAt: new Date().toISOString(),
      });
    };

    const poll = async () => {
      if (isVoiceFinalizationStale(voiceFinalization.updatedAt)) {
        markFailed(
          "Timed out while waiting for session memory to finish saving."
        );
        return;
      }

      try {
        const status = await getLiveKitVoiceFinalizationStatus(finalizationThreadId);
        if (cancelled) {
          return;
        }

        if (status !== null) {
          if (
            status.status === "in_progress" &&
            isVoiceFinalizationStale(status.updated_at)
          ) {
            markFailed(
              "Timed out while waiting for session memory to finish saving."
            );
            return;
          }

          setVoiceFinalization({
            threadId: finalizationThreadId,
            status: status.status,
            detail: status.detail,
            updatedAt: status.updated_at,
          });

          if (status.status === "completed") {
            bumpMemoryRefreshVersion();
            try {
              const sessions = await getMemorySessions(
                finalizationThreadId,
                userId || undefined
              );
              const savedSession = latestSessionForThread(
                sessions,
                finalizationThreadId
              );
              setLastEndedSession(
                savedSession
                  ? {
                      threadId: finalizationThreadId,
                      summary: savedSession.summary,
                      themes: savedSession.themes,
                      mood_opened: savedSession.mood_opened,
                      mood_closed: savedSession.mood_closed,
                      turn_count: savedSession.turn_count,
                    }
                  : {
                      threadId: finalizationThreadId,
                      summary: null,
                      detail: status.detail || "Voice session ended.",
                    }
              );
            } catch {
              setLastEndedSession({
                threadId: finalizationThreadId,
                summary: null,
                detail:
                  "Voice session ended, but the saved summary could not be loaded.",
              });
            }
          }

          if (status.status !== "in_progress") {
            return;
          }
        }
      } catch {
        // Keep polling. The local in-progress state is the source of truth
        // while the worker finishes its background finalization.
      }

      if (!cancelled) {
        nextPollId = window.setTimeout(() => {
          void poll();
        }, 1000);
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (nextPollId !== null) {
        window.clearTimeout(nextPollId);
      }
    };
  }, [
    bumpMemoryRefreshVersion,
    session.isConnected,
    setLastEndedSession,
    setVoiceFinalization,
    threadId,
    userId,
    voiceFinalization.status,
    voiceFinalization.threadId,
    voiceFinalization.updatedAt,
  ]);

  return (
    <SessionProvider session={session}>
      {children}
      <RoomAudioRenderer room={session.room} />
    </SessionProvider>
  );
}

export function VoiceSessionProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <VoiceSessionSync>{children}</VoiceSessionSync>;
}
