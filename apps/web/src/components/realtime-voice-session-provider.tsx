"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  connectRealtimeVoiceSession,
  type RealtimeVoiceConnectionStatus,
  type RealtimeVoiceSessionHandle,
  type RealtimeVoiceToolEvent,
} from "@/lib/realtime-voice-session";
import {
  buildEndedSessionResult,
  useSessionStore,
  type EndedSessionResult,
  type VoiceActivityEvent,
  type VoiceActivityName,
} from "@/lib/session";
import type {
  RealtimeVoiceSessionResponse,
} from "@/lib/api";

const REALTIME_SERVER_URL = "https://api.openai.com/v1/realtime/calls";

type RealtimeVoiceDisconnectOptions = {
  finalize?: boolean;
};

type RealtimeVoiceSessionContextValue = {
  status: RealtimeVoiceConnectionStatus;
  session: RealtimeVoiceSessionResponse | null;
  connected: boolean;
  busy: boolean;
  connect: () => Promise<void>;
  disconnect: (options?: RealtimeVoiceDisconnectOptions) => Promise<void>;
};

const RealtimeVoiceSessionContext =
  createContext<RealtimeVoiceSessionContextValue | null>(null);

type ToolActivityDefinition = {
  activity: VoiceActivityName;
  label: string;
};

const TOOL_ACTIVITY_BY_NAME: Record<string, ToolActivityDefinition> = {
  show_memory_status: {
    activity: "memory_recall_updated",
    label: "Memory status",
  },
  show_saved_memory: {
    activity: "memory_recall_updated",
    label: "Saved memory",
  },
  set_proactive_memory_recall: {
    activity: "memory_recall_updated",
    label: "Memory recall",
  },
  save_response_preference: {
    activity: "memory_saved",
    label: "Preference saved",
  },
  prepare_memory_deletion_by_index: {
    activity: "memory_delete_pending",
    label: "Memory deletion",
  },
  prepare_memory_deletion_by_query: {
    activity: "memory_delete_pending",
    label: "Memory deletion",
  },
  confirm_memory_deletion: {
    activity: "memory_deleted",
    label: "Memory deleted",
  },
  cancel_memory_deletion: {
    activity: "memory_delete_pending",
    label: "Memory deletion cancelled",
  },
  answer_grounded_lookup: {
    activity: "factual_lookup",
    label: "Grounded lookup",
  },
  lookup_crisis_resources: {
    activity: "crisis_resources_lookup",
    label: "Crisis resources",
  },
  list_guided_exercise_skills: {
    activity: "exercise",
    label: "Exercise options",
  },
  load_guided_exercise_skill: {
    activity: "exercise",
    label: "Exercise skill",
  },
  record_guided_exercise_progress: {
    activity: "exercise",
    label: "Exercise progress",
  },
  load_therapeutic_response_skill: {
    activity: "therapeutic_skill",
    label: "Response skill",
  },
};

const MEMORY_REFRESH_TOOL_NAMES = new Set([
  "save_response_preference",
  "set_proactive_memory_recall",
  "confirm_memory_deletion",
]);

function createVoiceActivityId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function fallbackToolActivity(name: string): ToolActivityDefinition {
  return {
    activity: "therapeutic_skill",
    label: name.replace(/_/g, " "),
  };
}

function extractToolDetail(output: Record<string, unknown> | undefined): string {
  if (!output) return "";

  const detail = output.detail;
  if (typeof detail === "string") return detail;

  const responseText = output.response_text;
  if (typeof responseText === "string") return responseText;

  const groundedLookup = output.grounded_lookup;
  if (typeof groundedLookup === "object" && groundedLookup !== null) {
    const status = (groundedLookup as Record<string, unknown>).status;
    if (typeof status === "string") return `Lookup ${status}.`;
  }

  return "";
}

function voiceActivityFromToolEvent(
  event: RealtimeVoiceToolEvent
): VoiceActivityEvent {
  const definition = TOOL_ACTIVITY_BY_NAME[event.name] ?? fallbackToolActivity(event.name);
  return {
    id: createVoiceActivityId(),
    activity: definition.activity,
    status: event.status,
    label: definition.label,
    detail: event.detail || extractToolDetail(event.output),
    timestamp: new Date().toISOString(),
  };
}

function shouldRefreshMemoryForTool(event: RealtimeVoiceToolEvent): boolean {
  return event.status === "completed" && MEMORY_REFRESH_TOOL_NAMES.has(event.name);
}

function readRealtimeSessionVoice(
  session: RealtimeVoiceSessionResponse,
  fallback: string
): string {
  const config = session.session_config;
  const audio = config.audio;
  if (typeof audio !== "object" || audio === null) return fallback;
  const output = (audio as Record<string, unknown>).output;
  if (typeof output !== "object" || output === null) return fallback;
  const voice = (output as Record<string, unknown>).voice;
  return typeof voice === "string" && voice ? voice : fallback;
}

export function useRealtimeVoiceSession(): RealtimeVoiceSessionContextValue {
  const context = useContext(RealtimeVoiceSessionContext);
  if (!context) {
    throw new Error(
      "useRealtimeVoiceSession must be used within RealtimeVoiceSessionProvider."
    );
  }
  return context;
}

export function RealtimeVoiceSessionProvider({
  children,
}: {
  children: ReactNode;
}) {
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const assistantVoiceSelected = useSessionStore((s) => s.assistantVoiceSelected);
  const setVoiceConnected = useSessionStore((s) => s.setVoiceConnected);
  const setVoiceAgentSpeaking = useSessionStore((s) => s.setVoiceAgentSpeaking);
  const setVoiceReadyToSpeak = useSessionStore((s) => s.setVoiceReadyToSpeak);
  const addVoiceTranscript = useSessionStore((s) => s.addVoiceTranscript);
  const addVoiceActivity = useSessionStore((s) => s.addVoiceActivity);
  const clearVoiceActivities = useSessionStore((s) => s.clearVoiceActivities);
  const clearVoiceTranscripts = useSessionStore((s) => s.clearVoiceTranscripts);
  const setVoiceFinalization = useSessionStore((s) => s.setVoiceFinalization);
  const clearVoiceFinalization = useSessionStore((s) => s.clearVoiceFinalization);
  const setVoiceSessionInfo = useSessionStore((s) => s.setVoiceSessionInfo);
  const setVoiceError = useSessionStore((s) => s.setVoiceError);
  const setLastEndedSession = useSessionStore((s) => s.setLastEndedSession);
  const voiceSetRefs = useSessionStore((s) => s.voiceSetRefs);
  const bumpMemoryRefreshVersion = useSessionStore((s) => s.bumpMemoryRefreshVersion);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const handleRef = useRef<RealtimeVoiceSessionHandle | null>(null);
  const [status, setStatus] =
    useState<RealtimeVoiceConnectionStatus>("disconnected");
  const [session, setSession] = useState<RealtimeVoiceSessionResponse | null>(null);

  const markFinalizationFailed = useCallback(
    (detail: string) => {
      setVoiceFinalization({
        threadId,
        status: "failed",
        detail,
        updatedAt: new Date().toISOString(),
      });
    },
    [setVoiceFinalization, threadId]
  );

  const handleEnded = useCallback(
    (response: EndedSessionResult) => {
      setVoiceFinalization({
        threadId: response.threadId,
        status: "completed",
        detail: response.detail || "Voice session ended.",
        updatedAt: new Date().toISOString(),
      });
      setLastEndedSession(response);
      if (sessionMode === "persistent" && response.summary) {
        bumpMemoryRefreshVersion();
      }
    },
    [bumpMemoryRefreshVersion, sessionMode, setLastEndedSession, setVoiceFinalization]
  );

  const disconnect = useCallback(
    async ({ finalize = true }: RealtimeVoiceDisconnectOptions = {}) => {
      const handle = handleRef.current;
      if (!handle) {
        setStatus("disconnected");
        setVoiceConnected(false);
        setVoiceAgentSpeaking(false);
        setVoiceReadyToSpeak(false);
        return;
      }

      if (finalize) {
        setVoiceFinalization({
          threadId,
          status: "in_progress",
          detail:
            sessionMode === "incognito"
              ? "Ending incognito voice session..."
              : "Saving session memory...",
          updatedAt: new Date().toISOString(),
        });
      }

      try {
        await handle.disconnect({ finalize });
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Voice session ended, but finalization failed.";
        setVoiceError(message);
        if (finalize) {
          markFinalizationFailed(message);
        }
      } finally {
        handleRef.current = null;
        voiceSetRefs({ connection: null });
        setVoiceConnected(false);
        setVoiceAgentSpeaking(false);
        setVoiceReadyToSpeak(false);
        setStatus("disconnected");
      }
    },
    [
      markFinalizationFailed,
      sessionMode,
      setVoiceAgentSpeaking,
      setVoiceConnected,
      setVoiceError,
      setVoiceFinalization,
      setVoiceReadyToSpeak,
      threadId,
      voiceSetRefs,
    ]
  );

  const connect = useCallback(async () => {
    if (handleRef.current) return;
    const audioElement = audioRef.current;
    if (!audioElement) {
      throw new Error("Realtime audio output is not ready.");
    }

    setVoiceError(null);
    clearVoiceFinalization();
    clearVoiceActivities();
    clearVoiceTranscripts();
    setLastEndedSession(null);
    setSession(null);
    setVoiceConnected(false);
    setVoiceAgentSpeaking(false);
    setVoiceReadyToSpeak(false);

    try {
      const handle = await connectRealtimeVoiceSession({
        threadId,
        userId: userId || undefined,
        memoryMode: sessionMode,
        assistantVoice: assistantVoiceSelected,
        audioElement,
        onStatus: (nextStatus) => {
          setStatus(nextStatus);
          setVoiceConnected(nextStatus === "connected");
        },
        onSession: (nextSession) => {
          setSession(nextSession);
          setVoiceSessionInfo({
            roomName: nextSession.thread_id,
            identity: nextSession.user_id || "incognito",
            memoryMode: nextSession.memory_mode,
            assistantVoice: readRealtimeSessionVoice(
              nextSession,
              assistantVoiceSelected
            ),
            serverUrl: REALTIME_SERVER_URL,
            connectedAt: new Date().toISOString(),
          });
        },
        onTranscript: (update) => {
          addVoiceTranscript({
            role: update.role,
            text: update.text,
            itemId: update.itemId,
          });
        },
        onToolEvent: (event) => {
          addVoiceActivity(voiceActivityFromToolEvent(event));
          if (sessionMode === "persistent" && shouldRefreshMemoryForTool(event)) {
            bumpMemoryRefreshVersion();
          }
        },
        onTurnRecorded: (response) => {
          if (sessionMode === "persistent" && response.recorded) {
            bumpMemoryRefreshVersion();
          }
        },
        onEnded: (response) => {
          handleEnded(
            buildEndedSessionResult({
              threadId,
              result: response,
              modality: "voice",
            })
          );
        },
        onAgentSpeaking: setVoiceAgentSpeaking,
        onReadyToSpeak: setVoiceReadyToSpeak,
        onError: (error) => setVoiceError(error.message),
      });

      handleRef.current = handle;
      voiceSetRefs({ connection: handle });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to start Realtime voice session.";
      setVoiceError(message);
      setVoiceConnected(false);
      setVoiceAgentSpeaking(false);
      setVoiceReadyToSpeak(false);
      setStatus("disconnected");
      throw error;
    }
  }, [
    addVoiceActivity,
    addVoiceTranscript,
    assistantVoiceSelected,
    bumpMemoryRefreshVersion,
    clearVoiceActivities,
    clearVoiceFinalization,
    clearVoiceTranscripts,
    handleEnded,
    sessionMode,
    setLastEndedSession,
    setVoiceAgentSpeaking,
    setVoiceConnected,
    setVoiceError,
    setVoiceReadyToSpeak,
    setVoiceSessionInfo,
    threadId,
    userId,
    voiceSetRefs,
  ]);

  useEffect(() => {
    return () => {
      const handle = handleRef.current;
      if (!handle) return;
      handleRef.current = null;
      voiceSetRefs({ connection: null });
      void handle.disconnect({ finalize: false });
    };
  }, [voiceSetRefs]);

  const value = useMemo<RealtimeVoiceSessionContextValue>(
    () => ({
      status,
      session,
      connected: status === "connected",
      busy:
        status === "requesting_session" ||
        status === "requesting_microphone" ||
        status === "connecting" ||
        status === "finalizing",
      connect,
      disconnect,
    }),
    [connect, disconnect, session, status]
  );

  return (
    <RealtimeVoiceSessionContext.Provider value={value}>
      {children}
      <audio ref={audioRef} autoPlay playsInline className="hidden" />
    </RealtimeVoiceSessionContext.Provider>
  );
}
