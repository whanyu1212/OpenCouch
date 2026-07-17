import {
  createRealtimeVoiceSession,
  endRealtimeVoiceSession,
  executeRealtimeVoiceTool,
  recordRealtimeVoiceTurn,
  type RealtimeVoiceEndSessionResponse,
  type RealtimeVoiceRecordedToolCall,
  type RealtimeVoiceSessionResponse,
  type RealtimeVoiceTurnRecordResponse,
  type AssistantVoiceOption,
  type VoiceMemoryMode,
} from "./api";
import {
  buildFunctionCallOutputEvent,
  buildResponseCreateEvent,
  parseRealtimeServerEvent,
  serializeRealtimeEvent,
  type ParsedRealtimeServerEvent,
  type RealtimeFunctionCall,
  type RealtimeTranscriptUpdate,
} from "./realtime-voice-events";
import { buildRealtimeVoiceTurnRecordInput } from "./realtime-voice-turn-record";
import {
  finalizeAfterPendingRealtimeVoiceTurn,
  onRealtimeVoiceTurnRecordingSettled,
  RealtimeVoiceDisconnectCoordinator,
} from "./realtime-voice-finalization";
import {
  readRealtimeVoiceUserQuote,
  realtimeVoiceEvidenceMatchesUserQuote,
  shouldCreateResponseAfterRealtimeVoiceTool,
  shouldRecordRealtimeVoiceToolCall,
  shouldWaitForRealtimeVoiceTranscriptEvidence,
} from "./realtime-voice-tool-flow";

const REALTIME_WEBRTC_URL = "https://api.openai.com/v1/realtime/calls";

export type RealtimeVoiceConnectionStatus =
  | "requesting_session"
  | "requesting_microphone"
  | "connecting"
  | "connected"
  | "finalizing"
  | "disconnected";

export interface RealtimeVoiceToolEvent {
  callId: string;
  name: string;
  status: "started" | "completed" | "failed";
  detail?: string;
  output?: Record<string, unknown>;
}

export interface RealtimeVoiceSessionOptions {
  threadId: string;
  userId?: string;
  memoryMode: VoiceMemoryMode;
  assistantVoice?: AssistantVoiceOption;
  audioElement: HTMLAudioElement;
  onStatus?: (status: RealtimeVoiceConnectionStatus) => void;
  onSession?: (session: RealtimeVoiceSessionResponse) => void;
  onRawEvent?: (event: Record<string, unknown>) => void;
  onParsedEvent?: (event: ParsedRealtimeServerEvent) => void;
  onTranscript?: (update: RealtimeTranscriptUpdate) => void;
  onToolEvent?: (event: RealtimeVoiceToolEvent) => void;
  onTurnRecorded?: (response: RealtimeVoiceTurnRecordResponse) => void;
  onEnded?: (response: RealtimeVoiceEndSessionResponse) => void;
  onAgentSpeaking?: (speaking: boolean) => void;
  onReadyToSpeak?: (ready: boolean) => void;
  onError?: (error: Error) => void;
}

export interface RealtimeVoiceSessionHandle {
  session: RealtimeVoiceSessionResponse;
  sendClientEvent: (event: Record<string, unknown>) => void;
  disconnect: (options?: { finalize?: boolean }) => Promise<void>;
}

type TranscriptLogEntry = {
  role: "user" | "assistant";
  content: string;
  item_id?: string;
};

type UserTranscriptEvidenceWaiter = {
  quote: string;
  resolve: (evidence: string) => void;
  timeout: ReturnType<typeof setTimeout>;
};

const USER_TRANSCRIPT_EVIDENCE_TIMEOUT_MS = 2500;

export async function connectRealtimeVoiceSession(
  options: RealtimeVoiceSessionOptions
): Promise<RealtimeVoiceSessionHandle> {
  let peerConnection: RTCPeerConnection | null = null;
  let dataChannel: RTCDataChannel | null = null;
  let mediaStream: MediaStream | null = null;
  let finalized = false;
  let disconnecting = false;
  const disconnectCoordinator = new RealtimeVoiceDisconnectCoordinator();

  const handledCallIds = new Set<string>();
  const transcriptLog: TranscriptLogEntry[] = [];
  const completedToolCalls: RealtimeVoiceRecordedToolCall[] = [];
  const userTranscriptDrafts = new Map<string, string>();
  const userTranscriptEvidenceWaiters: UserTranscriptEvidenceWaiter[] = [];
  let pendingUserText = "";
  let pendingAssistantText = "";
  let latestUserTranscriptDraft = "";
  let pendingTurnRecording: Promise<void> | null = null;

  const setStatus = (status: RealtimeVoiceConnectionStatus) => {
    options.onStatus?.(status);
  };

  const markTransportClosed = () => {
    options.onAgentSpeaking?.(false);
    options.onReadyToSpeak?.(false);
    if (!disconnecting) {
      setStatus("disconnected");
    }
  };

  const disconnect = ({
    finalize = true,
  }: { finalize?: boolean } = {}): Promise<void> =>
    disconnectCoordinator.disconnect(async () => {
      disconnecting = true;
      try {
        dataChannel?.close();
        peerConnection?.close();
        mediaStream?.getTracks().forEach((track) => track.stop());
        options.audioElement.srcObject = null;

        if (finalize && !finalized) {
          setStatus("finalizing");
          const response = await finalizeAfterPendingRealtimeVoiceTurn(
            maybeRecordTurn(),
            () => endRealtimeVoiceSession(options.threadId, options.memoryMode)
          );
          finalized = true;
          options.onEnded?.(response);
        }

        options.onAgentSpeaking?.(false);
        options.onReadyToSpeak?.(false);
        setStatus("disconnected");
      } finally {
        disconnecting = false;
      }
    });

  try {
    setStatus("requesting_session");
    const session = await createRealtimeVoiceSession({
      threadId: options.threadId,
      userId: options.userId,
      memoryMode: options.memoryMode,
      assistantVoice: options.assistantVoice,
    });
    options.onSession?.(session);

    setStatus("requesting_microphone");
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    peerConnection = new RTCPeerConnection();
    peerConnection.ontrack = (event) => {
      options.audioElement.srcObject = event.streams[0];
    };
    peerConnection.onconnectionstatechange = () => {
      if (
        peerConnection?.connectionState === "failed" ||
        peerConnection?.connectionState === "closed"
      ) {
        markTransportClosed();
      }
    };

    for (const track of mediaStream.getTracks()) {
      peerConnection.addTrack(track, mediaStream);
    }

    dataChannel = peerConnection.createDataChannel("oai-events");
    dataChannel.addEventListener("open", () => {
      options.onReadyToSpeak?.(true);
      setStatus("connected");
    });
    dataChannel.addEventListener("message", (event) => {
      void handleDataChannelMessage(event.data);
    });
    dataChannel.addEventListener("error", () => {
      options.onError?.(new Error("Realtime data channel error."));
    });
    dataChannel.addEventListener("close", markTransportClosed);

    setStatus("connecting");
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    const offerSdp = peerConnection.localDescription?.sdp;
    if (!offerSdp) {
      throw new Error("Realtime WebRTC offer did not include SDP.");
    }

    const sdpResponse = await fetch(REALTIME_WEBRTC_URL, {
      method: "POST",
      body: offerSdp,
      headers: {
        Authorization: `Bearer ${session.client_secret}`,
        "Content-Type": "application/sdp",
      },
    });
    if (!sdpResponse.ok) {
      throw new Error(`OpenAI Realtime SDP exchange failed: ${sdpResponse.status}`);
    }

    const answerSdp = await sdpResponse.text();
    await peerConnection.setRemoteDescription({
      type: "answer",
      sdp: answerSdp,
    });

    return {
      session,
      sendClientEvent(event: Record<string, unknown>) {
        if (!dataChannel || dataChannel.readyState !== "open") {
          throw new Error("Realtime data channel is not open.");
        }
        dataChannel.send(serializeRealtimeEvent(event));
      },
      disconnect,
    };
  } catch (error) {
    dataChannel?.close();
    peerConnection?.close();
    mediaStream?.getTracks().forEach((track) => track.stop());
    options.audioElement.srcObject = null;
    const normalized =
      error instanceof Error
        ? error
        : new Error("Could not connect Realtime voice session.");
    options.onError?.(normalized);
    setStatus("disconnected");
    throw normalized;
  }

  async function handleDataChannelMessage(rawData: unknown): Promise<void> {
    if (typeof rawData !== "string") {
      options.onError?.(new Error("Realtime data channel sent non-text data."));
      return;
    }

    let rawEvent: Record<string, unknown>;
    try {
      rawEvent = JSON.parse(rawData) as Record<string, unknown>;
    } catch {
      options.onError?.(new Error("Could not parse Realtime server event."));
      return;
    }

    options.onRawEvent?.(rawEvent);
    const parsed = parseRealtimeServerEvent(rawEvent);
    options.onParsedEvent?.(parsed);

    if (parsed.agentSpeaking !== undefined) {
      options.onAgentSpeaking?.(parsed.agentSpeaking);
    }
    if (parsed.readyToSpeak !== undefined) {
      options.onReadyToSpeak?.(parsed.readyToSpeak);
    }
    if (parsed.errorMessage) {
      options.onError?.(new Error(parsed.errorMessage));
    }
    if (parsed.transcript) {
      handleTranscriptUpdate(parsed.transcript);
      options.onTranscript?.(parsed.transcript);
    }

    for (const call of parsed.functionCalls) {
      await executeToolCall(call);
    }
  }

  function handleTranscriptUpdate(update: RealtimeTranscriptUpdate): void {
    const rawText = update.text;
    const text = rawText.trim();
    if (!text) return;

    if (update.role === "user" && !update.final) {
      const itemId = update.itemId || "__latest_user_audio__";
      const nextDraft = `${userTranscriptDrafts.get(itemId) || ""}${rawText}`.trim();
      userTranscriptDrafts.set(itemId, nextDraft);
      latestUserTranscriptDraft = nextDraft;
      resolveUserTranscriptEvidenceWaiters({ final: false });
      return;
    }

    if (!update.final) return;

    transcriptLog.push({
      role: update.role,
      content: text,
      item_id: update.itemId,
    });

    if (update.role === "user") {
      if (update.itemId) userTranscriptDrafts.delete(update.itemId);
      latestUserTranscriptDraft = "";
      pendingUserText = text;
      resolveUserTranscriptEvidenceWaiters({ final: true });
    } else {
      pendingAssistantText = text;
    }

    void maybeRecordTurn().catch(() => undefined);
  }

  function maybeRecordTurn(): Promise<void> {
    if (pendingTurnRecording) return pendingTurnRecording;
    if (!pendingUserText.trim() || !pendingAssistantText.trim()) {
      return Promise.resolve();
    }

    const recording = (async () => {
      while (pendingUserText.trim() && pendingAssistantText.trim()) {
        const userText = pendingUserText;
        const assistantText = pendingAssistantText;
        pendingUserText = "";
        pendingAssistantText = "";

        try {
          const toolCalls = completedToolCalls.splice(0);
          const response = await recordRealtimeVoiceTurn(
            buildRealtimeVoiceTurnRecordInput({
              threadId: options.threadId,
              userId: options.userId,
              userText,
              assistantText,
              memoryMode: options.memoryMode,
              toolCalls,
            })
          );
          options.onTurnRecorded?.(response);
        } catch (error) {
          pendingUserText = userText;
          pendingAssistantText = assistantText;
          const normalized =
            error instanceof Error
              ? error
              : new Error("Could not record Realtime voice turn.");
          options.onError?.(normalized);
          throw normalized;
        }
      }
    })();
    pendingTurnRecording = recording;
    onRealtimeVoiceTurnRecordingSettled(recording, (settledRecording) => {
      if (pendingTurnRecording === settledRecording) {
        pendingTurnRecording = null;
      }
    });
    return recording;
  }

  async function executeToolCall(call: RealtimeFunctionCall): Promise<void> {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    if (handledCallIds.has(call.callId)) return;
    handledCallIds.add(call.callId);

    options.onToolEvent?.({
      callId: call.callId,
      name: call.name,
      status: "started",
    });

    try {
      const currentUserMessage = await currentUserMessageForToolCall(call);
      const result = await executeRealtimeVoiceTool({
        threadId: options.threadId,
        userId: options.userId,
        currentUserMessage,
        transcript: transcriptLog,
        memoryMode: options.memoryMode,
        toolName: call.name,
        arguments: call.arguments,
      });
      if (shouldRecordRealtimeVoiceToolCall(call.name)) {
        completedToolCalls.push({
          tool_name: call.name,
          status: "completed",
          output: result.output,
        });
      }
      dataChannel.send(
        serializeRealtimeEvent(buildFunctionCallOutputEvent(call.callId, result.output))
      );
      if (shouldCreateResponseAfterRealtimeVoiceTool(call.name)) {
        dataChannel.send(serializeRealtimeEvent(buildResponseCreateEvent()));
      }
      options.onToolEvent?.({
        callId: call.callId,
        name: call.name,
        status: "completed",
        output: result.output,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Realtime voice tool failed.";
      if (shouldRecordRealtimeVoiceToolCall(call.name)) {
        completedToolCalls.push({
          tool_name: call.name,
          status: "failed",
          output: {},
          error: message,
        });
      }
      dataChannel.send(
        serializeRealtimeEvent(
          buildFunctionCallOutputEvent(call.callId, { error: message })
        )
      );
      if (shouldCreateResponseAfterRealtimeVoiceTool(call.name)) {
        dataChannel.send(serializeRealtimeEvent(buildResponseCreateEvent()));
      }
      options.onToolEvent?.({
        callId: call.callId,
        name: call.name,
        status: "failed",
        detail: message,
      });
      options.onError?.(new Error(message));
    }
  }

  function latestUserTranscriptEvidence(): string {
    return pendingUserText.trim() || latestUserTranscriptDraft.trim();
  }

  async function currentUserMessageForToolCall(
    call: RealtimeFunctionCall
  ): Promise<string> {
    if (!shouldWaitForRealtimeVoiceTranscriptEvidence(call.name)) {
      return pendingUserText;
    }

    const quote = readRealtimeVoiceUserQuote(call.arguments);
    const evidence = latestUserTranscriptEvidence();
    if (!quote || realtimeVoiceEvidenceMatchesUserQuote({ evidence, userQuote: quote })) {
      return evidence;
    }
    if (pendingUserText.trim()) return pendingUserText;
    return waitForUserTranscriptEvidence(quote);
  }

  function waitForUserTranscriptEvidence(quote: string): Promise<string> {
    return new Promise((resolve) => {
      const waiter: UserTranscriptEvidenceWaiter = {
        quote,
        resolve,
        timeout: setTimeout(() => {
          resolveUserTranscriptEvidenceWaiter(waiter);
        }, USER_TRANSCRIPT_EVIDENCE_TIMEOUT_MS),
      };
      userTranscriptEvidenceWaiters.push(waiter);
    });
  }

  function resolveUserTranscriptEvidenceWaiters({
    final,
  }: {
    final: boolean;
  }): void {
    for (const waiter of [...userTranscriptEvidenceWaiters]) {
      const evidence = latestUserTranscriptEvidence();
      if (
        final ||
        realtimeVoiceEvidenceMatchesUserQuote({
          evidence,
          userQuote: waiter.quote,
        })
      ) {
        resolveUserTranscriptEvidenceWaiter(waiter);
      }
    }
  }

  function resolveUserTranscriptEvidenceWaiter(
    waiter: UserTranscriptEvidenceWaiter
  ): void {
    const index = userTranscriptEvidenceWaiters.indexOf(waiter);
    if (index !== -1) userTranscriptEvidenceWaiters.splice(index, 1);
    clearTimeout(waiter.timeout);
    waiter.resolve(latestUserTranscriptEvidence());
  }
}
