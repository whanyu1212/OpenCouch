import {
  checkRealtimeVoiceSafety,
  createRealtimeVoiceSession,
  endRealtimeVoiceSession,
  executeRealtimeVoiceTool,
  recordRealtimeVoiceTurn,
  type RealtimeVoiceEndSessionResponse,
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
import {
  buildRealtimeVoiceTurnRecordInput,
  RealtimeVoiceTurnTracker,
  type RealtimeVoiceTrackedTurn,
} from "./realtime-voice-turn-record";
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
  onFinalizationFailed?: (error: Error) => void;
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
const FOLLOW_UP_RESPONSE_TIMEOUT_MS = 10_000;

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
  const pendingToolExecutions = new Set<Promise<void>>();
  const transcriptLog: TranscriptLogEntry[] = [];
  const turnTracker = new RealtimeVoiceTurnTracker();
  const userTranscriptDrafts = new Map<string, string>();
  const userTranscriptEvidenceWaiters: UserTranscriptEvidenceWaiter[] = [];
  const followUpResponseTimeouts = new Map<
    string,
    ReturnType<typeof setTimeout>
  >();
  let latestUserTranscriptDraft = "";
  let priorMessageCount = 0;
  let pendingTurnRecording: Promise<void> | null = null;

  const setStatus = (status: RealtimeVoiceConnectionStatus) => {
    options.onStatus?.(status);
  };

  const markTransportClosed = () => {
    options.onAgentSpeaking?.(false);
    options.onReadyToSpeak?.(false);
    if (!disconnecting) {
      void disconnect().catch((error) => {
        const normalized =
          error instanceof Error
            ? error
            : new Error("Could not finalize disconnected Realtime voice session.");
        options.onError?.(normalized);
        options.onFinalizationFailed?.(normalized);
        setStatus("disconnected");
      });
    }
  };

  function disconnect({
    finalize = true,
  }: { finalize?: boolean } = {}): Promise<void> {
    return disconnectCoordinator.disconnect(async () => {
      disconnecting = true;
      try {
        if (finalize && !finalized) setStatus("finalizing");
        dataChannel?.close();
        peerConnection?.close();
        mediaStream?.getTracks().forEach((track) => track.stop());
        options.audioElement.srcObject = null;
        await Promise.allSettled([...pendingToolExecutions]);
        clearFollowUpResponseTimeouts();
        turnTracker.transportClosed();

        if (finalize && !finalized) {
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
  }

  try {
    setStatus("requesting_session");
    const session = await createRealtimeVoiceSession({
      threadId: options.threadId,
      userId: options.userId,
      memoryMode: options.memoryMode,
      assistantVoice: options.assistantVoice,
    });
    priorMessageCount = session.message_count;
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

    if (parsed.type === "input_audio_buffer.committed" && parsed.userItemId) {
      turnTracker.userInputCommitted(parsed.userItemId);
    }
    if (parsed.type === "response.created" && parsed.responseId) {
      if (parsed.responseRequestId) {
        clearFollowUpResponseTimeout(parsed.responseRequestId);
      }
      turnTracker.responseCreated(parsed.responseId, parsed.responseRequestId);
    }

    if (parsed.agentSpeaking !== undefined) {
      options.onAgentSpeaking?.(parsed.agentSpeaking);
    }
    if (parsed.readyToSpeak !== undefined) {
      options.onReadyToSpeak?.(parsed.readyToSpeak);
    }
    if (parsed.errorMessage) {
      options.onError?.(new Error(parsed.errorMessage));
    }
    if (parsed.errorEventId) {
      releaseFollowUpResponseExpectation(parsed.errorEventId);
    }
    if (parsed.transcript) {
      handleTranscriptUpdate(parsed.transcript);
      options.onTranscript?.(parsed.transcript);
    }

    for (const call of parsed.functionCalls) {
      if (turnTracker.isResponseIgnored(call.responseId)) continue;
      const execution = executeToolCall(call);
      pendingToolExecutions.add(execution);
      try {
        await execution;
      } finally {
        pendingToolExecutions.delete(execution);
      }
    }
    if (parsed.type === "response.done" && parsed.responseId) {
      turnTracker.responseFinished(parsed.responseId);
      void maybeRecordTurn().catch(() => undefined);
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
      const turn = turnTracker.addFinalUserTranscript({
        itemId: update.itemId,
        text,
      });
      if (turn.isNew) {
        void checkRealtimeVoiceSafety({
          threadId: options.threadId,
          userId: options.userId,
          memoryMode: options.memoryMode,
          clientTurnId: turn.clientTurnId,
          userText: turn.userText,
          priorMessageCount,
          pendingPriorTranscript: turnTracker.priorTranscriptForTurn(
            turn.clientTurnId
          ),
        }).catch(() => undefined);
      }
      resolveUserTranscriptEvidenceWaiters({ final: true });
    } else {
      turnTracker.addFinalAssistantTranscript({
        responseId: update.responseId,
        text,
      });
    }

    void maybeRecordTurn().catch(() => undefined);
  }

  function maybeRecordTurn(): Promise<void> {
    if (pendingTurnRecording) return pendingTurnRecording;
    const firstTurn = turnTracker.markNextRecordableTurn();
    if (!firstTurn) {
      return Promise.resolve();
    }

    const recording = (async () => {
      let turn: RealtimeVoiceTrackedTurn | null = firstTurn;
      while (turn) {
        let response: RealtimeVoiceTurnRecordResponse;
        try {
          response = await recordRealtimeVoiceTurn(
            buildRealtimeVoiceTurnRecordInput({
              threadId: options.threadId,
              userId: options.userId,
              clientTurnId: turn.clientTurnId,
              userText: turn.userText,
              assistantText: turn.assistantText,
              memoryMode: options.memoryMode,
              toolCalls: turn.toolCalls,
            })
          );
        } catch (error) {
          turnTracker.recordingFailed(turn.clientTurnId);
          const normalized =
            error instanceof Error
              ? error
              : new Error("Could not record Realtime voice turn.");
          options.onError?.(normalized);
          throw normalized;
        }
        turnTracker.recordingSucceeded(turn.clientTurnId);
        priorMessageCount = response.message_count;
        options.onTurnRecorded?.(response);
        turn = turnTracker.markNextRecordableTurn();
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
    const clientTurnId = turnTracker.correlateToolCall(call.responseId);
    turnTracker.toolCallStarted(clientTurnId);

    options.onToolEvent?.({
      callId: call.callId,
      name: call.name,
      status: "started",
    });

    try {
      const currentUserMessage = await currentUserMessageForToolCall(
        call,
        clientTurnId
      );
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
        turnTracker.addToolResult(clientTurnId, {
          tool_name: call.name,
          status: "completed",
          output: result.output,
        });
      }
      if (disconnecting || dataChannel.readyState !== "open") return;
      dataChannel.send(
        serializeRealtimeEvent(buildFunctionCallOutputEvent(call.callId, result.output))
      );
      if (shouldCreateResponseAfterRealtimeVoiceTool(call.name)) {
        sendFollowUpResponse(clientTurnId);
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
        turnTracker.addToolResult(clientTurnId, {
          tool_name: call.name,
          status: "failed",
          output: {},
          error: message,
        });
      }
      if (!disconnecting && dataChannel.readyState === "open") {
        dataChannel.send(
          serializeRealtimeEvent(
            buildFunctionCallOutputEvent(call.callId, { error: message })
          )
        );
        if (shouldCreateResponseAfterRealtimeVoiceTool(call.name)) {
          sendFollowUpResponse(clientTurnId);
        }
      }
      options.onToolEvent?.({
        callId: call.callId,
        name: call.name,
        status: "failed",
        detail: message,
      });
      options.onError?.(new Error(message));
    } finally {
      turnTracker.toolCallFinished(clientTurnId);
      void maybeRecordTurn().catch(() => undefined);
    }
  }

  function sendFollowUpResponse(clientTurnId: string | undefined): void {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    if (!clientTurnId) {
      dataChannel.send(serializeRealtimeEvent(buildResponseCreateEvent()));
      return;
    }

    const requestEventId = `response-create-${globalThis.crypto.randomUUID()}`;
    dataChannel.send(
      serializeRealtimeEvent(buildResponseCreateEvent(null, requestEventId))
    );
    if (!turnTracker.expectNextResponseForTurn(clientTurnId, requestEventId)) return;

    followUpResponseTimeouts.set(
      requestEventId,
      setTimeout(() => {
        releaseFollowUpResponseExpectation(requestEventId);
      }, FOLLOW_UP_RESPONSE_TIMEOUT_MS)
    );
  }

  function releaseFollowUpResponseExpectation(requestEventId: string): void {
    clearFollowUpResponseTimeout(requestEventId);
    if (turnTracker.failExpectedResponse(requestEventId)) {
      void maybeRecordTurn().catch(() => undefined);
    }
  }

  function clearFollowUpResponseTimeout(requestEventId: string): void {
    const timeout = followUpResponseTimeouts.get(requestEventId);
    if (timeout) clearTimeout(timeout);
    followUpResponseTimeouts.delete(requestEventId);
  }

  function clearFollowUpResponseTimeouts(): void {
    for (const timeout of followUpResponseTimeouts.values()) {
      clearTimeout(timeout);
    }
    followUpResponseTimeouts.clear();
  }

  function latestUserTranscriptEvidence(): string {
    return turnTracker.latestUserText().trim() || latestUserTranscriptDraft.trim();
  }

  async function currentUserMessageForToolCall(
    call: RealtimeFunctionCall,
    clientTurnId?: string
  ): Promise<string> {
    const correlatedUserText = turnTracker.userTextForTurn(clientTurnId).trim();
    if (!shouldWaitForRealtimeVoiceTranscriptEvidence(call.name)) {
      return correlatedUserText || latestUserTranscriptEvidence();
    }

    const quote = readRealtimeVoiceUserQuote(call.arguments);
    const evidence = correlatedUserText || latestUserTranscriptEvidence();
    if (!quote || realtimeVoiceEvidenceMatchesUserQuote({ evidence, userQuote: quote })) {
      return evidence;
    }
    if (correlatedUserText) return correlatedUserText;
    if (turnTracker.latestUserText().trim()) return turnTracker.latestUserText();
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
