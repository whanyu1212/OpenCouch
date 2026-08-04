import {
  ApiError,
  checkRealtimeVoiceSafety,
  createRealtimeVoiceSession,
  endRealtimeVoiceSession,
  executeRealtimeVoiceTool,
  heartbeatRealtimeVoiceRetryHandle,
  recordRealtimeVoiceTurn,
  type RealtimeVoiceEndSessionResponse,
  type RealtimeVoiceSessionResponse,
  type RealtimeVoiceSafetyCheckResponse,
  type RealtimeVoiceSafetyRequest,
  type RealtimeVoiceTurnRecordResponse,
  type AssistantVoiceOption,
  type VoiceMemoryMode,
} from "./api";
import {
  buildFunctionCallOutputEvent,
  buildResponseCancelEvent,
  buildResponseCreateEvent,
  parseRealtimeServerEvent,
  serializeRealtimeEvent,
  type ParsedRealtimeServerEvent,
  type RealtimeFunctionCall,
  type RealtimeTranscriptUpdate,
} from "./realtime-voice-events";
import {
  buildRealtimeVoiceTurnRecordInput,
  readLatestUserTranscriptDraft,
  RealtimeVoiceTurnTracker,
  type RealtimeVoiceSafetyInterruptionCleanup,
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

export interface RealtimeVoiceSafetyInterruption {
  response: RealtimeVoiceSafetyCheckResponse;
  request: Omit<RealtimeVoiceSafetyRequest, "signal">;
  cleanup: RealtimeVoiceSafetyInterruptionCleanup;
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
  onSafetyInterruption?: (event: RealtimeVoiceSafetyInterruption) => void;
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
const VOICE_TOOL_EXECUTION_TIMEOUT_MS = 30_000;
const VOICE_SAFETY_REQUEST_TIMEOUT_MS = 10_000;
const VOICE_PERSISTENCE_REQUEST_TIMEOUT_MS = 30_000;
const VOICE_RETRY_HANDLE_HEARTBEAT_MS = 5 * 60_000;

function createRetryHandleId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export async function connectRealtimeVoiceSession(
  options: RealtimeVoiceSessionOptions
): Promise<RealtimeVoiceSessionHandle> {
  let peerConnection: RTCPeerConnection | null = null;
  let dataChannel: RTCDataChannel | null = null;
  let mediaStream: MediaStream | null = null;
  let finalized = false;
  let disconnecting = false;
  let safetyInterrupted = false;
  let safetyGenerationActive = true;
  const safetyConnectionToken = Symbol("realtime-voice-connection");
  const retryHandleId = createRetryHandleId();
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
  let priorMessageCount = 0;
  let pendingTurnRecording: Promise<void> | null = null;
  let retryHandleHeartbeat: ReturnType<typeof setInterval> | null = null;
  const pendingSafetyRequests = new Map<
    string,
    {
      abortController: AbortController;
      timeout: ReturnType<typeof setTimeout>;
      token: symbol;
    }
  >();

  const setStatus = (status: RealtimeVoiceConnectionStatus) => {
    options.onStatus?.(status);
  };

  const startRetryHandleHeartbeat = () => {
    if (retryHandleHeartbeat) return;
    const heartbeat = () => {
      void heartbeatRealtimeVoiceRetryHandle({
        threadId: options.threadId,
        memoryMode: options.memoryMode,
        retryHandleId,
      }).catch(() => undefined);
    };
    heartbeat();
    retryHandleHeartbeat = setInterval(heartbeat, VOICE_RETRY_HANDLE_HEARTBEAT_MS);
  };

  const stopRetryHandleHeartbeat = () => {
    if (!retryHandleHeartbeat) return;
    clearInterval(retryHandleHeartbeat);
    retryHandleHeartbeat = null;
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

  function disconnect(
    {
      finalize = true,
    }: { finalize?: boolean } = {},
    safetyInterruption = false
  ): Promise<void> {
    return disconnectCoordinator.disconnect(async () => {
      disconnecting = true;
      try {
        if (!safetyInterruption) abortSafetyChecksFailOpen();
        if (finalize && !finalized) setStatus("finalizing");
        dataChannel?.close();
        peerConnection?.close();
        mediaStream?.getTracks().forEach((track) => track.stop());
        options.audioElement.pause();
        options.audioElement.srcObject = null;
        await Promise.allSettled([...pendingToolExecutions]);
        pendingToolExecutions.clear();
        clearFollowUpResponseTimeouts();
        turnTracker.transportClosed();

        if (finalize && !finalized) {
          const response = await finalizeAfterPendingRealtimeVoiceTurn(
            maybeRecordTurn(),
            endSessionWithTimeout
          );
          finalized = true;
          options.onEnded?.(response);
        }

        options.onAgentSpeaking?.(false);
        options.onReadyToSpeak?.(false);
        setStatus("disconnected");
      } finally {
        if (!finalize || finalized) stopRetryHandleHeartbeat();
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
    if (safetyInterrupted) return;

    if (parsed.type === "input_audio_buffer.committed" && parsed.userItemId) {
      turnTracker.userInputCommitted(parsed.userItemId);
    }
    if (parsed.failedUserTranscriptionItemId) {
      turnTracker.finishUserTranscription(parsed.failedUserTranscriptionItemId);
      void maybeRecordTurn().catch(() => undefined);
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
      if (handleTranscriptUpdate(parsed.transcript)) {
        options.onTranscript?.(parsed.transcript);
      }
    }

    for (const call of parsed.functionCalls) {
      if (safetyInterrupted) break;
      if (turnTracker.isResponseIgnored(call.responseId)) continue;
      const execution = executeToolCall(call);
      pendingToolExecutions.add(execution);
      try {
        await execution;
      } finally {
        pendingToolExecutions.delete(execution);
        void maybeRecordTurn().catch(() => undefined);
      }
    }
    if (parsed.responseTerminal && parsed.responseId) {
      turnTracker.responseFinished(parsed.responseId);
      void maybeRecordTurn().catch(() => undefined);
    }
  }

  function handleTranscriptUpdate(update: RealtimeTranscriptUpdate): boolean {
    if (
      update.role === "assistant" &&
      turnTracker.isResponseIgnored(update.responseId)
    ) {
      return false;
    }
    if (update.role === "assistant") {
      turnTracker.trackAssistantItem(update.responseId, update.itemId);
    }

    const rawText = update.text;
    const text = rawText.trim();
    if (!text) {
      if (update.role === "user" && update.final) {
        turnTracker.finishUserTranscription(update.itemId);
        void maybeRecordTurn().catch(() => undefined);
      }
      return true;
    }

    if (update.role === "user" && !update.final) {
      const itemId = update.itemId || "__latest_user_audio__";
      const nextDraft = `${userTranscriptDrafts.get(itemId) || ""}${rawText}`.trim();
      userTranscriptDrafts.set(itemId, nextDraft);
      resolveUserTranscriptEvidenceWaiters({ final: false });
      return true;
    }

    if (!update.final) return true;

    transcriptLog.push({
      role: update.role,
      content: text,
      item_id: update.itemId,
    });

    if (update.role === "user") {
      userTranscriptDrafts.delete(update.itemId || "__latest_user_audio__");
      const turn = turnTracker.addFinalUserTranscript({
        itemId: update.itemId,
        text,
      });
      if (turn.isNew) {
        turnTracker.markSafetyPending(turn.clientTurnId);
        beginSafetyCheck({
          threadId: options.threadId,
          userId: options.userId,
          memoryMode: options.memoryMode,
          clientTurnId: turn.clientTurnId,
          userText: turn.userText,
          priorMessageCount,
          pendingPriorTranscript: turnTracker.priorTranscriptForTurn(
            turn.clientTurnId
          ),
        });
      }
      resolveUserTranscriptEvidenceWaiters({ final: true });
    } else {
      turnTracker.addFinalAssistantTranscript({
        responseId: update.responseId,
        itemId: update.itemId,
        text,
      });
    }

    void maybeRecordTurn().catch(() => undefined);
    return true;
  }

  function maybeRecordTurn(): Promise<void> {
    if (safetyInterrupted && pendingToolExecutions.size > 0) {
      return Promise.resolve();
    }
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
          const abortController = new AbortController();
          const timeout = setTimeout(
            () => abortController.abort(),
            VOICE_PERSISTENCE_REQUEST_TIMEOUT_MS
          );
          try {
            response = await recordRealtimeVoiceTurn({
              ...buildRealtimeVoiceTurnRecordInput({
                threadId: options.threadId,
                userId: options.userId,
                clientTurnId: turn.clientTurnId,
                userText: turn.userText,
                assistantText: turn.assistantText,
                memoryMode: options.memoryMode,
                toolCalls: turn.toolCalls,
                outcome: turn.outcome,
                interruptionToken: turn.interruptionToken,
                retryHandleId,
              }),
              signal: abortController.signal,
            });
          } finally {
            clearTimeout(timeout);
          }
        } catch (error) {
          turnTracker.recordingFailed(turn.clientTurnId);
          if (!(error instanceof ApiError) || error.status >= 500) {
            startRetryHandleHeartbeat();
          }
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
      stopRetryHandleHeartbeat();
    })();
    pendingTurnRecording = recording;
    onRealtimeVoiceTurnRecordingSettled(recording, (settledRecording) => {
      if (pendingTurnRecording === settledRecording) {
        pendingTurnRecording = null;
      }
    });
    return recording;
  }

  function beginSafetyCheck(
    request: Omit<RealtimeVoiceSafetyRequest, "signal">
  ): void {
    const abortController = new AbortController();
    const timeout = setTimeout(
      () => abortController.abort(),
      VOICE_SAFETY_REQUEST_TIMEOUT_MS
    );
    const requestState = {
      abortController,
      timeout,
      token: safetyConnectionToken,
    };
    pendingSafetyRequests.set(request.clientTurnId, requestState);

    void checkRealtimeVoiceSafety({
      ...request,
      signal: abortController.signal,
    }).then(
      (response) => {
        if (!takeCurrentSafetyRequest(request.clientTurnId, requestState)) return;
        if (response.client_turn_id !== request.clientTurnId) {
          turnTracker.releaseSafetyCheck(request.clientTurnId);
          void maybeRecordTurn().catch(() => undefined);
          return;
        }
        if (response.action === "continue") {
          turnTracker.releaseSafetyCheck(request.clientTurnId);
          void maybeRecordTurn().catch(() => undefined);
          return;
        }

        turnTracker.attachLatestUserDraft(
          readLatestUserTranscriptDraft(userTranscriptDrafts)
        );
        const cleanup = turnTracker.interruptForSafety(
          request.clientTurnId,
          response.interruption_token || ""
        );
        if (!cleanup) return;
        safetyInterrupted = true;
        safetyGenerationActive = false;
        options.onSafetyInterruption?.({ response, request, cleanup });
        abortSafetyChecksFailOpen();
        options.audioElement.pause();
        options.audioElement.srcObject = null;
        options.onAgentSpeaking?.(false);
        options.onReadyToSpeak?.(false);
        if (dataChannel?.readyState === "open") {
          try {
            dataChannel.send(serializeRealtimeEvent(buildResponseCancelEvent()));
          } catch {
            // The terminal close below is the authoritative interruption.
          }
        }
        void disconnect({ finalize: true }, true).catch((error) => {
          const normalized =
            error instanceof Error
              ? error
              : new Error("Could not finalize interrupted Realtime voice session.");
          options.onError?.(normalized);
          options.onFinalizationFailed?.(normalized);
          setStatus("disconnected");
        });
      },
      () => {
        if (!takeCurrentSafetyRequest(request.clientTurnId, requestState)) return;
        turnTracker.releaseSafetyCheck(request.clientTurnId);
        void maybeRecordTurn().catch(() => undefined);
      }
    );
  }

  async function endSessionWithTimeout(): Promise<RealtimeVoiceEndSessionResponse> {
    const abortController = new AbortController();
    const timeout = setTimeout(
      () => abortController.abort(),
      VOICE_PERSISTENCE_REQUEST_TIMEOUT_MS
    );
    try {
      return await endRealtimeVoiceSession(
        options.threadId,
        options.memoryMode,
        abortController.signal
      );
    } finally {
      clearTimeout(timeout);
    }
  }

  function takeCurrentSafetyRequest(
    clientTurnId: string,
    requestState: {
      abortController: AbortController;
      timeout: ReturnType<typeof setTimeout>;
      token: symbol;
    }
  ): boolean {
    if (
      !safetyGenerationActive ||
      requestState.token !== safetyConnectionToken ||
      pendingSafetyRequests.get(clientTurnId) !== requestState
    ) {
      return false;
    }
    pendingSafetyRequests.delete(clientTurnId);
    clearTimeout(requestState.timeout);
    return true;
  }

  function abortSafetyChecksFailOpen(): void {
    safetyGenerationActive = false;
    for (const request of pendingSafetyRequests.values()) {
      clearTimeout(request.timeout);
      request.abortController.abort();
    }
    pendingSafetyRequests.clear();
    turnTracker.failOpenPendingSafetyChecks();
  }

  async function executeToolCall(call: RealtimeFunctionCall): Promise<void> {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    if (handledCallIds.has(call.callId)) return;
    handledCallIds.add(call.callId);
    const clientTurnId = turnTracker.correlateToolCall(call.responseId);
    turnTracker.toolCallStarted(clientTurnId);
    const abortController = new AbortController();
    const executionTimeout = setTimeout(
      () => abortController.abort(),
      VOICE_TOOL_EXECUTION_TIMEOUT_MS
    );

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
        clientTurnId,
        currentUserMessage,
        transcript: transcriptLog,
        memoryMode: options.memoryMode,
        toolName: call.name,
        arguments: call.arguments,
        signal: abortController.signal,
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
      clearTimeout(executionTimeout);
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
    return (
      turnTracker.latestUserText().trim() ||
      readLatestUserTranscriptDraft(userTranscriptDrafts).trim()
    );
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
