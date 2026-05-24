import {
  createRealtimeVoiceSession,
  endRealtimeVoiceSession,
  executeRealtimeVoiceTool,
  prepareRealtimeVoiceTurnPolicy,
  recordRealtimeVoiceTurn,
  type RealtimeVoiceEndSessionResponse,
  type RealtimeVoiceRecordedToolCall,
  type RealtimeVoiceSessionResponse,
  type RealtimeVoiceTurnRecordResponse,
  type RealtimeVoiceTurnPolicyResponse,
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
  onTurnPolicy?: (policy: RealtimeVoiceTurnPolicyResponse) => void;
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

export function buildRealtimeVoiceResponseCreateEvent(
  policy?: RealtimeVoiceTurnPolicyResponse | null
): Record<string, unknown> {
  return buildResponseCreateEvent(policy?.instructions);
}

export async function connectRealtimeVoiceSession(
  options: RealtimeVoiceSessionOptions
): Promise<RealtimeVoiceSessionHandle> {
  let peerConnection: RTCPeerConnection | null = null;
  let dataChannel: RTCDataChannel | null = null;
  let mediaStream: MediaStream | null = null;
  let disconnected = false;
  let finalized = false;

  const handledCallIds = new Set<string>();
  const transcriptLog: TranscriptLogEntry[] = [];
  const completedToolCalls: RealtimeVoiceRecordedToolCall[] = [];
  let pendingUserText = "";
  let pendingAssistantText = "";
  let latestPolicy: RealtimeVoiceTurnPolicyResponse | null = null;
  let pendingPolicyPromise: Promise<RealtimeVoiceTurnPolicyResponse | null> | null =
    null;
  let recordingTurn = false;

  const setStatus = (status: RealtimeVoiceConnectionStatus) => {
    options.onStatus?.(status);
  };

  const disconnect = async ({
    finalize = true,
  }: { finalize?: boolean } = {}): Promise<void> => {
    if (disconnected) return;
    disconnected = true;

    dataChannel?.close();
    peerConnection?.close();
    mediaStream?.getTracks().forEach((track) => track.stop());
    options.audioElement.srcObject = null;

    if (finalize && !finalized) {
      finalized = true;
      setStatus("finalizing");
      const response = await endRealtimeVoiceSession(
        options.threadId,
        options.memoryMode
      );
      options.onEnded?.(response);
    }

    options.onAgentSpeaking?.(false);
    options.onReadyToSpeak?.(false);
    setStatus("disconnected");
  };

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
        options.onAgentSpeaking?.(false);
        options.onReadyToSpeak?.(false);
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
    dataChannel.addEventListener("close", () => {
      options.onAgentSpeaking?.(false);
      options.onReadyToSpeak?.(false);
    });

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
    if (!update.final) return;

    const text = update.text.trim();
    if (!text) return;

    transcriptLog.push({
      role: update.role,
      content: text,
      item_id: update.itemId,
    });

    if (update.role === "user") {
      pendingUserText = text;
      latestPolicy = null;
      pendingPolicyPromise = prepareTurnPolicy(text);
    } else {
      pendingAssistantText = text;
    }

    void maybeRecordTurn();
  }

  async function maybeRecordTurn(): Promise<void> {
    if (recordingTurn) return;
    if (!pendingUserText.trim() || !pendingAssistantText.trim()) return;

    const userText = pendingUserText;
    const assistantText = pendingAssistantText;
    pendingUserText = "";
    pendingAssistantText = "";
    recordingTurn = true;

    try {
      const policy = pendingPolicyPromise
        ? await pendingPolicyPromise
        : latestPolicy;
      const toolCalls = completedToolCalls.splice(0);
      const response = await recordRealtimeVoiceTurn(
        buildRealtimeVoiceTurnRecordInput({
          threadId: options.threadId,
          userId: options.userId,
          userText,
          assistantText,
          memoryMode: options.memoryMode,
          policy,
          toolCalls,
        })
      );
      options.onTurnRecorded?.(response);
    } catch (error) {
      pendingUserText = userText;
      pendingAssistantText = assistantText;
      options.onError?.(
        error instanceof Error
          ? error
          : new Error("Could not record Realtime voice turn.")
      );
    } finally {
      recordingTurn = false;
    }
  }

  function prepareTurnPolicy(
    userText: string
  ): Promise<RealtimeVoiceTurnPolicyResponse | null> {
    const request = prepareRealtimeVoiceTurnPolicy({
      threadId: options.threadId,
      userId: options.userId,
      userText,
      memoryMode: options.memoryMode,
    });
    const tracked = request
      .then((policy) => {
        latestPolicy = policy;
        options.onTurnPolicy?.(policy);
        sendResponseCreate(policy);
        return policy;
      })
      .catch((error) => {
        options.onError?.(
          error instanceof Error
            ? error
            : new Error("Could not prepare Realtime voice turn policy.")
        );
        sendResponseCreate(null);
        return null;
      })
      .finally(() => {
        if (pendingPolicyPromise === tracked) pendingPolicyPromise = null;
      });
    return tracked;
  }

  function sendResponseCreate(
    policy: RealtimeVoiceTurnPolicyResponse | null
  ): void {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    dataChannel.send(
      serializeRealtimeEvent(buildRealtimeVoiceResponseCreateEvent(policy))
    );
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
      const result = await executeRealtimeVoiceTool({
        threadId: options.threadId,
        userId: options.userId,
        currentUserMessage: pendingUserText,
        transcript: transcriptLog,
        memoryMode: options.memoryMode,
        toolName: call.name,
        arguments: call.arguments,
      });
      completedToolCalls.push({
        tool_name: call.name,
        status: "completed",
        output: result.output,
      });
      dataChannel.send(
        serializeRealtimeEvent(buildFunctionCallOutputEvent(call.callId, result.output))
      );
      dataChannel.send(serializeRealtimeEvent(buildResponseCreateEvent()));
      options.onToolEvent?.({
        callId: call.callId,
        name: call.name,
        status: "completed",
        output: result.output,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Realtime voice tool failed.";
      completedToolCalls.push({
        tool_name: call.name,
        status: "failed",
        output: {},
        error: message,
      });
      dataChannel.send(
        serializeRealtimeEvent(
          buildFunctionCallOutputEvent(call.callId, { error: message })
        )
      );
      dataChannel.send(serializeRealtimeEvent(buildResponseCreateEvent()));
      options.onToolEvent?.({
        callId: call.callId,
        name: call.name,
        status: "failed",
        detail: message,
      });
      options.onError?.(new Error(message));
    }
  }
}
