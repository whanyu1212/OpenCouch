export type RealtimeTranscriptRole = "user" | "assistant";

export interface RealtimeTranscriptUpdate {
  role: RealtimeTranscriptRole;
  itemId?: string;
  responseId?: string;
  text: string;
  final: boolean;
}

export interface RealtimeFunctionCall {
  callId: string;
  itemId?: string;
  responseId?: string;
  name: string;
  arguments: Record<string, unknown>;
  rawArguments: string;
}

export interface ParsedRealtimeServerEvent {
  type: string;
  responseId?: string;
  responseTerminal?: boolean;
  responseRequestId?: string;
  userItemId?: string;
  failedUserTranscriptionItemId?: string;
  transcript?: RealtimeTranscriptUpdate;
  functionCalls: RealtimeFunctionCall[];
  agentSpeaking?: boolean;
  readyToSpeak?: boolean;
  errorMessage?: string;
  errorEventId?: string;
}

type JsonRecord = Record<string, unknown>;

const RESPONSE_REQUEST_METADATA_KEY = "opencouch_response_request_id";

export function parseRealtimeServerEvent(raw: unknown): ParsedRealtimeServerEvent {
  const event = asRecord(raw);
  const type = readString(event.type) ?? "unknown";
  const parsed: ParsedRealtimeServerEvent = {
    type,
    functionCalls: [],
  };

  if (type === "input_audio_buffer.committed") {
    const userItemId = readString(event.item_id);
    if (userItemId) parsed.userItemId = userItemId;
  } else if (type === "conversation.item.input_audio_transcription.delta") {
    parsed.transcript = {
      role: "user",
      itemId: readString(event.item_id),
      text: readString(event.delta) ?? "",
      final: false,
    };
  } else if (type === "conversation.item.input_audio_transcription.completed") {
    parsed.transcript = {
      role: "user",
      itemId: readString(event.item_id),
      text: readString(event.transcript) ?? "",
      final: true,
    };
  } else if (type === "conversation.item.input_audio_transcription.failed") {
    const itemId = readString(event.item_id);
    if (itemId) parsed.failedUserTranscriptionItemId = itemId;
  } else if (type === "response.output_audio_transcript.delta") {
    const responseId = readString(event.response_id);
    parsed.transcript = {
      role: "assistant",
      itemId: readString(event.item_id),
      ...(responseId ? { responseId } : {}),
      text: readString(event.delta) ?? "",
      final: false,
    };
  } else if (type === "response.output_audio_transcript.done") {
    const responseId = readString(event.response_id);
    parsed.transcript = {
      role: "assistant",
      itemId: readString(event.item_id),
      ...(responseId ? { responseId } : {}),
      text: readString(event.transcript) ?? "",
      final: true,
    };
  } else if (type === "response.function_call_arguments.done") {
    const responseId = readString(event.response_id);
    const call = functionCallFromRecord(event, responseId);
    if (call) parsed.functionCalls.push(call);
  } else if (type === "response.done") {
    const responseId = readString(asRecord(event.response).id);
    if (responseId) parsed.responseId = responseId;
    parsed.functionCalls = functionCallsFromResponseDone(event, responseId);
    parsed.responseTerminal = true;
    parsed.agentSpeaking = false;
    parsed.readyToSpeak = true;
  } else if (type === "response.created") {
    const response = asRecord(event.response);
    const responseId = readString(response.id);
    const responseRequestId = readString(
      asRecord(response.metadata)[RESPONSE_REQUEST_METADATA_KEY]
    );
    if (responseId) parsed.responseId = responseId;
    if (responseRequestId) parsed.responseRequestId = responseRequestId;
    parsed.agentSpeaking = true;
    parsed.readyToSpeak = false;
  } else if (type === "response.cancelled" || type === "response.failed") {
    const responseId =
      readString(event.response_id) ?? readString(asRecord(event.response).id);
    if (responseId) parsed.responseId = responseId;
    parsed.responseTerminal = true;
    parsed.agentSpeaking = false;
  } else if (type === "response.output_audio.done") {
    parsed.agentSpeaking = false;
  } else if (type === "input_audio_buffer.speech_started") {
    parsed.readyToSpeak = false;
  } else if (type === "input_audio_buffer.speech_stopped") {
    parsed.readyToSpeak = false;
  } else if (type === "error") {
    parsed.errorMessage = readErrorMessage(event);
    parsed.errorEventId = readString(asRecord(event.error).event_id);
  }

  return parsed;
}

export function buildFunctionCallOutputEvent(
  callId: string,
  output: Record<string, unknown>
): JsonRecord {
  return {
    type: "conversation.item.create",
    item: {
      type: "function_call_output",
      call_id: callId,
      output: JSON.stringify(output),
    },
  };
}

export function buildResponseCreateEvent(
  instructions?: string | null,
  requestEventId?: string
): JsonRecord {
  const trimmedInstructions = instructions?.trim();
  if (!trimmedInstructions && !requestEventId) return { type: "response.create" };

  const response: JsonRecord = {};
  if (trimmedInstructions) response.instructions = trimmedInstructions;
  if (requestEventId) {
    response.metadata = {
      [RESPONSE_REQUEST_METADATA_KEY]: requestEventId,
    };
  }
  return {
    type: "response.create",
    ...(requestEventId ? { event_id: requestEventId } : {}),
    response,
  };
}

export function buildResponseCancelEvent(): JsonRecord {
  return { type: "response.cancel" };
}

export function serializeRealtimeEvent(event: JsonRecord): string {
  return JSON.stringify(event);
}

function functionCallsFromResponseDone(
  event: JsonRecord,
  responseId?: string
): RealtimeFunctionCall[] {
  const response = asRecord(event.response);
  const output = Array.isArray(response.output) ? response.output : [];
  const calls: RealtimeFunctionCall[] = [];

  for (const item of output) {
    const call = functionCallFromRecord(asRecord(item), responseId);
    if (call) calls.push(call);
  }

  return calls;
}

function functionCallFromRecord(
  item: JsonRecord,
  responseId?: string
): RealtimeFunctionCall | null {
  const type = readString(item.type);
  if (
    type !== "function_call" &&
    type !== "response.function_call_arguments.done"
  ) {
    return null;
  }

  const name = readString(item.name);
  const callId = readString(item.call_id);
  if (!name || !callId) return null;

  const rawArguments = readString(item.arguments) ?? "{}";

  return {
    callId,
    itemId: readString(item.item_id) ?? readString(item.id),
    ...(responseId ? { responseId } : {}),
    name,
    arguments: parseFunctionArguments(rawArguments),
    rawArguments,
  };
}

function parseFunctionArguments(raw: string): Record<string, unknown> {
  try {
    return asRecord(JSON.parse(raw || "{}"));
  } catch {
    return {};
  }
}

function readErrorMessage(event: JsonRecord): string | undefined {
  const nested = asRecord(event.error);
  return (
    readString(nested.message) ??
    readString(nested.code) ??
    readString(event.message)
  );
}

function asRecord(value: unknown): JsonRecord {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as JsonRecord;
  }
  return {};
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}
