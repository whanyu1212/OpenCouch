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
  userItemId?: string;
  transcript?: RealtimeTranscriptUpdate;
  functionCalls: RealtimeFunctionCall[];
  agentSpeaking?: boolean;
  readyToSpeak?: boolean;
  errorMessage?: string;
}

type JsonRecord = Record<string, unknown>;

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
    parsed.agentSpeaking = false;
    parsed.readyToSpeak = true;
  } else if (type === "response.created") {
    const responseId = readString(asRecord(event.response).id);
    if (responseId) parsed.responseId = responseId;
    parsed.agentSpeaking = true;
    parsed.readyToSpeak = false;
  } else if (
    type === "response.output_audio.done" ||
    type === "response.cancelled" ||
    type === "response.failed"
  ) {
    parsed.agentSpeaking = false;
  } else if (type === "input_audio_buffer.speech_started") {
    parsed.readyToSpeak = false;
  } else if (type === "input_audio_buffer.speech_stopped") {
    parsed.readyToSpeak = false;
  } else if (type === "error") {
    parsed.errorMessage = readErrorMessage(event);
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

export function buildResponseCreateEvent(instructions?: string | null): JsonRecord {
  const trimmedInstructions = instructions?.trim();
  if (!trimmedInstructions) return { type: "response.create" };
  return {
    type: "response.create",
    response: {
      instructions: trimmedInstructions,
    },
  };
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
