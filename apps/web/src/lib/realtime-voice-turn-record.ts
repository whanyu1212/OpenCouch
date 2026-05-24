import type {
  RealtimeVoiceRecordedToolCall,
  RealtimeVoiceTurnPolicyResponse,
  VoiceMemoryMode,
} from "./api";

export function buildRealtimeVoiceTurnRecordInput({
  threadId,
  userId,
  userText,
  assistantText,
  memoryMode,
  policy,
  toolCalls,
}: {
  threadId: string;
  userId?: string;
  userText: string;
  assistantText: string;
  memoryMode: VoiceMemoryMode;
  policy?: RealtimeVoiceTurnPolicyResponse | null;
  toolCalls?: RealtimeVoiceRecordedToolCall[];
}) {
  return {
    threadId,
    userId,
    userText,
    assistantText,
    memoryMode,
    route: policy?.route,
    responseStyle: policy?.response_style,
    toolCalls: toolCalls || [],
  };
}
