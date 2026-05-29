import type {
  RealtimeVoiceRecordedToolCall,
  VoiceMemoryMode,
} from "./api";

export function buildRealtimeVoiceTurnRecordInput({
  threadId,
  userId,
  userText,
  assistantText,
  memoryMode,
  toolCalls,
}: {
  threadId: string;
  userId?: string;
  userText: string;
  assistantText: string;
  memoryMode: VoiceMemoryMode;
  toolCalls?: RealtimeVoiceRecordedToolCall[];
}) {
  return {
    threadId,
    userId,
    userText,
    assistantText,
    memoryMode,
    toolCalls: toolCalls || [],
  };
}
