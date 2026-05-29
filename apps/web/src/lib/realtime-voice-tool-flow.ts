export const WAIT_FOR_USER_TOOL_NAME = "wait_for_user";

export function isRealtimeVoiceWaitForUserTool(toolName: string): boolean {
  return toolName === WAIT_FOR_USER_TOOL_NAME;
}

export function shouldCreateResponseAfterRealtimeVoiceTool(
  toolName: string
): boolean {
  return !isRealtimeVoiceWaitForUserTool(toolName);
}

export function shouldRecordRealtimeVoiceToolCall(toolName: string): boolean {
  return !isRealtimeVoiceWaitForUserTool(toolName);
}
