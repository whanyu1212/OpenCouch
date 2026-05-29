export const WAIT_FOR_USER_TOOL_NAME = "wait_for_user";

const TRANSCRIPT_EVIDENCE_GATED_TOOL_NAMES = new Set([
  "save_response_preference",
  "set_proactive_memory_recall",
  "prepare_memory_deletion_by_index",
  "prepare_memory_deletion_by_query",
]);

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

export function shouldWaitForRealtimeVoiceTranscriptEvidence(
  toolName: string
): boolean {
  return TRANSCRIPT_EVIDENCE_GATED_TOOL_NAMES.has(toolName);
}

export function readRealtimeVoiceUserQuote(
  argumentsObject: Record<string, unknown>
): string {
  const value = argumentsObject.user_quote;
  return typeof value === "string" ? value.trim() : "";
}

export function realtimeVoiceEvidenceMatchesUserQuote({
  evidence,
  userQuote,
}: {
  evidence: string;
  userQuote: string;
}): boolean {
  const normalizedEvidence = normalizeVoiceEvidence(evidence);
  const normalizedQuote = normalizeVoiceEvidence(userQuote);
  return Boolean(normalizedQuote) && normalizedEvidence.includes(normalizedQuote);
}

function normalizeVoiceEvidence(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}
