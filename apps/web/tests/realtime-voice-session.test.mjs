import assert from "node:assert/strict";
import { test } from "node:test";

import {
  readRealtimeVoiceUserQuote,
  realtimeVoiceEvidenceMatchesUserQuote,
  shouldCreateResponseAfterRealtimeVoiceTool,
  shouldRecordRealtimeVoiceToolCall,
  shouldWaitForRealtimeVoiceTranscriptEvidence,
} from "../src/lib/realtime-voice-tool-flow.ts";
import { buildRealtimeVoiceTurnRecordInput } from "../src/lib/realtime-voice-turn-record.ts";

test("builds voice turn record input from completed tool calls only", () => {
  const input = buildRealtimeVoiceTurnRecordInput({
    threadId: "voice-thread",
    userId: "user-1",
    userText: "current guidance",
    assistantText: "verified answer",
    memoryMode: "persistent",
    toolCalls: [
      {
        tool_name: "answer_grounded_lookup",
        status: "completed",
        output: {
          grounded_lookup: {
            query: "current guidance",
            status: "answered",
          },
        },
      },
    ],
  });

  assert.deepEqual(input, {
    threadId: "voice-thread",
    userId: "user-1",
    userText: "current guidance",
    assistantText: "verified answer",
    memoryMode: "persistent",
    toolCalls: [
      {
        tool_name: "answer_grounded_lookup",
        status: "completed",
        output: {
          grounded_lookup: {
            query: "current guidance",
            status: "answered",
          },
        },
      },
    ],
  });
});

test("creates follow-up responses after actionable tools only", () => {
  assert.equal(
    shouldCreateResponseAfterRealtimeVoiceTool("answer_grounded_lookup"),
    true
  );
  assert.equal(shouldCreateResponseAfterRealtimeVoiceTool("wait_for_user"), false);
});

test("records actionable voice tool calls only", () => {
  assert.equal(shouldRecordRealtimeVoiceToolCall("answer_grounded_lookup"), true);
  assert.equal(shouldRecordRealtimeVoiceToolCall("wait_for_user"), false);
});

test("gates quote-mutating voice tools on transcript evidence", () => {
  assert.equal(
    shouldWaitForRealtimeVoiceTranscriptEvidence("save_response_preference"),
    true
  );
  assert.equal(
    shouldWaitForRealtimeVoiceTranscriptEvidence("set_proactive_memory_recall"),
    true
  );
  assert.equal(
    shouldWaitForRealtimeVoiceTranscriptEvidence("answer_grounded_lookup"),
    false
  );
});

test("matches voice user quote against transcript evidence", () => {
  assert.equal(
    readRealtimeVoiceUserQuote({
      user_quote: " remember that concise replies help ",
    }),
    "remember that concise replies help"
  );
  assert.equal(
    realtimeVoiceEvidenceMatchesUserQuote({
      evidence: "Please remember that concise replies help me stay focused.",
      userQuote: "remember that concise replies help",
    }),
    true
  );
  assert.equal(
    realtimeVoiceEvidenceMatchesUserQuote({
      evidence: "Please remember that concise replies help me stay focused.",
      userQuote: "remember that long replies help",
    }),
    false
  );
});
