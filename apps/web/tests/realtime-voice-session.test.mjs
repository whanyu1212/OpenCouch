import assert from "node:assert/strict";
import { test } from "node:test";

import {
  shouldCreateResponseAfterRealtimeVoiceTool,
  shouldRecordRealtimeVoiceToolCall,
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
