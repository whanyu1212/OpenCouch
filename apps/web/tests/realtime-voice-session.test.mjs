import assert from "node:assert/strict";
import { test } from "node:test";

import { buildRealtimeVoiceTurnRecordInput } from "../src/lib/realtime-voice-turn-record.ts";

test("builds voice turn record input from policy and completed tool calls", () => {
  const input = buildRealtimeVoiceTurnRecordInput({
    threadId: "voice-thread",
    userId: "user-1",
    userText: "current guidance",
    assistantText: "verified answer",
    memoryMode: "persistent",
    policy: {
      route: "grounded_lookup",
      response_style: "grounded_lookup",
      required_tool_name: "answer_grounded_lookup",
      required_tool_arguments: { query: "current guidance" },
      instructions: "Call answer_grounded_lookup.",
    },
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
    route: "grounded_lookup",
    responseStyle: "grounded_lookup",
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
