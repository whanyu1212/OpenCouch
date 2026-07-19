import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildFunctionCallOutputEvent,
  buildResponseCreateEvent,
  parseRealtimeServerEvent,
} from "../src/lib/realtime-voice-events.ts";

test("parses committed user item IDs before asynchronous transcription", () => {
  const parsed = parseRealtimeServerEvent({
    type: "input_audio_buffer.committed",
    item_id: "item-user-1",
    previous_item_id: "item-assistant-0",
  });

  assert.equal(parsed.userItemId, "item-user-1");
});

test("parses completed user input transcription events", () => {
  const parsed = parseRealtimeServerEvent({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "item-user-1",
    transcript: "I want to check what you remember.",
  });

  assert.equal(parsed.type, "conversation.item.input_audio_transcription.completed");
  assert.deepEqual(parsed.transcript, {
    role: "user",
    itemId: "item-user-1",
    text: "I want to check what you remember.",
    final: true,
  });
  assert.deepEqual(parsed.functionCalls, []);
});

test("parses assistant output transcript completion events", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.output_audio_transcript.done",
    response_id: "response-1",
    item_id: "item-assistant-1",
    transcript: "Memory is on for this persistent session.",
  });

  assert.deepEqual(parsed.transcript, {
    role: "assistant",
    itemId: "item-assistant-1",
    responseId: "response-1",
    text: "Memory is on for this persistent session.",
    final: true,
  });
});

test("preserves response IDs on assistant transcript deltas", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.output_audio_transcript.delta",
    response_id: "response-1",
    item_id: "item-assistant-1",
    delta: "Memory is",
  });

  assert.deepEqual(parsed.transcript, {
    role: "assistant",
    itemId: "item-assistant-1",
    responseId: "response-1",
    text: "Memory is",
    final: false,
  });
});

test("omits response IDs when assistant events do not provide one", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.output_audio_transcript.done",
    item_id: "item-assistant-1",
    transcript: "Legacy event shape.",
  });

  assert.deepEqual(parsed.transcript, {
    role: "assistant",
    itemId: "item-assistant-1",
    text: "Legacy event shape.",
    final: true,
  });
});

test("parses nested response IDs when responses are created", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.created",
    response: { id: "response-1" },
  });

  assert.equal(parsed.responseId, "response-1");
  assert.equal(parsed.agentSpeaking, true);
  assert.equal(parsed.readyToSpeak, false);
});

test("parses completed function calls from response.done", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.done",
    response: {
      id: "response-1",
      output: [
        {
          type: "function_call",
          name: "show_memory_status",
          call_id: "call_123",
          arguments: "{\"include_counts\":true}",
        },
      ],
    },
  });

  assert.equal(parsed.responseId, "response-1");
  assert.deepEqual(parsed.functionCalls, [
    {
      callId: "call_123",
      itemId: undefined,
      responseId: "response-1",
      name: "show_memory_status",
      arguments: { include_counts: true },
      rawArguments: "{\"include_counts\":true}",
    },
  ]);
});

test("parses function calls when streamed arguments are done", () => {
  const parsed = parseRealtimeServerEvent({
    type: "response.function_call_arguments.done",
    response_id: "response-1",
    item_id: "item-call-1",
    name: "show_memory_status",
    call_id: "call_123",
    arguments: "{}",
  });

  assert.deepEqual(parsed.functionCalls, [
    {
      callId: "call_123",
      itemId: "item-call-1",
      responseId: "response-1",
      name: "show_memory_status",
      arguments: {},
      rawArguments: "{}",
    },
  ]);
});

test("builds function call output events for the data channel", () => {
  assert.deepEqual(
    buildFunctionCallOutputEvent("call_123", { memory_mode: "persistent" }),
    {
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: "call_123",
        output: "{\"memory_mode\":\"persistent\"}",
      },
    }
  );
});

test("builds response create events with app-owned instructions", () => {
  assert.deepEqual(buildResponseCreateEvent("Call answer_grounded_lookup first."), {
    type: "response.create",
    response: {
      instructions: "Call answer_grounded_lookup first.",
    },
  });
});
