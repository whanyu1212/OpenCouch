import assert from "node:assert/strict";
import { test } from "node:test";

process.env.NEXT_PUBLIC_API_URL = "http://backend.test/api";

const api = await import("../src/lib/api.ts");

test("creates realtime voice sessions with the selected assistant voice", async () => {
  let capturedUrl = "";
  let capturedBody = {};
  globalThis.fetch = async (url, init) => {
    capturedUrl = String(url);
    capturedBody = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        client_secret: "ek_test_secret",
        thread_id: "voice-thread",
        user_id: "user-1",
        memory_mode: "persistent",
        session_config: {
          type: "realtime",
          audio: { output: { voice: "cedar" } },
        },
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  const response = await api.createRealtimeVoiceSession({
    threadId: "voice-thread",
    userId: "user-1",
    memoryMode: "persistent",
    assistantVoice: "cedar",
  });

  assert.equal(capturedUrl, "http://backend.test/api/voice/realtime/session");
  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    memory_mode: "persistent",
    assistant_voice: "cedar",
  });
  assert.equal(response.session_config.audio.output.voice, "cedar");
});

test("prepares realtime voice turn policy through the backend API", async () => {
  let capturedUrl = "";
  let capturedBody = {};
  globalThis.fetch = async (url, init) => {
    capturedUrl = String(url);
    capturedBody = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        route: "grounded_lookup",
        response_style: "grounded_lookup",
        required_tool_name: "answer_grounded_lookup",
        required_tool_arguments: { query: "current guidance" },
        instructions: "Call answer_grounded_lookup.",
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  const response = await api.prepareRealtimeVoiceTurnPolicy({
    threadId: "voice-thread",
    userId: "user-1",
    userText: "current guidance",
    memoryMode: "persistent",
  });

  assert.equal(capturedUrl, "http://backend.test/api/voice/realtime/turn-policy");
  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    user_text: "current guidance",
    memory_mode: "persistent",
  });
  assert.equal(response.required_tool_name, "answer_grounded_lookup");
});

test("records realtime voice turns with route and tool metadata", async () => {
  let capturedBody = {};
  globalThis.fetch = async (_url, init) => {
    capturedBody = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        recorded: true,
        thread_id: "voice-thread",
        message_count: 2,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  await api.recordRealtimeVoiceTurn({
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

  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    user_text: "current guidance",
    assistant_text: "verified answer",
    memory_mode: "persistent",
    route: "grounded_lookup",
    response_style: "grounded_lookup",
    tool_calls: [
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

test("ends realtime voice sessions with the requested memory mode", async () => {
  let capturedBody = {};
  globalThis.fetch = async (_url, init) => {
    capturedBody = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        finalized: false,
        summary: null,
        detail: "Incognito voice session ended without durable finalization.",
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  await api.endRealtimeVoiceSession("voice-thread", "incognito");

  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    memory_mode: "incognito",
  });
});
