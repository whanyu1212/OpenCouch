import assert from "node:assert/strict";
import { test } from "node:test";

process.env.NEXT_PUBLIC_API_URL = "http://backend.test/api";

const api = await import("../src/lib/api.ts");

test("does not expose the unused backend info helper", () => {
  assert.equal(api.getInfo, undefined);
});

test("sends memory mode through text chat and thread API helpers", async () => {
  const captured = [];
  globalThis.fetch = async (url, init = {}) => {
    captured.push({
      url: String(url),
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    });
    return new Response(JSON.stringify({ response_text: "ok" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  await api.postChat("hello", "thread-1", "user-1", "fast", "incognito");
  await api.getThreads(5, "incognito");
  await api.getHistory("thread-1", "incognito");
  await api.getThreadSessionStatus("thread-1", "incognito");
  await api.endSession("thread-1", "positive", "incognito");

  assert.deepEqual(captured, [
    {
      url: "http://backend.test/api/chat",
      body: {
        message: "hello",
        thread_id: "thread-1",
        user_id: "user-1",
        response_model_tier: "fast",
        memory_mode: "incognito",
      },
    },
    {
      url: "http://backend.test/api/threads?limit=5&memory_mode=incognito",
      body: undefined,
    },
    {
      url: "http://backend.test/api/threads/thread-1/history?memory_mode=incognito",
      body: undefined,
    },
    {
      url: "http://backend.test/api/threads/thread-1/session-status?memory_mode=incognito",
      body: undefined,
    },
    {
      url: "http://backend.test/api/threads/thread-1/end",
      body: {
        feedback: "positive",
        memory_mode: "incognito",
      },
    },
  ]);
});

test("submits standalone session feedback with modality", async () => {
  let capturedUrl = "";
  let capturedBody = {};
  globalThis.fetch = async (url, init = {}) => {
    capturedUrl = String(url);
    capturedBody = init.body ? JSON.parse(String(init.body)) : undefined;
    return new Response(JSON.stringify({ recorded: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  const response = await api.submitSessionFeedback(
    "voice-thread",
    "negative",
    "incognito",
    "voice"
  );

  assert.equal(capturedUrl, "http://backend.test/api/threads/voice-thread/feedback");
  assert.deepEqual(capturedBody, {
    feedback: "negative",
    memory_mode: "incognito",
    modality: "voice",
  });
  assert.deepEqual(response, { recorded: true });
});

test("sends memory mode through memory API helpers", async () => {
  const captured = [];
  globalThis.fetch = async (url, init = {}) => {
    captured.push({
      url: String(url),
      method: init.method || "GET",
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    });
    return new Response(JSON.stringify({}), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  await api.getMemoryStatus("thread-1", "user-1", "incognito");
  await api.getMemoryFacts("thread-1", "user-1", "incognito");
  await api.getMemorySessions("thread-1", "user-1", "incognito");
  await api.getMemoryRules("thread-1", "user-1", "incognito");
  await api.updateMemoryRecall(true, "thread-1", "user-1", "incognito");
  await api.deleteMemoryFact(1, "thread-1", "user-1", "incognito");
  await api.deleteMemorySession(2, "thread-1", "user-1", "incognito");
  await api.deleteMemoryRule(3, "thread-1", "user-1", "incognito");

  assert.deepEqual(captured, [
    {
      url: "http://backend.test/api/memory/status?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "GET",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/facts?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "GET",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/sessions?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "GET",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/rules?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "GET",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/recall?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "PATCH",
      body: { enabled: true },
    },
    {
      url: "http://backend.test/api/memory/facts/1?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "DELETE",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/sessions/2?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "DELETE",
      body: undefined,
    },
    {
      url: "http://backend.test/api/memory/rules/3?thread_id=thread-1&user_id=user-1&memory_mode=incognito",
      method: "DELETE",
      body: undefined,
    },
  ]);
});

test("sends memory mode in text chat WebSocket payloads", () => {
  let capturedUrl = "";
  let capturedBody = {};

  globalThis.WebSocket = class {
    constructor(url) {
      capturedUrl = String(url);
    }

    send(payload) {
      capturedBody = JSON.parse(String(payload));
    }

    close() {}
  };

  const ws = api.createChatStream({
    message: "hello",
    threadId: "thread-1",
    userId: "user-1",
    memoryMode: "incognito",
    responseModelTier: "quality",
  });
  ws.onopen();

  assert.equal(capturedUrl, "ws://backend.test/api/chat/stream");
  assert.deepEqual(capturedBody, {
    message: "hello",
    thread_id: "thread-1",
    user_id: "user-1",
    memory_mode: "incognito",
    response_model_tier: "quality",
  });
});

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
        message_count: 0,
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

test("does not expose realtime voice turn policy helper", () => {
  assert.equal(api.prepareRealtimeVoiceTurnPolicy, undefined);
});

test("records realtime voice turns with tool metadata", async () => {
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
    outcome: "completed",
  });

  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    user_text: "current guidance",
    assistant_text: "verified answer",
    memory_mode: "persistent",
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
    outcome: "completed",
  });
});

test("forwards abort signals to realtime voice tool requests", async () => {
  let capturedSignal;
  let capturedBody = {};
  globalThis.fetch = async (_url, init) => {
    capturedSignal = init.signal;
    capturedBody = JSON.parse(String(init.body));
    return new Response(JSON.stringify({ output: {} }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  const controller = new AbortController();

  await api.executeRealtimeVoiceTool({
    threadId: "voice-thread",
    clientTurnId: "client-turn-1",
    memoryMode: "incognito",
    toolName: "show_memory_status",
    signal: controller.signal,
  });

  assert.equal(capturedSignal, controller.signal);
  assert.equal(capturedBody.client_turn_id, "client-turn-1");
});

test("checks realtime voice safety with the exact correlated turn payload", async () => {
  let capturedUrl = "";
  let capturedBody = {};
  let capturedSignal;
  globalThis.fetch = async (url, init) => {
    capturedUrl = String(url);
    capturedBody = JSON.parse(String(init.body));
    capturedSignal = init.signal;
    return new Response(
      JSON.stringify({
        client_turn_id: "client-turn-1",
        status: "completed",
        reason: null,
        action: "continue",
        risk_level: null,
        support: null,
        interruption_token: null,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  const controller = new AbortController();
  const response = await api.checkRealtimeVoiceSafety({
    threadId: "voice-thread",
    userId: "user-1",
    memoryMode: "persistent",
    clientTurnId: "client-turn-1",
    userText: "I feel overwhelmed.",
    priorMessageCount: 4,
    signal: controller.signal,
  });

  assert.equal(
    capturedUrl,
    "http://backend.test/api/voice/realtime/safety/check"
  );
  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    memory_mode: "persistent",
    client_turn_id: "client-turn-1",
    user_text: "I feel overwhelmed.",
    prior_message_count: 4,
    pending_prior_transcript: [],
  });
  assert.deepEqual(response, {
    client_turn_id: "client-turn-1",
    status: "completed",
    reason: null,
    action: "continue",
    risk_level: null,
    support: null,
    interruption_token: null,
  });
  assert.equal(capturedSignal, controller.signal);
});

test("requests safety resources with the correlated payload and signal", async () => {
  let capturedUrl = "";
  let capturedBody = {};
  let capturedSignal;
  globalThis.fetch = async (url, init) => {
    capturedUrl = String(url);
    capturedBody = JSON.parse(String(init.body));
    capturedSignal = init.signal;
    return new Response(
      JSON.stringify({
        client_turn_id: "client-turn-2",
        status: "found",
        inferred_location: "US",
        resources: [],
        message: "Verified support is available.",
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };
  const controller = new AbortController();

  await api.getRealtimeVoiceSafetyResources({
    threadId: "voice-thread",
    userId: "user-1",
    memoryMode: "persistent",
    clientTurnId: "client-turn-2",
    userText: "I may hurt myself.",
    priorMessageCount: 6,
    pendingPriorTranscript: [{ role: "user", content: "Earlier context" }],
    signal: controller.signal,
  });

  assert.equal(
    capturedUrl,
    "http://backend.test/api/voice/realtime/safety/resources"
  );
  assert.equal(capturedSignal, controller.signal);
  assert.deepEqual(capturedBody, {
    thread_id: "voice-thread",
    user_id: "user-1",
    memory_mode: "persistent",
    client_turn_id: "client-turn-2",
    user_text: "I may hurt myself.",
    prior_message_count: 6,
    pending_prior_transcript: [{ role: "user", content: "Earlier context" }],
  });
});

test("adds a client turn ID to turn recording only when provided", async () => {
  const capturedBodies = [];
  globalThis.fetch = async (_url, init) => {
    capturedBodies.push(JSON.parse(String(init.body)));
    return new Response(
      JSON.stringify({
        recorded: true,
        thread_id: "voice-thread",
        message_count: 2,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };

  const baseTurn = {
    threadId: "voice-thread",
    userText: "hello",
    assistantText: "hi",
    memoryMode: "incognito",
  };
  await api.recordRealtimeVoiceTurn(baseTurn);
  await api.recordRealtimeVoiceTurn({
    ...baseTurn,
    clientTurnId: "client-turn-1",
  });
  await api.recordRealtimeVoiceTurn({
    ...baseTurn,
    clientTurnId: "client-turn-2",
    outcome: "safety_interrupted",
    interruptionToken: "signed-proof",
  });

  assert.deepEqual(capturedBodies, [
    {
      thread_id: "voice-thread",
      user_text: "hello",
      assistant_text: "hi",
      memory_mode: "incognito",
      tool_calls: [],
      outcome: "completed",
    },
    {
      thread_id: "voice-thread",
      client_turn_id: "client-turn-1",
      user_text: "hello",
      assistant_text: "hi",
      memory_mode: "incognito",
      tool_calls: [],
      outcome: "completed",
    },
    {
      thread_id: "voice-thread",
      client_turn_id: "client-turn-2",
      user_text: "hello",
      assistant_text: "hi",
      memory_mode: "incognito",
      tool_calls: [],
      outcome: "safety_interrupted",
      interruption_token: "signed-proof",
    },
  ]);
});

test("ends realtime voice sessions with the requested memory mode", async () => {
  let capturedBody = {};
  globalThis.fetch = async (_url, init) => {
    capturedBody = JSON.parse(String(init.body));
    return new Response(
      JSON.stringify({
        finalized: false,
        summary: null,
        detail: "Incognito session ended without durable finalization.",
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
