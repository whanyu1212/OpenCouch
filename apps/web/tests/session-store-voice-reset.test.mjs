import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const storage = new Map();

globalThis.localStorage = {
  getItem: (key) => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: (key) => storage.delete(key),
  clear: () => storage.clear(),
  key: (index) => Array.from(storage.keys())[index] ?? null,
  get length() {
    return storage.size;
  },
};

async function importSessionStoreForNode() {
  const root = resolve(import.meta.dirname, "..");
  const sessionPath = join(root, "src/lib/session.ts");
  const apiUrl = pathToFileURL(join(root, "src/lib/api.ts")).href;
  const reactUrl = import.meta.resolve("react");
  const zustandUrl = import.meta.resolve("zustand");
  const zustandMiddlewareUrl = import.meta.resolve("zustand/middleware");
  const tempDir = await mkdtemp(join(tmpdir(), "opencouch-session-store-"));
  const tempSessionPath = join(tempDir, "session.ts");
  const source = await readFile(sessionPath, "utf8");

  await writeFile(
    tempSessionPath,
    source
      .replace('from "react";', `from ${JSON.stringify(reactUrl)};`)
      .replace('from "zustand";', `from ${JSON.stringify(zustandUrl)};`)
      .replace(
        'from "zustand/middleware";',
        `from ${JSON.stringify(zustandMiddlewareUrl)};`
      )
      .replace('from "./api";', `from ${JSON.stringify(apiUrl)};`)
  );

  return import(pathToFileURL(tempSessionPath).href);
}

const {
  buildEndedSessionResult,
  useSessionStore,
  voiceFinalizationBlocksSessionActions,
  voiceFinalizationBlocksTextTurns,
} = await importSessionStoreForNode();

function seedStaleVoiceState() {
  useSessionStore.setState({
    isSetup: true,
    sessionMode: "persistent",
    userId: "user-1",
    threadId: "old-thread",
    voiceConnected: true,
    voiceConnectionPending: true,
    voiceAgentSpeaking: true,
    voiceReadyToSpeak: true,
    voiceTranscripts: [
      { role: "user", text: "old transcript", itemId: "old-user-item" },
    ],
    voiceActivities: [
      {
        id: "old-activity",
        activity: "therapeutic_skill",
        status: "completed",
        label: "Old tool",
        detail: "old detail",
        timestamp: "2026-05-24T00:00:00.000Z",
      },
    ],
    voiceFinalization: {
      threadId: "old-thread",
      status: "completed",
      blocksTextTurns: false,
      detail: "Old session ended.",
      updatedAt: "2026-05-24T00:00:00.000Z",
    },
    voiceSessionInfo: {
      roomName: "old-thread",
      identity: "user-1",
      memoryMode: "persistent",
      assistantVoice: "cedar",
      serverUrl: "https://api.openai.com/v1/realtime/calls",
      connectedAt: "2026-05-24T00:00:00.000Z",
    },
    voiceError: "old voice error",
    voiceSafetyOverlay: {
      clientTurnId: "old-turn",
      open: true,
      riskLevel: 3,
      headline: "Old support",
      validation: "Old validation",
      immediateStep: "Old step",
      resourceStatus: "loading",
      inferredLocation: "",
      resources: [],
      message: "Old lookup",
    },
    voiceSafetyResourceWorkActive: true,
    voiceSuppressedAssistantResponseIds: ["old-response"],
    voiceSuppressedAssistantItemIds: ["old-assistant-item"],
    lastEndedSession: {
      threadId: "old-thread",
      modality: "voice",
      finalized: true,
      summary: "old summary",
      detail: "old detail",
    },
  });
}

function assertVoiceUiReset() {
  const state = useSessionStore.getState();

  assert.equal(state.voiceConnected, false);
  assert.equal(state.voiceConnectionPending, false);
  assert.equal(state.voiceAgentSpeaking, false);
  assert.equal(state.voiceReadyToSpeak, false);
  assert.deepEqual(state.voiceTranscripts, []);
  assert.deepEqual(state.voiceActivities, []);
  assert.deepEqual(state.voiceFinalization, {
    threadId: null,
    status: "idle",
    blocksTextTurns: false,
    detail: null,
    updatedAt: null,
  });
  assert.equal(state.voiceSessionInfo, null);
  assert.equal(state.voiceError, null);
  assert.equal(state.voiceSafetyOverlay, null);
  assert.equal(state.voiceSafetyResourceWorkActive, false);
  assert.deepEqual(state.voiceSuppressedAssistantResponseIds, []);
  assert.deepEqual(state.voiceSuppressedAssistantItemIds, []);
  assert.equal(state.lastEndedSession, null);
}

beforeEach(() => {
  storage.clear();
  useSessionStore.setState(useSessionStore.getInitialState(), true);
});

test("startSession clears stale voice transcript and ended-call state", () => {
  seedStaleVoiceState();

  useSessionStore.getState().startSession("persistent", "user-2", "new-thread");

  assert.equal(useSessionStore.getState().threadId, "new-thread");
  assertVoiceUiReset();
});

test("newSession clears stale voice transcript and ended-call state", () => {
  seedStaleVoiceState();

  useSessionStore.getState().newSession();

  assert.equal(useSessionStore.getState().isSetup, false);
  assert.equal(useSessionStore.getState().threadId, "");
  assertVoiceUiReset();
});

test("ended session metadata carries the originating modality", () => {
  const textSession = buildEndedSessionResult({
    threadId: "text-thread",
    result: { summary: "text summary", detail: "text ended" },
    modality: "text",
  });
  const voiceSession = buildEndedSessionResult({
    threadId: "voice-thread",
    result: { summary: "voice summary", detail: "voice ended" },
    modality: "voice",
  });

  assert.equal(textSession.modality, "text");
  assert.equal(voiceSession.modality, "voice");
});

test("only interrupted voice settlement blocks text turns", () => {
  const normalFinalization = {
    threadId: "voice-thread",
    status: "failed",
    blocksTextTurns: false,
    detail: "Memory save failed.",
    updatedAt: "2026-05-24T00:00:00.000Z",
  };
  const interruptedFinalization = {
    ...normalFinalization,
    blocksTextTurns: true,
  };

  assert.equal(
    voiceFinalizationBlocksTextTurns(normalFinalization, "voice-thread"),
    false
  );
  assert.equal(
    voiceFinalizationBlocksTextTurns(interruptedFinalization, "voice-thread"),
    true
  );
  assert.equal(
    voiceFinalizationBlocksTextTurns(interruptedFinalization, "other-thread"),
    false
  );
});

test("session actions block only for current pending or safety-failed finalization", () => {
  const finalization = {
    threadId: "voice-thread",
    status: "in_progress",
    blocksTextTurns: false,
    detail: "Saving session memory.",
    updatedAt: "2026-05-24T00:00:00.000Z",
  };

  assert.equal(
    voiceFinalizationBlocksSessionActions(finalization, "voice-thread"),
    true
  );
  assert.equal(
    voiceFinalizationBlocksSessionActions(
      { ...finalization, status: "failed" },
      "voice-thread"
    ),
    false
  );
  assert.equal(
    voiceFinalizationBlocksSessionActions(
      { ...finalization, status: "failed", blocksTextTurns: true },
      "voice-thread"
    ),
    true
  );
  assert.equal(
    voiceFinalizationBlocksSessionActions(finalization, "other-thread"),
    false
  );
});

test("safety overlay updates only for the matching open turn and dismisses", () => {
  const store = useSessionStore.getState();
  store.setVoiceSafetyOverlay({
    clientTurnId: "client-turn-1",
    riskLevel: 3,
    support: {
      headline: "Stay with me",
      validation: "You deserve support.",
      immediate_step: "Call someone nearby now.",
    },
  });

  store.updateVoiceSafetyResources("wrong-turn", {
    client_turn_id: "wrong-turn",
    status: "found",
    inferred_location: "US",
    resources: [],
    message: "Wrong result",
  });
  assert.equal(useSessionStore.getState().voiceSafetyOverlay.resourceStatus, "loading");

  store.updateVoiceSafetyResources("client-turn-1", {
    client_turn_id: "client-turn-1",
    status: "found",
    inferred_location: "US",
    resources: [
      {
        name: "988 Lifeline",
        phone: "988",
        url: "https://988lifeline.org/",
        region: "US",
      },
    ],
    message: "Verified support is available.",
  });
  assert.equal(useSessionStore.getState().voiceSafetyOverlay.resourceStatus, "found");

  store.dismissVoiceSafetyOverlay();
  assert.equal(useSessionStore.getState().voiceSafetyOverlay, null);
});

test("suppresses existing and future assistant transcripts by response or item", () => {
  const store = useSessionStore.getState();
  store.addVoiceTranscript({
    role: "assistant",
    text: "cancelled draft",
    itemId: "assistant-1",
    responseId: "response-1",
  });
  store.addVoiceTranscript({ role: "user", text: "keep me", itemId: "user-1" });
  store.suppressVoiceAssistantTranscripts({
    responseIds: ["response-1"],
    itemIds: ["assistant-2"],
  });
  store.addVoiceTranscript({
    role: "assistant",
    text: "late cancelled draft",
    itemId: "assistant-2",
  });

  assert.deepEqual(useSessionStore.getState().voiceTranscripts, [
    { role: "user", text: "keep me", itemId: "user-1" },
  ]);
});
