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

const { useSessionStore } = await importSessionStoreForNode();

function seedStaleVoiceState() {
  useSessionStore.setState({
    isSetup: true,
    sessionMode: "persistent",
    userId: "user-1",
    threadId: "old-thread",
    voiceConnected: true,
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
    lastEndedSession: {
      threadId: "old-thread",
      finalized: true,
      summary: "old summary",
      detail: "old detail",
    },
  });
}

function assertVoiceUiReset() {
  const state = useSessionStore.getState();

  assert.equal(state.voiceConnected, false);
  assert.equal(state.voiceAgentSpeaking, false);
  assert.equal(state.voiceReadyToSpeak, false);
  assert.deepEqual(state.voiceTranscripts, []);
  assert.deepEqual(state.voiceActivities, []);
  assert.deepEqual(state.voiceFinalization, {
    threadId: null,
    status: "idle",
    detail: null,
    updatedAt: null,
  });
  assert.equal(state.voiceSessionInfo, null);
  assert.equal(state.voiceError, null);
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
