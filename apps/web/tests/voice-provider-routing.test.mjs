import assert from "node:assert/strict";
import { test } from "node:test";

import { shouldUseRealtimeVoiceProvider } from "../src/lib/voice-provider-routing.ts";

test("does not mount Realtime provider for realtime dogfood route after connect", () => {
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/voice/realtime-dev",
      voiceConnected: true,
      voiceFinalizationStatus: "idle",
    }),
    false
  );
});

test("keeps Realtime provider mounted for the production voice route", () => {
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/voice",
      voiceConnected: false,
      voiceFinalizationStatus: "idle",
    }),
    true
  );
});

test("keeps Realtime provider mounted off-route for active voice sessions", () => {
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/memory",
      voiceConnected: true,
      voiceFinalizationStatus: "idle",
    }),
    true
  );
});

test("keeps Realtime provider mounted while connection setup or retry is pending", () => {
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/memory",
      voiceConnected: false,
      voiceConnectionPending: true,
      voiceFinalizationStatus: "idle",
    }),
    true
  );
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/memory",
      voiceConnected: false,
      voiceFinalizationStatus: "failed",
    }),
    true
  );
});

test("keeps Realtime provider mounted for safety overlay and resource work", () => {
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/memory",
      voiceConnected: false,
      voiceFinalizationStatus: "completed",
      voiceSafetyOverlayActive: true,
    }),
    true
  );
  assert.equal(
    shouldUseRealtimeVoiceProvider({
      pathname: "/memory",
      voiceConnected: false,
      voiceFinalizationStatus: "completed",
      voiceSafetyResourceWorkActive: true,
    }),
    true
  );
});
