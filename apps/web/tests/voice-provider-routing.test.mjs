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
