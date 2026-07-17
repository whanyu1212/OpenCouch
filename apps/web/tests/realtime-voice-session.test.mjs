import assert from "node:assert/strict";
import { test } from "node:test";

import {
  readRealtimeVoiceUserQuote,
  realtimeVoiceEvidenceMatchesUserQuote,
  shouldCreateResponseAfterRealtimeVoiceTool,
  shouldRecordRealtimeVoiceToolCall,
  shouldWaitForRealtimeVoiceTranscriptEvidence,
} from "../src/lib/realtime-voice-tool-flow.ts";
import {
  finalizeAfterPendingRealtimeVoiceTurn,
  onRealtimeVoiceTurnRecordingSettled,
} from "../src/lib/realtime-voice-finalization.ts";
import { buildRealtimeVoiceTurnRecordInput } from "../src/lib/realtime-voice-turn-record.ts";

test("waits for pending turn recording before finalizing", async () => {
  let releaseRecording;
  const events = [];
  const pendingRecording = new Promise((resolve) => {
    releaseRecording = () => {
      events.push("recorded");
      resolve();
    };
  });

  const finalizing = finalizeAfterPendingRealtimeVoiceTurn(
    pendingRecording,
    async () => {
      events.push("finalized");
      return "ended";
    }
  );

  await Promise.resolve();
  assert.deepEqual(events, []);
  releaseRecording();
  assert.equal(await finalizing, "ended");
  assert.deepEqual(events, ["recorded", "finalized"]);
});

test("does not finalize when pending turn recording fails", async () => {
  const recordingError = new Error("recording failed");
  let finalized = false;

  await assert.rejects(
    finalizeAfterPendingRealtimeVoiceTurn(Promise.reject(recordingError), async () => {
      finalized = true;
    }),
    recordingError
  );
  assert.equal(finalized, false);
});

test("clears rejected turn recordings so they can be retried", async () => {
  const recording = Promise.reject(new Error("transient recording failure"));
  let settledRecording = null;

  onRealtimeVoiceTurnRecordingSettled(recording, (settled) => {
    settledRecording = settled;
  });
  await assert.rejects(recording);
  await Promise.resolve();

  assert.equal(settledRecording, recording);
});

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
