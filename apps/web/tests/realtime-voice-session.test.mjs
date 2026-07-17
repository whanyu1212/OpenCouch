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
  clearHandleAfterSuccessfulDisconnect,
  finalizeAfterPendingRealtimeVoiceTurn,
  onRealtimeVoiceTurnRecordingSettled,
  RealtimeVoiceDisconnectCoordinator,
} from "../src/lib/realtime-voice-finalization.ts";
import {
  buildRealtimeVoiceTurnRecordInput,
  restoreRealtimeVoiceRecordedToolCalls,
} from "../src/lib/realtime-voice-turn-record.ts";

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

test("retries disconnect after a failed recording attempt", async () => {
  const coordinator = new RealtimeVoiceDisconnectCoordinator();
  let attempts = 0;

  await assert.rejects(
    coordinator.disconnect(async () => {
      attempts += 1;
      throw new Error("recording failed");
    }),
    /recording failed/
  );
  await Promise.resolve();

  await coordinator.disconnect(async () => {
    attempts += 1;
  });
  await coordinator.disconnect(async () => {
    attempts += 1;
  });

  assert.equal(attempts, 2);
});

test("deduplicates concurrent disconnect attempts", async () => {
  const coordinator = new RealtimeVoiceDisconnectCoordinator();
  let releaseAttempt;
  let attempts = 0;
  const attempt = () => {
    attempts += 1;
    return new Promise((resolve) => {
      releaseAttempt = resolve;
    });
  };

  const first = coordinator.disconnect(attempt);
  const second = coordinator.disconnect(attempt);
  assert.equal(first, second);
  assert.equal(attempts, 1);

  releaseAttempt();
  await first;
});

test("keeps the disconnect handle when finalization fails", async () => {
  let cleared = false;

  await assert.rejects(
    clearHandleAfterSuccessfulDisconnect(
      async () => {
        throw new Error("finalization failed");
      },
      () => {
        cleared = true;
      }
    ),
    /finalization failed/
  );
  assert.equal(cleared, false);

  await clearHandleAfterSuccessfulDisconnect(async () => undefined, () => {
    cleared = true;
  });
  assert.equal(cleared, true);
});

test("restores failed turn tool calls ahead of newer queued calls", () => {
  const failedTurn = [{ tool_name: "answer_grounded_lookup", status: "completed" }];
  const queued = [{ tool_name: "lookup_crisis_resources", status: "completed" }];

  assert.deepEqual(restoreRealtimeVoiceRecordedToolCalls(failedTurn, queued), [
    ...failedTurn,
    ...queued,
  ]);
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
