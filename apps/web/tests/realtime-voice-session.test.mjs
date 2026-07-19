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
  shouldRetryRealtimeVoiceFinalization,
} from "../src/lib/realtime-voice-finalization.ts";
import {
  buildRealtimeVoiceTurnRecordInput,
  RealtimeVoiceTurnTracker,
} from "../src/lib/realtime-voice-turn-record.ts";

function createTurnTracker() {
  let nextId = 1;
  return new RealtimeVoiceTurnTracker(() => `client-turn-${nextId++}`);
}

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

test("retries failed finalization only while its handle is available", () => {
  assert.equal(shouldRetryRealtimeVoiceFinalization(true, true), true);
  assert.equal(shouldRetryRealtimeVoiceFinalization(true, false), false);
  assert.equal(shouldRetryRealtimeVoiceFinalization(false, true), false);
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

test("correlates response-created before and after final user transcripts", () => {
  const tracker = createTurnTracker();

  const firstUser = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "first question",
  });
  tracker.responseCreated("response-1");
  tracker.responseCreated("response-2");
  const secondUser = tracker.addFinalUserTranscript({
    itemId: "user-2",
    text: "second question",
  });

  tracker.addFinalAssistantTranscript({
    responseId: "response-2",
    text: "second answer",
  });
  tracker.responseFinished("response-2");
  assert.equal(tracker.markNextRecordableTurn(), null);

  tracker.addFinalAssistantTranscript({
    responseId: "response-1",
    text: "first answer",
  });
  tracker.responseFinished("response-1");
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: firstUser.clientTurnId,
    userText: "first question",
    assistantText: "first answer",
    toolCalls: [],
  });
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: secondUser.clientTurnId,
    userText: "second question",
    assistantText: "second answer",
    toolCalls: [],
  });
});

test("uses committed item IDs when user transcripts finish out of order", () => {
  const tracker = createTurnTracker();
  const firstTurnId = tracker.userInputCommitted("user-1");
  tracker.responseCreated("response-1");
  const secondTurnId = tracker.userInputCommitted("user-2");
  tracker.responseCreated("response-2");
  tracker.addFinalAssistantTranscript({
    responseId: "response-1",
    text: "first answer",
  });
  tracker.responseFinished("response-1");
  tracker.addFinalAssistantTranscript({
    responseId: "response-2",
    text: "second answer",
  });
  tracker.responseFinished("response-2");

  const secondUser = tracker.addFinalUserTranscript({
    itemId: "user-2",
    text: "second question",
  });
  assert.equal(tracker.markNextRecordableTurn(), null);
  const firstUser = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "first question",
  });

  assert.equal(firstUser.clientTurnId, firstTurnId);
  assert.equal(secondUser.clientTurnId, secondTurnId);
  assert.deepEqual(tracker.priorTranscriptForTurn(secondTurnId), [
    { role: "user", content: "first question" },
    { role: "assistant", content: "first answer" },
  ]);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: firstTurnId,
    userText: "first question",
    assistantText: "first answer",
    toolCalls: [],
  });
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: secondTurnId,
    userText: "second question",
    assistantText: "second answer",
    toolCalls: [],
  });
});

test("keeps one client turn ID for duplicate final user items", () => {
  const tracker = createTurnTracker();
  const first = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "same final transcript",
  });
  const duplicate = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "same final transcript",
  });

  assert.equal(first.clientTurnId, "client-turn-1");
  assert.deepEqual(duplicate, { ...first, isNew: false });
});

test("maps tool follow-up responses and results to the original turn", () => {
  const tracker = createTurnTracker();
  const user = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "look this up",
  });
  tracker.responseCreated("response-tool-call");
  const correlatedTurnId = tracker.correlateToolCall("response-tool-call");
  tracker.toolCallStarted(correlatedTurnId);
  tracker.addFinalAssistantTranscript({
    responseId: "response-tool-call",
    text: "Let me check that.",
  });
  tracker.addToolResult(correlatedTurnId, {
    tool_name: "answer_grounded_lookup",
    status: "completed",
    output: { answer: "verified" },
  });
  tracker.expectNextResponseForTurn(correlatedTurnId);
  tracker.toolCallFinished(correlatedTurnId);
  tracker.responseFinished("response-tool-call");
  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.responseCreated("response-tool-follow-up");
  const followUpTurnId = tracker.correlateToolCall("response-tool-follow-up");
  tracker.toolCallStarted(followUpTurnId);
  tracker.addToolResult(followUpTurnId, {
    tool_name: "save_response_preference",
    status: "completed",
    output: { saved: true },
  });
  tracker.expectNextResponseForTurn(followUpTurnId);
  tracker.toolCallFinished(followUpTurnId);
  tracker.responseFinished("response-tool-follow-up");
  tracker.responseCreated("response-final-follow-up");
  tracker.addFinalAssistantTranscript({
    responseId: "response-final-follow-up",
    text: "Here is the verified answer.",
  });
  tracker.responseFinished("response-final-follow-up");

  assert.equal(correlatedTurnId, user.clientTurnId);
  assert.equal(followUpTurnId, user.clientTurnId);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: user.clientTurnId,
    userText: "look this up",
    assistantText: "Let me check that. Here is the verified answer.",
    toolCalls: [
      {
        tool_name: "answer_grounded_lookup",
        status: "completed",
        output: { answer: "verified" },
      },
      {
        tool_name: "save_response_preference",
        status: "completed",
        output: { saved: true },
      },
    ],
  });
});

test("uses the latest compatible turn when response IDs are absent", () => {
  const tracker = createTurnTracker();
  tracker.addFinalUserTranscript({ itemId: "user-1", text: "first" });
  tracker.responseCreated("response-1");
  tracker.responseFinished("response-1");
  const latest = tracker.addFinalUserTranscript({
    itemId: "user-2",
    text: "latest",
  });

  assert.equal(tracker.correlateToolCall(), latest.clientTurnId);
  tracker.addFinalAssistantTranscript({ text: "latest answer" });
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: latest.clientTurnId,
    userText: "latest",
    assistantText: "latest answer",
    toolCalls: [],
  });
});

test("removes only successful recordings and retains failed recordings", () => {
  const tracker = createTurnTracker();
  const first = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "first",
  });
  tracker.responseCreated("response-1");
  tracker.addFinalAssistantTranscript({
    responseId: "response-1",
    text: "first answer",
  });
  tracker.responseFinished("response-1");
  const second = tracker.addFinalUserTranscript({
    itemId: "user-2",
    text: "second",
  });
  tracker.responseCreated("response-2");
  tracker.addFinalAssistantTranscript({
    responseId: "response-2",
    text: "second answer",
  });
  tracker.responseFinished("response-2");

  assert.equal(tracker.markNextRecordableTurn().clientTurnId, first.clientTurnId);
  assert.equal(tracker.markNextRecordableTurn().clientTurnId, second.clientTurnId);
  tracker.recordingSucceeded(second.clientTurnId);
  tracker.recordingFailed(first.clientTurnId);

  assert.equal(tracker.markNextRecordableTurn().clientTurnId, first.clientTurnId);
  tracker.recordingSucceeded(first.clientTurnId);
  assert.equal(tracker.markNextRecordableTurn(), null);
});

test("does not let an incomplete wait-for-user turn block a later turn", () => {
  const tracker = createTurnTracker();
  tracker.addFinalUserTranscript({ itemId: "user-1", text: "pause here" });
  tracker.responseCreated("response-wait");
  tracker.correlateToolCall("response-wait");
  tracker.responseFinished("response-wait");

  const later = tracker.addFinalUserTranscript({
    itemId: "user-2",
    text: "continue now",
  });
  tracker.responseCreated("response-later");
  tracker.addFinalAssistantTranscript({
    responseId: "response-later",
    text: "continuing",
  });
  tracker.responseFinished("response-later");

  assert.equal(tracker.markNextRecordableTurn().clientTurnId, later.clientTurnId);
  assert.deepEqual(tracker.priorTranscriptForTurn(later.clientTurnId), []);
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
