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
  readLatestUserTranscriptDraft,
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
    outcome: "completed",
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
    outcome: "completed",
  });
});

test("keeps a newer partial user draft when an earlier item completes", () => {
  const drafts = new Map([
    ["user-earlier", "Earlier partial"],
    ["user-newer", "Newer partial that must persist"],
  ]);

  drafts.delete("user-earlier");

  assert.equal(
    readLatestUserTranscriptDraft(drafts),
    "Newer partial that must persist"
  );
});

test("safety pending blocks persistence until continue releases the turn", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-safety-pending",
    text: "I need help",
  });
  tracker.markSafetyPending(turn.clientTurnId);
  tracker.responseCreated("response-safety-pending");
  tracker.addFinalAssistantTranscript({
    responseId: "response-safety-pending",
    itemId: "assistant-safety-pending",
    text: "I am here.",
  });
  tracker.responseFinished("response-safety-pending");

  assert.equal(tracker.markNextRecordableTurn(), null);
  assert.equal(tracker.releaseSafetyCheck(turn.clientTurnId), true);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: turn.clientTurnId,
    userText: "I need help",
    assistantText: "I am here.",
    toolCalls: [],
  });
});

test("manual fail-open releases all pending safety gates", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-fail-open",
    text: "Please continue",
  });
  tracker.markSafetyPending(turn.clientTurnId);
  tracker.responseCreated("response-fail-open");
  tracker.addFinalAssistantTranscript({
    responseId: "response-fail-open",
    text: "Continuing.",
  });
  tracker.responseFinished("response-fail-open");

  assert.deepEqual(tracker.failOpenPendingSafetyChecks(), [turn.clientTurnId]);
  assert.equal(tracker.markNextRecordableTurn()?.clientTurnId, turn.clientTurnId);
});

test("safety interruption quarantines target and later assistant drafts", () => {
  const tracker = createTurnTracker();
  const earlier = tracker.addFinalUserTranscript({
    itemId: "user-earlier",
    text: "Earlier question",
  });
  tracker.responseCreated("response-earlier");
  tracker.addFinalAssistantTranscript({
    responseId: "response-earlier",
    itemId: "assistant-earlier",
    text: "Earlier completed answer",
  });
  tracker.responseFinished("response-earlier");

  const target = tracker.addFinalUserTranscript({
    itemId: "user-target",
    text: "Target safety turn",
  });
  tracker.markSafetyPending(target.clientTurnId);
  tracker.responseCreated("response-target");
  tracker.addFinalAssistantTranscript({
    responseId: "response-target",
    itemId: "assistant-target",
    text: "Cancelled target draft",
  });
  tracker.toolCallStarted(target.clientTurnId);

  tracker.addFinalUserTranscript({ itemId: "user-later", text: "Later turn" });
  tracker.responseCreated("response-later");
  tracker.addFinalAssistantTranscript({
    responseId: "response-later",
    itemId: "assistant-later",
    text: "Cancelled later draft",
  });

  assert.deepEqual(tracker.interruptForSafety(target.clientTurnId, "proof-token"), {
    clientTurnId: target.clientTurnId,
    responseIds: ["response-target", "response-later"],
    itemIds: ["assistant-target", "assistant-later"],
  });
  assert.equal(tracker.isResponseIgnored("response-target"), true);
  assert.equal(tracker.isResponseIgnored("response-later"), true);
  assert.equal(
    tracker.addFinalAssistantTranscript({
      responseId: "response-target",
      text: "late target draft",
    }),
    undefined
  );

  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: earlier.clientTurnId,
    userText: "Earlier question",
    assistantText: "Earlier completed answer",
    toolCalls: [],
  });
  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.addToolResult(target.clientTurnId, {
    tool_name: "save_response_preference",
    status: "completed",
    output: { saved: true },
  });
  tracker.toolCallFinished(target.clientTurnId);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: target.clientTurnId,
    userText: "Target safety turn",
    assistantText: "",
    toolCalls: [
      {
        tool_name: "save_response_preference",
        status: "completed",
        output: { saved: true },
      },
    ],
    outcome: "safety_interrupted",
    interruptionToken: "proof-token",
  });
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: "client-turn-3",
    userText: "Later turn",
    assistantText: "",
    toolCalls: [],
    outcome: "connection_interrupted",
  });
});

test("a later interruption does not fail open an earlier pending safety turn", () => {
  const tracker = createTurnTracker();
  const earlier = tracker.addFinalUserTranscript({
    itemId: "user-earlier-pending",
    text: "Earlier pending turn",
  });
  tracker.markSafetyPending(earlier.clientTurnId);
  tracker.responseCreated("response-earlier-pending");
  tracker.addFinalAssistantTranscript({
    responseId: "response-earlier-pending",
    text: "Earlier unchecked draft",
  });
  tracker.responseFinished("response-earlier-pending");

  const target = tracker.addFinalUserTranscript({
    itemId: "user-target-interrupt",
    text: "Target crisis turn",
  });
  tracker.markSafetyPending(target.clientTurnId);
  tracker.responseCreated("response-target-interrupt");

  tracker.interruptForSafety(target.clientTurnId, "proof-token");
  tracker.failOpenPendingSafetyChecks();

  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: earlier.clientTurnId,
    userText: "Earlier pending turn",
    assistantText: "",
    toolCalls: [],
    outcome: "connection_interrupted",
  });
});

test("an interrupted partial turn retains draft evidence for settled tools", () => {
  const tracker = createTurnTracker();
  const target = tracker.addFinalUserTranscript({
    itemId: "user-target",
    text: "Target crisis turn",
  });
  tracker.markSafetyPending(target.clientTurnId);
  tracker.responseCreated("response-target");
  tracker.responseFinished("response-target");
  tracker.userInputCommitted("user-partial");
  tracker.responseCreated("response-partial");
  const partialTurnId = tracker.correlateToolCall("response-partial");
  tracker.toolCallStarted(partialTurnId);
  tracker.attachLatestUserDraft("Please remember that I need short replies");

  tracker.interruptForSafety(target.clientTurnId, "proof-token");
  tracker.addToolResult(partialTurnId, {
    tool_name: "save_response_preference",
    status: "completed",
    output: { saved: true },
  });
  tracker.toolCallFinished(partialTurnId);

  tracker.markNextRecordableTurn();
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: partialTurnId,
    userText: "Please remember that I need short replies",
    assistantText: "",
    toolCalls: [
      {
        tool_name: "save_response_preference",
        status: "completed",
        output: { saved: true },
      },
    ],
    outcome: "connection_interrupted",
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
  tracker.expectNextResponseForTurn(correlatedTurnId, "response-create-1");
  tracker.toolCallFinished(correlatedTurnId);
  tracker.responseFinished("response-tool-call");
  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.responseCreated("response-tool-follow-up", "response-create-1");
  const followUpTurnId = tracker.correlateToolCall("response-tool-follow-up");
  tracker.toolCallStarted(followUpTurnId);
  tracker.addToolResult(followUpTurnId, {
    tool_name: "save_response_preference",
    status: "completed",
    output: { saved: true },
  });
  tracker.expectNextResponseForTurn(followUpTurnId, "response-create-2");
  tracker.toolCallFinished(followUpTurnId);
  tracker.responseFinished("response-tool-follow-up");
  tracker.responseCreated("response-final-follow-up", "response-create-2");
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

test("discards settled turns whose user transcription failed", () => {
  const tracker = createTurnTracker();
  tracker.userInputCommitted("user-failed");
  tracker.responseCreated("response-failed-transcript");
  tracker.responseFinished("response-failed-transcript");
  tracker.finishUserTranscription("user-failed");

  tracker.userInputCommitted("user-valid");
  const valid = tracker.addFinalUserTranscript({
    itemId: "user-valid",
    text: "valid question",
  });
  tracker.responseCreated("response-valid");
  tracker.addFinalAssistantTranscript({
    responseId: "response-valid",
    text: "valid answer",
  });
  tracker.responseFinished("response-valid");

  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: valid.clientTurnId,
    userText: "valid question",
    assistantText: "valid answer",
    toolCalls: [],
  });
});

test("keeps waiting for an unsettled turn with no user transcript", () => {
  const tracker = createTurnTracker();
  tracker.userInputCommitted("user-pending");
  tracker.responseCreated("response-pending");

  tracker.userInputCommitted("user-valid");
  tracker.addFinalUserTranscript({ itemId: "user-valid", text: "valid question" });
  tracker.responseCreated("response-valid");
  tracker.addFinalAssistantTranscript({
    responseId: "response-valid",
    text: "valid answer",
  });
  tracker.responseFinished("response-valid");

  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.responseFinished("response-pending");
  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.finishUserTranscription("user-pending");
  assert.equal(tracker.markNextRecordableTurn()?.userText, "valid question");
});

test("records a finalized transcript when transport closes before response done", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "final question",
  });
  tracker.responseCreated("response-1");
  tracker.addFinalAssistantTranscript({
    responseId: "response-1",
    text: "final answer",
  });

  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.transportClosed();
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: turn.clientTurnId,
    userText: "final question",
    assistantText: "final answer",
    toolCalls: [],
  });
});

test("drops a user-only turn when transport closes", () => {
  const tracker = createTurnTracker();
  tracker.addFinalUserTranscript({ itemId: "user-1", text: "unfinished" });
  tracker.responseCreated("response-1");

  tracker.transportClosed();
  assert.equal(tracker.markNextRecordableTurn(), null);
});

test("releases only the rejected follow-up response expectation", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "look this up",
  });
  tracker.responseCreated("response-tool");
  tracker.addFinalAssistantTranscript({
    responseId: "response-tool",
    text: "I could not finish the lookup.",
  });
  tracker.expectNextResponseForTurn(turn.clientTurnId, "response-create-1");
  tracker.expectNextResponseForTurn(turn.clientTurnId, "response-create-2");
  tracker.responseFinished("response-tool");

  assert.equal(tracker.failExpectedResponse("unrelated-event"), false);
  assert.equal(tracker.failExpectedResponse("response-create-1"), true);
  assert.equal(tracker.markNextRecordableTurn(), null);
  assert.equal(tracker.failExpectedResponse("response-create-2"), true);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: turn.clientTurnId,
    userText: "look this up",
    assistantText: "I could not finish the lookup.",
    toolCalls: [],
  });
});

test("quarantines a late response after its expectation expires", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "look this up",
  });
  tracker.responseCreated("response-tool");
  tracker.expectNextResponseForTurn(turn.clientTurnId, "response-create-1");
  tracker.responseFinished("response-tool");
  tracker.failExpectedResponse("response-create-1");

  assert.equal(
    tracker.responseCreated("response-late", "response-create-1"),
    undefined
  );
  assert.equal(tracker.isResponseIgnored("response-late"), true);
  assert.equal(
    tracker.addFinalAssistantTranscript({
      responseId: "response-late",
      text: "late answer",
    }),
    undefined
  );
  tracker.responseFinished("response-late");
  assert.equal(tracker.isResponseIgnored("response-late"), false);
});

test("terminal transport closure preserves an active tool barrier", () => {
  const tracker = createTurnTracker();
  const turn = tracker.addFinalUserTranscript({
    itemId: "user-1",
    text: "look this up",
  });
  tracker.responseCreated("response-tool");
  tracker.addFinalAssistantTranscript({
    responseId: "response-tool",
    text: "I started the lookup.",
  });
  tracker.toolCallStarted(turn.clientTurnId);

  tracker.transportClosed();
  assert.equal(tracker.markNextRecordableTurn(), null);
  tracker.toolCallFinished(turn.clientTurnId);
  assert.deepEqual(tracker.markNextRecordableTurn(), {
    clientTurnId: turn.clientTurnId,
    userText: "look this up",
    assistantText: "I started the lookup.",
    toolCalls: [],
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
