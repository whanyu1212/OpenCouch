import type {
  RealtimeVoiceRecordedToolCall,
  RealtimeVoiceTurnOutcome,
  VoiceMemoryMode,
} from "./api";

export function buildRealtimeVoiceTurnRecordInput({
  threadId,
  userId,
  clientTurnId,
  userText,
  assistantText,
  memoryMode,
  toolCalls,
  outcome = "completed",
  interruptionToken,
  retryHandleId,
}: {
  threadId: string;
  userId?: string;
  clientTurnId?: string;
  userText: string;
  assistantText: string;
  memoryMode: VoiceMemoryMode;
  toolCalls?: RealtimeVoiceRecordedToolCall[];
  outcome?: RealtimeVoiceTurnOutcome;
  interruptionToken?: string;
  retryHandleId?: string;
}) {
  return {
    threadId,
    userId,
    ...(clientTurnId ? { clientTurnId } : {}),
    userText,
    assistantText,
    memoryMode,
    toolCalls: toolCalls || [],
    outcome,
    ...(interruptionToken ? { interruptionToken } : {}),
    ...(retryHandleId ? { retryHandleId } : {}),
  };
}

export function readLatestUserTranscriptDraft(
  drafts: ReadonlyMap<string, string>
): string {
  const values = Array.from(drafts.values());
  return values[values.length - 1] || "";
}

export interface RealtimeVoiceTrackedTurn {
  clientTurnId: string;
  userText: string;
  assistantText: string;
  toolCalls: RealtimeVoiceRecordedToolCall[];
  outcome?: RealtimeVoiceTurnOutcome;
  interruptionToken?: string;
}

export interface RealtimeVoiceSafetyInterruptionCleanup {
  clientTurnId: string;
  responseIds: string[];
  itemIds: string[];
}

type TrackedTurn = RealtimeVoiceTrackedTurn & {
  userItemId?: string;
  userTranscriptionFinished: boolean;
  awaitingInitialResponse: boolean;
  responseIds: Set<string>;
  activeResponseIds: Set<string>;
  activeToolCallCount: number;
  expectedResponseEventIds: Set<string>;
  assistantItemIds: Set<string>;
  safetyState: "unchecked" | "pending" | "released" | "interrupted";
  recording: boolean;
};

type FinalUserTurn = {
  clientTurnId: string;
  userText: string;
  isNew: boolean;
};

export class RealtimeVoiceTurnTracker {
  private readonly createClientTurnId: () => string;
  private readonly turns: TrackedTurn[] = [];
  private readonly responseTurns = new Map<string, TrackedTurn>();
  private readonly userItemTurns = new Map<string, TrackedTurn>();
  private readonly expectedResponseTurns = new Map<string, TrackedTurn>();
  private readonly ignoredResponseIds = new Set<string>();
  private readonly quarantinedResponseIds = new Set<string>();

  constructor(
    createClientTurnId: () => string = () => globalThis.crypto.randomUUID()
  ) {
    this.createClientTurnId = createClientTurnId;
  }

  responseCreated(
    responseId: string,
    requestEventId?: string
  ): string | undefined {
    if (this.quarantinedResponseIds.has(responseId)) return undefined;
    const existing = this.responseTurns.get(responseId);
    if (existing) return existing.clientTurnId;

    if (requestEventId) {
      const expected = this.takeExpectedResponse(requestEventId);
      if (!expected) {
        this.ignoredResponseIds.add(responseId);
        return undefined;
      }
      this.attachResponse(expected, responseId);
      return expected.clientTurnId;
    }

    const turn =
      this.turns.find(
        (candidate) => candidate.userItemId && candidate.responseIds.size === 0
      ) ??
      this.turns.find(
        (candidate) => candidate.userText && candidate.responseIds.size === 0
      ) ??
      this.createTurn();
    this.attachResponse(turn, responseId);
    return turn.clientTurnId;
  }

  userInputCommitted(itemId: string): string {
    const existing = this.userItemTurns.get(itemId);
    if (existing) return existing.clientTurnId;

    const turn =
      this.turns.find(
        (candidate) => !candidate.userItemId && candidate.responseIds.size === 0
      ) ?? this.createTurn();
    turn.userItemId = itemId;
    turn.userTranscriptionFinished = false;
    turn.awaitingInitialResponse = turn.responseIds.size === 0;
    this.userItemTurns.set(itemId, turn);
    return turn.clientTurnId;
  }

  finishUserTranscription(itemId: string | undefined): void {
    if (!itemId) return;
    const turn = this.userItemTurns.get(itemId);
    if (turn) turn.userTranscriptionFinished = true;
  }

  responseFinished(responseId: string): void {
    if (this.quarantinedResponseIds.has(responseId)) return;
    if (this.ignoredResponseIds.delete(responseId)) return;
    this.responseTurns.get(responseId)?.activeResponseIds.delete(responseId);
  }

  isResponseIgnored(responseId: string | undefined): boolean {
    return Boolean(
      responseId &&
        (this.ignoredResponseIds.has(responseId) ||
          this.quarantinedResponseIds.has(responseId))
    );
  }

  addFinalUserTranscript({
    itemId,
    text,
  }: {
    itemId?: string;
    text: string;
  }): FinalUserTurn {
    const existing = itemId ? this.userItemTurns.get(itemId) : undefined;
    if (existing) {
      existing.userTranscriptionFinished = true;
      const isNew = !existing.userText;
      if (isNew) existing.userText = text;
      return {
        clientTurnId: existing.clientTurnId,
        userText: existing.userText,
        isNew,
      };
    }

    const turn =
      this.turns.find((candidate) => !candidate.userText) ?? this.createTurn();
    turn.userText = text;
    turn.userTranscriptionFinished = true;
    if (itemId) {
      turn.userItemId = itemId;
      turn.awaitingInitialResponse = turn.responseIds.size === 0;
      this.userItemTurns.set(itemId, turn);
    }
    return { clientTurnId: turn.clientTurnId, userText: text, isNew: true };
  }

  addFinalAssistantTranscript({
    responseId,
    itemId,
    text,
  }: {
    responseId?: string;
    itemId?: string;
    text: string;
  }): string | undefined {
    const turn = this.turnForResponse(responseId, true);
    if (!turn) return undefined;
    if (itemId) turn.assistantItemIds.add(itemId);
    turn.awaitingInitialResponse = false;
    turn.assistantText = [turn.assistantText, text].filter(Boolean).join(" ");
    return turn.clientTurnId;
  }

  trackAssistantItem(responseId: string | undefined, itemId: string | undefined): void {
    if (!itemId) return;
    this.turnForResponse(responseId, false)?.assistantItemIds.add(itemId);
  }

  markSafetyPending(clientTurnId: string): boolean {
    const turn = this.turnById(clientTurnId);
    if (!turn || turn.safetyState === "interrupted") return false;
    turn.safetyState = "pending";
    return true;
  }

  releaseSafetyCheck(clientTurnId: string): boolean {
    const turn = this.turnById(clientTurnId);
    if (!turn || turn.safetyState !== "pending") return false;
    turn.safetyState = "released";
    return true;
  }

  attachLatestUserDraft(text: string): void {
    const normalized = text.trim();
    if (!normalized) return;
    const turn = [...this.turns].reverse().find((candidate) => !candidate.userText);
    if (!turn) return;
    turn.userText = normalized;
    turn.userTranscriptionFinished = true;
  }

  interruptForSafety(
    clientTurnId: string,
    interruptionToken: string
  ): RealtimeVoiceSafetyInterruptionCleanup | null {
    const targetIndex = this.turns.findIndex(
      (turn) => turn.clientTurnId === clientTurnId
    );
    if (targetIndex === -1) return null;

    const target = this.turns[targetIndex];
    if (target.safetyState === "interrupted") return null;

    const responseIds = new Set<string>();
    const itemIds = new Set<string>();
    const affectedTurns = this.turns.filter(
      (turn, index) => index >= targetIndex || turn.safetyState === "pending"
    );
    for (const turn of affectedTurns) {
      for (const responseId of turn.responseIds) {
        responseIds.add(responseId);
        this.quarantinedResponseIds.add(responseId);
      }
      for (const itemId of turn.assistantItemIds) itemIds.add(itemId);
      turn.assistantText = "";
      turn.activeResponseIds.clear();
      turn.awaitingInitialResponse = false;
      this.removeExpectedResponseTurn(turn);
      turn.safetyState = "interrupted";
      turn.userTranscriptionFinished = true;
      if (turn.userText) {
        if (turn === target) {
          turn.outcome = "safety_interrupted";
          turn.interruptionToken = interruptionToken;
        } else {
          turn.outcome = "connection_interrupted";
          turn.interruptionToken = undefined;
        }
      }
    }

    return {
      clientTurnId,
      responseIds: [...responseIds],
      itemIds: [...itemIds],
    };
  }

  failOpenPendingSafetyChecks(): string[] {
    const released: string[] = [];
    for (const turn of this.turns) {
      if (turn.safetyState !== "pending") continue;
      turn.safetyState = "released";
      released.push(turn.clientTurnId);
    }
    return released;
  }

  correlateToolCall(responseId?: string): string | undefined {
    return this.turnForResponse(responseId, false)?.clientTurnId;
  }

  toolCallStarted(clientTurnId: string | undefined): void {
    const turn = this.turnById(clientTurnId);
    if (turn) turn.activeToolCallCount += 1;
  }

  toolCallFinished(clientTurnId: string | undefined): void {
    const turn = this.turnById(clientTurnId);
    if (turn) turn.activeToolCallCount = Math.max(0, turn.activeToolCallCount - 1);
  }

  addToolResult(
    clientTurnId: string | undefined,
    toolCall: RealtimeVoiceRecordedToolCall
  ): void {
    const turn = this.turnById(clientTurnId);
    if (turn) turn.toolCalls.push(toolCall);
  }

  expectNextResponseForTurn(
    clientTurnId: string | undefined,
    requestEventId: string
  ): boolean {
    const turn = this.turnById(clientTurnId);
    if (!turn) return false;
    turn.expectedResponseEventIds.add(requestEventId);
    this.expectedResponseTurns.set(requestEventId, turn);
    return true;
  }

  failExpectedResponse(requestEventId: string): boolean {
    return this.takeExpectedResponse(requestEventId) !== undefined;
  }

  transportClosed(): void {
    for (const turn of this.turns) {
      turn.awaitingInitialResponse = false;
      turn.activeResponseIds.clear();
      turn.userTranscriptionFinished = true;
      for (const requestEventId of turn.expectedResponseEventIds) {
        this.expectedResponseTurns.delete(requestEventId);
      }
      turn.expectedResponseEventIds.clear();
    }
  }

  userTextForTurn(clientTurnId: string | undefined): string {
    return this.turnById(clientTurnId)?.userText ?? "";
  }

  latestUserText(): string {
    return [...this.turns].reverse().find((turn) => turn.userText)?.userText ?? "";
  }

  priorTranscriptForTurn(
    clientTurnId: string
  ): Array<{ role: "user" | "assistant"; content: string }> {
    const currentIndex = this.turns.findIndex(
      (turn) => turn.clientTurnId === clientTurnId
    );
    if (currentIndex <= 0) return [];

    return this.turns.slice(0, currentIndex).flatMap((turn) => {
      const entries: Array<{ role: "user" | "assistant"; content: string }> = [];
      if (turn.userText) entries.push({ role: "user", content: turn.userText });
      if (turn.assistantText) {
        entries.push({ role: "assistant", content: turn.assistantText });
      }
      return entries;
    });
  }

  markNextRecordableTurn(): RealtimeVoiceTrackedTurn | null {
    let turn: TrackedTurn | undefined;
    for (let index = 0; index < this.turns.length; ) {
      const candidate = this.turns[index];
      if (candidate.recording) {
        index += 1;
        continue;
      }
      if (!this.isSettled(candidate)) return null;
      if (!candidate.userText && candidate.userTranscriptionFinished) {
        this.removeTurn(candidate);
        continue;
      }
      if (!candidate.userText) return null;
      if (
        !candidate.assistantText &&
        candidate.outcome !== "connection_interrupted" &&
        candidate.outcome !== "safety_interrupted"
      ) {
        this.removeTurn(candidate);
        continue;
      }
      turn = candidate;
      break;
    }
    if (!turn) return null;

    turn.recording = true;
    return {
      clientTurnId: turn.clientTurnId,
      userText: turn.userText,
      assistantText: turn.assistantText,
      toolCalls: [...turn.toolCalls],
      ...(turn.outcome ? { outcome: turn.outcome } : {}),
      ...(turn.interruptionToken ? { interruptionToken: turn.interruptionToken } : {}),
    };
  }

  recordingSucceeded(clientTurnId: string): void {
    const turn = this.turnById(clientTurnId);
    if (!turn) return;
    this.removeTurn(turn);
  }

  recordingFailed(clientTurnId: string): void {
    const turn = this.turnById(clientTurnId);
    if (turn) turn.recording = false;
  }

  private turnForResponse(
    responseId: string | undefined,
    preferWithoutAssistant: boolean
  ): TrackedTurn | undefined {
    if (
      responseId &&
      (this.ignoredResponseIds.has(responseId) ||
        this.quarantinedResponseIds.has(responseId))
    ) {
      return undefined;
    }
    if (responseId) {
      const existing = this.responseTurns.get(responseId);
      if (existing) return existing;
    }

    const candidates = [...this.turns].reverse();
    const turn =
      (preferWithoutAssistant
        ? candidates.find((candidate) => candidate.userText && !candidate.assistantText)
        : candidates.find((candidate) => candidate.userText)) ?? candidates[0];
    if (turn && responseId) this.attachResponse(turn, responseId);
    return turn;
  }

  private createTurn(): TrackedTurn {
    const turn: TrackedTurn = {
      clientTurnId: this.createClientTurnId(),
      userText: "",
      assistantText: "",
      toolCalls: [],
      userTranscriptionFinished: false,
      awaitingInitialResponse: false,
      responseIds: new Set(),
      activeResponseIds: new Set(),
      activeToolCallCount: 0,
      expectedResponseEventIds: new Set(),
      assistantItemIds: new Set(),
      safetyState: "unchecked",
      recording: false,
    };
    this.turns.push(turn);
    return turn;
  }

  private attachResponse(turn: TrackedTurn, responseId: string): void {
    turn.awaitingInitialResponse = false;
    turn.responseIds.add(responseId);
    turn.activeResponseIds.add(responseId);
    this.responseTurns.set(responseId, turn);
  }

  private takeExpectedResponse(requestEventId: string): TrackedTurn | undefined {
    const turn = this.expectedResponseTurns.get(requestEventId);
    if (!turn) return undefined;
    this.expectedResponseTurns.delete(requestEventId);
    turn.expectedResponseEventIds.delete(requestEventId);
    return this.turns.includes(turn) ? turn : undefined;
  }

  private removeExpectedResponseTurn(turn: TrackedTurn): void {
    for (const requestEventId of turn.expectedResponseEventIds) {
      if (this.expectedResponseTurns.get(requestEventId) === turn) {
        this.expectedResponseTurns.delete(requestEventId);
      }
    }
    turn.expectedResponseEventIds.clear();
  }

  private removeTurn(turn: TrackedTurn): void {
    const index = this.turns.indexOf(turn);
    if (index !== -1) this.turns.splice(index, 1);
    for (const responseId of turn.responseIds) {
      if (this.responseTurns.get(responseId) === turn) {
        this.responseTurns.delete(responseId);
      }
    }
    if (turn.userItemId && this.userItemTurns.get(turn.userItemId) === turn) {
      this.userItemTurns.delete(turn.userItemId);
    }
    this.removeExpectedResponseTurn(turn);
  }

  private turnById(clientTurnId: string | undefined): TrackedTurn | undefined {
    if (!clientTurnId) return undefined;
    return this.turns.find((turn) => turn.clientTurnId === clientTurnId);
  }

  private isSettled(turn: TrackedTurn): boolean {
    return (
      !turn.awaitingInitialResponse &&
      turn.activeResponseIds.size === 0 &&
      turn.activeToolCallCount === 0 &&
      turn.expectedResponseEventIds.size === 0 &&
      turn.safetyState !== "pending"
    );
  }
}
