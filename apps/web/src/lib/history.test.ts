import { describe, expect, it } from "vitest";

import { parseHistory } from "./history";

const question = { id: "m1", role: "user", content: "Is this normal?", interrupted: false };
const reply = { id: "m2", role: "assistant", content: "Let's look at it.", interrupted: true };

const SHAPE_ERROR = "History response was not in the expected shape";

describe("parseHistory", () => {
  it("splits messages, cut-off replies and the card", () => {
    const history = parseHistory({ messages: [question, reply], crisis_resources: ["sos"] });

    expect(history).toEqual({
      messages: [
        { id: "m1", role: "user", content: "Is this normal?" },
        { id: "m2", role: "assistant", content: "Let's look at it." },
      ],
      interrupted: new Set(["m2"]),
      crisisResources: ["sos"],
    });
  });

  it("reads no card when none is saved", () => {
    expect(parseHistory({ messages: [question] }).crisisResources).toEqual([]);
  });

  it.each([
    { case: "null", body: null },
    { case: "not an object", body: "messages" },
    { case: "no messages", body: {} },
    { case: "messages not a list", body: { messages: question } },
  ])("rejects a response with $case", ({ body }) => {
    expect(() => parseHistory(body)).toThrow(SHAPE_ERROR);
  });

  it.each([
    { case: "null", message: null },
    { case: "not an object", message: "Is this normal?" },
    { case: "no id", message: { role: "user", content: "Hi", interrupted: false } },
    { case: "a non-string id", message: { ...question, id: 1 } },
    { case: "a system role", message: { ...question, role: "system" } },
    { case: "no role", message: { id: "m1", content: "Hi", interrupted: false } },
    { case: "non-string content", message: { ...question, content: ["Is this normal?"] } },
    { case: "no content", message: { id: "m1", role: "user", interrupted: false } },
    { case: "a non-boolean interrupted", message: { ...question, interrupted: "false" } },
    { case: "no interrupted", message: { id: "m1", role: "user", content: "Hi" } },
  ])("rejects the history for a message with $case", ({ message }) => {
    expect(() => parseHistory({ messages: [reply, message] })).toThrow(SHAPE_ERROR);
  });
});
