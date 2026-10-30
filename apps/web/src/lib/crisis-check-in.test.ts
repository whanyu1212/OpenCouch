import { describe, expect, it } from "vitest";

import {
  CRISIS_CHECK_IN_CHOICES,
  crisisCheckInRunParameters,
  crisisCheckInText,
} from "./crisis-check-in";

describe("crisis check-in", () => {
  it("uses the exact screened text for each answer", () => {
    expect(CRISIS_CHECK_IN_CHOICES).toEqual([
      { answer: "safe", text: "I'm safe right now" },
      { answer: "not_sure", text: "I'm not sure" },
      { answer: "need_help", text: "I need help now" },
    ]);
    expect(crisisCheckInText("need_help")).toBe("I need help now");
  });

  it("puts the opaque offer and answer only in forwarded properties", () => {
    expect(crisisCheckInRunParameters("offer-1", "not_sure")).toEqual({
      forwardedProps: {
        crisis_check_in: { offer_id: "offer-1", answer: "not_sure" },
      },
    });
  });
});
