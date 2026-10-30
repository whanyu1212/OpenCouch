import { describe, expect, it } from "vitest";

import { crisisResourcesAt, EMPTY_CRISIS_UI_STATE, parseCrisisUiState } from "./crisis-state";

describe("parseCrisisUiState", () => {
  it("checks every crisis UI field at the boundary", () => {
    expect(
      parseCrisisUiState({
        crisis_resources: ["sos", "imh"],
        crisis_resources_until: "2026-10-30T10:30:00Z",
        get_help_resources: ["sos"],
        crisis_check_in: "offer-1",
      }),
    ).toEqual({
      crisisResources: ["sos", "imh"],
      crisisResourcesUntil: Date.parse("2026-10-30T10:30:00Z"),
      getHelpResources: ["sos"],
      crisisCheckIn: "offer-1",
    });
  });

  it.each([
    { case: "an old response", state: {} },
    { case: "null", state: null },
    {
      case: "invalid fields",
      state: {
        crisis_resources: ["sos", 1],
        crisis_resources_until: "later",
        get_help_resources: "sos",
        crisis_check_in: 1,
      },
    },
  ])("uses safe defaults for $case", ({ state }) => {
    expect(parseCrisisUiState(state)).toEqual(EMPTY_CRISIS_UI_STATE);
  });

  it("keeps a null expiry for an ongoing crisis", () => {
    expect(
      parseCrisisUiState({
        crisis_resources: ["sos"],
        crisis_resources_until: null,
      }),
    ).toMatchObject({ crisisResources: ["sos"], crisisResourcesUntil: null });
  });
});

describe("crisisResourcesAt", () => {
  const state = parseCrisisUiState({
    crisis_resources: ["sos"],
    crisis_resources_until: "2026-10-30T10:30:00Z",
  });
  const expiry = Date.parse("2026-10-30T10:30:00Z");

  it("keeps pinned resources before their expiry", () => {
    expect(crisisResourcesAt(state, expiry - 1)).toEqual(["sos"]);
  });

  it("expires pinned resources at the supplied time", () => {
    expect(crisisResourcesAt(state, expiry)).toEqual([]);
  });
});
