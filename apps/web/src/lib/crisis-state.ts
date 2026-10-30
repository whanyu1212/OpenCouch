import { isRecord } from "./is-record";

/** Server-owned crisis UI state, checked once where it enters the client. */
export type CrisisUiState = {
  crisisResources: readonly string[];
  crisisResourcesUntil: number | null;
  getHelpResources: readonly string[];
  crisisCheckIn: string | null;
};

export const EMPTY_CRISIS_UI_STATE: CrisisUiState = {
  crisisResources: [],
  crisisResourcesUntil: null,
  getHelpResources: [],
  crisisCheckIn: null,
};

function stringList(value: unknown): string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string") ? value : [];
}

/** Read a state snapshot or history response, with old-response defaults. */
export function parseCrisisUiState(value: unknown): CrisisUiState {
  if (!isRecord(value)) return EMPTY_CRISIS_UI_STATE;
  const {
    crisis_resources: resources,
    crisis_resources_until: resourcesUntil,
    get_help_resources: getHelpResources,
    crisis_check_in: crisisCheckIn,
  } = value;
  const expiry = typeof resourcesUntil === "string" ? Date.parse(resourcesUntil) : Number.NaN;
  return {
    crisisResources: stringList(resources),
    crisisResourcesUntil: Number.isNaN(expiry) ? null : expiry,
    getHelpResources: stringList(getHelpResources),
    crisisCheckIn: typeof crisisCheckIn === "string" ? crisisCheckIn : null,
  };
}

/** Resources still pinned at this instant; expiry is exclusive. */
export function crisisResourcesAt(state: CrisisUiState, now: number): readonly string[] {
  return state.crisisResourcesUntil === null || now < state.crisisResourcesUntil
    ? state.crisisResources
    : [];
}
