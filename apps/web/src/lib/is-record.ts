/**
 * Whether untrusted JSON is an object, so its fields can be read and checked.
 * A missing field reads as `undefined`.
 */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
