import { describe, expect, it } from "vitest";

import { parseCrisisCatalog } from "./crisis-catalog";

const call = { kind: "call", label: "Call", display: "1767" };
const sms = { kind: "sms", label: "Text", display: "9151 1767" };
const whatsapp = {
  kind: "whatsapp",
  label: "WhatsApp",
  display: "9151 1767",
  url: "https://wa.me/6591511767",
};
const resource = {
  key: "sos",
  name: "Samaritans of Singapore",
  hours: "24 hours",
  channels: [call],
};

const ENTRY_ERROR = "Catalog response had an entry not in the expected shape";

describe("parseCrisisCatalog", () => {
  it("keys resources by catalog key", () => {
    const other = { key: "imh", name: "IMH Helpline", hours: null, channels: [sms, whatsapp] };

    const catalog = parseCrisisCatalog({ resources: [resource, other] });

    expect(catalog).toEqual(
      new Map<string, unknown>([
        ["sos", resource],
        ["imh", other],
      ]),
    );
  });

  it.each([
    { case: "string", hours: "24 hours", expected: "24 hours" },
    { case: "missing", hours: undefined, expected: null },
    { case: "not a string", hours: 24, expected: null },
  ])("reads hours when $case", ({ hours, expected }) => {
    const catalog = parseCrisisCatalog({ resources: [{ ...resource, hours }] });

    expect(catalog.get("sos")?.hours).toBe(expected);
  });

  it.each([
    { case: "null", body: null },
    { case: "not an object", body: "resources" },
    { case: "no resources", body: {} },
    { case: "resources not a list", body: { resources: resource } },
  ])("rejects a response with $case", ({ body }) => {
    expect(() => parseCrisisCatalog(body)).toThrow(
      "Catalog response was not in the expected shape",
    );
  });

  it.each([
    { case: "null", entry: null },
    { case: "not an object", entry: "sos" },
    { case: "no key", entry: { name: "SOS", channels: [] } },
    { case: "a non-string key", entry: { key: 1, name: "SOS", channels: [] } },
    { case: "no name", entry: { key: "sos", channels: [] } },
    { case: "a non-string name", entry: { key: "sos", name: 1, channels: [] } },
    { case: "no channels", entry: { key: "sos", name: "SOS" } },
    { case: "channels not a list", entry: { key: "sos", name: "SOS", channels: call } },
  ])("rejects the catalog for a resource with $case", ({ entry }) => {
    expect(() => parseCrisisCatalog({ resources: [resource, entry] })).toThrow(ENTRY_ERROR);
  });

  it.each([
    { case: "null", channel: null },
    { case: "not an object", channel: "call" },
    { case: "no kind", channel: { label: "Call", display: "1767" } },
    { case: "an unknown kind", channel: { ...call, kind: "email" } },
    { case: "no label", channel: { kind: "call", display: "1767" } },
    { case: "a non-string label", channel: { ...call, label: 1 } },
    { case: "no display", channel: { kind: "call", label: "Call" } },
    { case: "a non-string display", channel: { ...call, display: 1767 } },
    { case: "a WhatsApp with no url", channel: { ...call, kind: "whatsapp" } },
    { case: "a non-string url", channel: { ...whatsapp, url: 1 } },
    { case: "a url off wa.me", channel: { ...whatsapp, url: "https://example.com/6591511767" } },
    { case: "a plain-http url", channel: { ...whatsapp, url: "http://wa.me/6591511767" } },
  ])("rejects the catalog for a channel with $case", ({ channel }) => {
    const entry = { ...resource, channels: [call, channel] };

    expect(() => parseCrisisCatalog({ resources: [entry] })).toThrow(ENTRY_ERROR);
  });
});
