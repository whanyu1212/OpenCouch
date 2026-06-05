import type { ReactNode } from "react";
import { createElement as h } from "react";

export interface ToneClasses {
  bg: string;
  text: string;
  border: string;
}

/**
 * Per-category color identity for semantic memory facts.
 * Single source of truth — both the memory panel and the memory page read from here.
 * Class strings are written in full (not concatenated) so Tailwind's JIT can see them.
 */
export const CATEGORY_COLORS: Record<string, ToneClasses> = {
  loss: { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200/60" },
  preference: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200/60" },
  coping_strategy: { bg: "bg-teal-50", text: "text-teal-700", border: "border-teal-200/60" },
  relationship: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200/60" },
  trigger: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200/60" },
  goal: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200/60" },
  context: { bg: "bg-indigo-50", text: "text-indigo-700", border: "border-indigo-200/60" },
};

export const DEFAULT_CATEGORY_COLORS: ToneClasses = {
  bg: "bg-oc-warm-100",
  text: "text-oc-warm-600",
  border: "border-oc-warm-200",
};

export function categoryColors(category: string): ToneClasses {
  return CATEGORY_COLORS[category] ?? DEFAULT_CATEGORY_COLORS;
}

export function formatCategory(category: string): string {
  return category.replace(/_/g, " ");
}

/** Memory kinds — the three top-level buckets surfaced on the Memory page. */
export type MemoryKind = "facts" | "sessions" | "rules";

export interface MemoryKindIdentity {
  label: string;
  /** Accent color used for the count-card top rule and icon chip. */
  accent: ToneClasses;
  /** Top-rule background class (solid swatch for the hairline accent). */
  rail: string;
  icon: ReactNode;
}

const iconProps = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: "1.6",
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  className: "w-4 h-4",
};

// Icons authored with createElement so this stays a .ts module (no JSX runtime needed here).
export const MEMORY_KINDS: Record<MemoryKind, MemoryKindIdentity> = {
  facts: {
    label: "facts",
    accent: { bg: "bg-oc-teal-50", text: "text-oc-teal-700", border: "border-oc-teal-200/60" },
    rail: "bg-oc-teal-400",
    icon: h("svg", iconProps, h("circle", { cx: 12, cy: 12, r: 9 }), h("path", { d: "M9 12l2 2 4-4" })),
  },
  sessions: {
    label: "sessions",
    accent: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200/60" },
    rail: "bg-oc-cta",
    icon: h("svg", iconProps, h("circle", { cx: 12, cy: 12, r: 9 }), h("path", { d: "M12 7v5l3 2" })),
  },
  rules: {
    label: "rules",
    accent: { bg: "bg-oc-teal-100", text: "text-oc-teal-800", border: "border-oc-teal-300/60" },
    rail: "bg-oc-teal-600",
    icon: h(
      "svg",
      iconProps,
      h("path", { d: "M4 6h16M4 12h10M4 18h7" }),
    ),
  },
};

/**
 * The backend reports counts under storage-layer names (the CoALA memory
 * taxonomy); the product UI uses friendlier names. Map either onto one identity.
 */
const KIND_ALIASES: Record<string, MemoryKind> = {
  facts: "facts",
  semantic: "facts",
  sessions: "sessions",
  episodic: "sessions",
  rules: "rules",
  procedural: "rules",
};

/** Resolve a storage-layer or product name to its canonical MemoryKind, or null. */
export function memoryKindName(kind: string): MemoryKind | null {
  return KIND_ALIASES[kind] ?? null;
}

export function memoryKind(kind: string): MemoryKindIdentity | null {
  const resolved = memoryKindName(kind);
  return resolved ? MEMORY_KINDS[resolved] : null;
}

/**
 * Best-effort sentiment from a free-text mood string → a tone for the timeline node.
 * Moods are author-written and unbounded, so this is keyword matching with a neutral fallback.
 */
const MOOD_NEGATIVE = /\b(anx|fear|afraid|panic|heavy|low|sad|grief|angry|overwhelm|tense|stress|numb|hopeless|drained|exhaust|tight|frustrat)\w*/i;
const MOOD_POSITIVE = /\b(calm|calmer|light|lighter|relief|relieved|hope|hopeful|settl|steady|grounded|better|ease|eased|peace|clear|warm|content|rest)\w*/i;

export type MoodTone = "negative" | "positive" | "neutral";

export function moodTone(mood: string | null | undefined): MoodTone {
  if (!mood) return "neutral";
  if (MOOD_POSITIVE.test(mood)) return "positive";
  if (MOOD_NEGATIVE.test(mood)) return "negative";
  return "neutral";
}

/** Tailwind background class for a timeline node dot, by mood sentiment. */
export function moodDotClass(mood: string | null | undefined): string {
  switch (moodTone(mood)) {
    case "positive":
      return "bg-oc-teal-400";
    case "negative":
      return "bg-oc-cta";
    default:
      return "bg-oc-warm-300";
  }
}
