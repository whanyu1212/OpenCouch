import type { ReactNode } from "react";
import { categoryColors } from "./memory-kinds";

export type ChipTone =
  | "teal"
  | "muted"
  | "green"
  | "red"
  | "amber"
  | "category";

const TONE_CLASSES: Record<Exclude<ChipTone, "category">, string> = {
  teal: "bg-oc-teal-50 text-oc-teal-700 border-oc-teal-200/60",
  muted: "bg-oc-warm-100 text-oc-warm-600 border-oc-warm-200",
  green: "bg-emerald-50 text-emerald-700 border-emerald-200/60",
  red: "bg-red-50 text-red-700 border-red-200/60",
  amber: "bg-amber-50 text-amber-700 border-amber-200/60",
};

/**
 * Small monospace label pill. Unifies the former `Tag` (memory page) and `Pill` (chat).
 * `tone="category"` derives its colors from the shared memory category map.
 */
export function Chip({
  children,
  tone = "muted",
  category,
  className = "",
}: {
  children: ReactNode;
  tone?: ChipTone;
  /** Required when tone="category"; selects colors from the shared category map. */
  category?: string;
  className?: string;
}) {
  const palette =
    tone === "category"
      ? (() => {
          const c = categoryColors(category ?? "");
          return `${c.bg} ${c.text} ${c.border}`;
        })()
      : TONE_CLASSES[tone];

  return (
    <span
      className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded-md border ${palette} ${className}`}
    >
      {children}
    </span>
  );
}
