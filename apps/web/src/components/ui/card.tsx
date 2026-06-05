import type { ReactNode } from "react";

export type CardVariant = "surface" | "elevated" | "accent";

const VARIANT_CLASSES: Record<CardVariant, string> = {
  // Default warm card on the page background.
  surface: "bg-oc-bg-card border-oc-border",
  // Brighter, lifted card — for items that should read as foreground.
  elevated: "bg-white border-oc-border shadow-sm",
  // Teal-tinted card for emphasized content (e.g. style rules).
  accent: "bg-oc-teal-50 border-oc-teal-200/60",
};

/**
 * Shared card surface. Replaces the repeated
 * `rounded-xl border ... p-5 hover:border-oc-border-strong shadow-sm` pattern.
 */
export function Card({
  children,
  variant = "surface",
  interactive = false,
  className = "",
  ...rest
}: {
  children: ReactNode;
  variant?: CardVariant;
  /** Adds hover affordance (border emphasis + pointer). */
  interactive?: boolean;
  className?: string;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`rounded-xl border ${VARIANT_CLASSES[variant]} ${
        interactive
          ? "transition-colors hover:border-oc-border-strong cursor-pointer"
          : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}
