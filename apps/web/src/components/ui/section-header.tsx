import type { ReactNode } from "react";

/**
 * The recurring mono / uppercase / wide-tracking eyebrow used above sections.
 */
export function SectionHeader({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`text-[10.5px] font-mono uppercase tracking-widest text-oc-text-dim ${className}`}
    >
      {children}
    </div>
  );
}
