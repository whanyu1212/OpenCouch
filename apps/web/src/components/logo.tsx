"use client";

/**
 * OpenCouch logo — matches the Docusaurus docs site favicon.
 *
 * Two variants:
 * - "color" (default): uses the brand teal palette, matches docs site
 * - "mono": uses currentColor for embedding on colored backgrounds
 *
 * Usage:
 *   <CouchLogo className="w-6 h-6" />
 *   <CouchLogo variant="mono" className="w-5 h-5 text-white" />
 */
export function CouchLogo({
  className,
  variant = "color",
}: {
  className?: string;
  variant?: "color" | "mono";
}) {
  if (variant === "mono") {
    return (
      <svg
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
      >
        <rect x="4" y="14" width="24" height="8" rx="3" fill="currentColor" />
        <path d="M6 14V9a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v5" fill="currentColor" opacity="0.6" />
        <rect x="2" y="11" width="4" height="11" rx="2" fill="currentColor" />
        <rect x="26" y="11" width="4" height="11" rx="2" fill="currentColor" />
        <rect x="7" y="22" width="2.5" height="4" rx="1" fill="currentColor" opacity="0.5" />
        <rect x="22.5" y="22" width="2.5" height="4" rx="1" fill="currentColor" opacity="0.5" />
      </svg>
    );
  }

  // Color variant — exact match with docs/static/img/favicon.svg
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <rect x="4" y="14" width="24" height="8" rx="3" fill="#215f5a" />
      <path d="M6 14V9a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v5" fill="#3d8b84" />
      <rect x="2" y="11" width="4" height="11" rx="2" fill="#215f5a" />
      <rect x="26" y="11" width="4" height="11" rx="2" fill="#215f5a" />
      <rect x="7" y="22" width="2.5" height="4" rx="1" fill="#78b8af" />
      <rect x="22.5" y="22" width="2.5" height="4" rx="1" fill="#78b8af" />
    </svg>
  );
}
