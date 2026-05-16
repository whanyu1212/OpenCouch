import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-1 items-center justify-center bg-oc-bg px-6 text-oc-text">
      <div className="max-w-md rounded-xl border border-oc-border bg-oc-bg-card p-6">
        <p className="font-display text-xl text-oc-teal-900">Page not found</p>
        <p className="mt-2 text-[14px] leading-relaxed text-oc-text-muted">
          This OpenCouch view does not exist.
        </p>
        <Link
          href="/"
          className="mt-5 inline-flex rounded-lg border border-oc-teal-200 bg-oc-teal-50 px-4 py-2 text-[13px] font-medium text-oc-teal-800 transition-colors hover:bg-oc-teal-100"
        >
          Back to chat
        </Link>
      </div>
    </div>
  );
}
