"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-screen flex-1 items-center justify-center bg-oc-bg px-6 text-oc-text">
      <div className="max-w-md rounded-xl border border-oc-border bg-oc-bg-card p-6">
        <p className="font-display text-xl text-oc-teal-900">
          Something went wrong
        </p>
        <p className="mt-2 text-[14px] leading-relaxed text-oc-text-muted">
          The web app hit an unexpected error while rendering this view.
        </p>
        {error.digest ? (
          <p className="mt-3 font-mono text-[11px] text-oc-text-dim">
            digest: {error.digest}
          </p>
        ) : null}
        <button
          type="button"
          onClick={reset}
          className="mt-5 rounded-lg border border-oc-teal-200 bg-oc-teal-50 px-4 py-2 text-[13px] font-medium text-oc-teal-800 transition-colors hover:bg-oc-teal-100"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
