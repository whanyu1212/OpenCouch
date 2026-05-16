export default function Loading() {
  return (
    <div className="flex min-h-screen flex-1 items-center justify-center bg-oc-bg px-6 text-oc-text">
      <div className="flex items-center gap-3 rounded-xl border border-oc-border bg-oc-bg-card px-4 py-3">
        <span className="relative flex h-3.5 w-3.5 items-center justify-center">
          <span className="absolute inline-flex h-2.5 w-2.5 animate-ping rounded-full bg-oc-cta opacity-50" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-oc-cta" />
        </span>
        <span className="font-mono text-[13px] text-oc-text-muted">
          loading
        </span>
      </div>
    </div>
  );
}
