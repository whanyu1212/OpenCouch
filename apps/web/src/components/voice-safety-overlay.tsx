"use client";

import { useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useSessionStore } from "@/lib/session";

export function VoiceSafetyOverlay() {
  const overlay = useSessionStore((state) => state.voiceSafetyOverlay);
  const dismiss = useSessionStore((state) => state.dismissVoiceSafetyOverlay);
  const finalizationStatus = useSessionStore(
    (state) => state.voiceFinalization.status
  );
  const finalizationBlocked = finalizationStatus === "in_progress";
  const router = useRouter();
  const pathname = usePathname();
  const dialogRef = useRef<HTMLElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (!overlay?.open) return;
    const previousFocus = document.activeElement;
    titleRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;

      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (
        document.activeElement instanceof Node &&
        !dialogRef.current?.contains(document.activeElement)
      ) {
        event.preventDefault();
        first.focus();
        return;
      }
      if (
        event.shiftKey &&
        (document.activeElement === first ||
          document.activeElement === titleRef.current)
      ) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      if (previousFocus instanceof HTMLElement) previousFocus.focus();
    };
  }, [dismiss, overlay?.clientTurnId, overlay?.open]);

  if (!overlay?.open) return null;

  const continueInText = () => {
    dismiss();
    router.push(
      finalizationStatus === "failed"
        ? pathname.startsWith("/voice/realtime-dev")
          ? pathname
          : "/voice"
        : "/"
    );
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-end justify-center bg-[rgba(21,32,29,0.56)] p-0 backdrop-blur-[2px] sm:items-center sm:p-6">
      <section
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="voice-safety-title"
        aria-describedby="voice-safety-description voice-safety-immediate-step"
        className="max-h-[94dvh] w-full max-w-[620px] overflow-y-auto rounded-t-[24px] border border-oc-line bg-[#fdfcfa] shadow-[0_30px_90px_-28px_rgba(9,30,27,0.72)] sm:rounded-[24px]"
      >
        <div className="border-b border-oc-line-2 bg-oc-primary-tint/55 px-5 py-5 sm:px-7 sm:py-6">
          <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-oc-primary">
            Immediate support
          </p>
          <h2
            ref={titleRef}
            tabIndex={-1}
            id="voice-safety-title"
            className="mt-2 font-display text-[27px] font-semibold leading-[1.12] text-oc-ink outline-none sm:text-[32px]"
          >
            {overlay.headline}
          </h2>
          <p
            id="voice-safety-description"
            className="mt-3 text-[14px] leading-6 text-oc-ink-2 sm:text-[15px]"
          >
            {overlay.validation}
          </p>
        </div>

        <div className="space-y-5 px-5 py-5 sm:px-7 sm:py-6">
          <div className="rounded-2xl border border-oc-cta/25 bg-oc-cta-subtle px-4 py-4">
            <p className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-oc-cta">
              Do this now
            </p>
            <p
              id="voice-safety-immediate-step"
              className="mt-2 text-[14px] font-medium leading-6 text-oc-ink"
            >
              {overlay.immediateStep}
            </p>
          </div>

          <section aria-labelledby="voice-safety-resources-title">
            <div className="flex items-center justify-between gap-3">
              <h3
                id="voice-safety-resources-title"
                className="font-display text-[20px] font-semibold text-oc-ink"
              >
                Support contacts
              </h3>
              {overlay.resourceStatus === "loading" && (
                <span className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-oc-muted">
                  checking
                </span>
              )}
            </div>

            {overlay.resourceStatus === "loading" ? (
              <div className="mt-3 flex items-center gap-3 rounded-xl border border-oc-line bg-white px-4 py-3" role="status">
                <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-oc-primary" />
                <p className="text-[13px] leading-5 text-oc-muted">
                  {overlay.message}
                </p>
              </div>
            ) : (
              <div className="mt-3 space-y-3">
                {overlay.message && (
                  <p className="rounded-xl border border-oc-line bg-white px-4 py-3 text-[13px] leading-5 text-oc-muted" role="status">
                    {overlay.message}
                  </p>
                )}
                {overlay.resources.map((resource, index) => (
                  <article
                    key={`${resource.name}:${resource.phone}:${resource.url}:${index}`}
                    className="rounded-xl border border-oc-primary/15 bg-white px-4 py-4"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <h4 className="text-[14px] font-semibold text-oc-ink">
                        {resource.name}
                      </h4>
                      {resource.region && (
                        <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-oc-muted">
                          {resource.region}
                        </span>
                      )}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-[13px] font-medium">
                      {resource.phone && (
                        <span className="text-oc-ink">{resource.phone}</span>
                      )}
                      {safeHttpUrl(resource.url) && (
                        <a
                          href={safeHttpUrl(resource.url) ?? undefined}
                          target="_blank"
                          rel="noreferrer"
                          className="break-all text-oc-primary underline decoration-oc-primary/30 underline-offset-4 hover:text-oc-primary-2"
                        >
                          {resource.url}
                        </a>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <div className="grid gap-2 border-t border-oc-line-2 pt-5 sm:grid-cols-[1fr_auto]">
            <button
              type="button"
              onClick={continueInText}
              disabled={finalizationBlocked}
              className="rounded-xl bg-oc-primary px-4 py-3 text-[14px] font-semibold text-[#f6f1e5] transition-colors hover:bg-oc-primary-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-oc-primary disabled:cursor-wait disabled:opacity-60"
            >
              {finalizationStatus === "failed"
                ? "Retry saving in Voice"
                : finalizationStatus === "in_progress"
                  ? "Preparing text chat..."
                  : "Continue in text"}
            </button>
            <button
              type="button"
              onClick={dismiss}
              className="rounded-xl border border-oc-line bg-white px-5 py-3 text-[14px] font-medium text-oc-muted transition-colors hover:bg-oc-surface-tint hover:text-oc-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-oc-primary"
            >
              Dismiss
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function safeHttpUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}
