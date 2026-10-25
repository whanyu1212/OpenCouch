"use client";

import { CrisisCard } from "@/components/chat/crisis-card";
import { useCrisisCatalog } from "@/lib/crisis-catalog";

/** Every vetted crisis service, shown before sign-in so no one meets a wall first. */
export function CrisisServices() {
  const catalog = useCrisisCatalog(true);
  const keys = catalog.status === "ready" ? [...catalog.resources.keys()] : [];

  return (
    <div className="mt-12">
      <p className="text-ink-soft">
        Need to talk to someone right now? You don&apos;t need an account for this. If
        you&apos;re in danger, call <strong className="font-semibold text-ink">995</strong> now.
      </p>
      <CrisisCard keys={keys} catalog={catalog} />
    </div>
  );
}
