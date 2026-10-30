import type { CrisisCatalog, CrisisChannel, CrisisResource } from "@/lib/crisis-catalog";

function describe(channel: CrisisChannel): string {
  const how = channel.kind === "sms" ? "Text message" : "Phone";
  return channel.label === "Call" ? how : `${how}: ${channel.label}`;
}

function Channel({ channel, service }: { channel: CrisisChannel; service: string }) {
  if (channel.kind === "whatsapp") {
    return (
      <li className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <a
          href={channel.url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`Message ${service} on WhatsApp`}
          className="rounded-full bg-accent px-4 py-1.5 text-sm font-medium text-accent-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          Open WhatsApp
        </a>
        <span className="text-sm text-ink-soft">
          {channel.label}, {channel.display}
        </span>
      </li>
    );
  }
  return (
    <li className="flex items-baseline gap-x-3">
      <span className="shrink-0 text-lg font-semibold tabular-nums">{channel.display}</span>
      <span className="min-w-0 text-sm text-ink-soft">{describe(channel)}</span>
    </li>
  );
}

function Service({ resource }: { resource: CrisisResource }) {
  return (
    <li className="py-3">
      <p className="font-medium">{resource.name}</p>
      {resource.hours && <p className="text-sm text-ink-soft">Open {resource.hours}</p>}
      <ul className="mt-2 space-y-2">
        {resource.channels.map((channel) => (
          <Channel
            key={`${channel.kind}-${channel.display}`}
            channel={channel}
            service={resource.name}
          />
        ))}
      </ul>
    </li>
  );
}

/**
 * The vetted crisis services the server chose for this turn, in its order.
 * Numbers are plain text so they show on every device; the model never writes them.
 */
export function CrisisCard({ keys, catalog }: { keys: readonly string[]; catalog: CrisisCatalog }) {
  const resources =
    catalog.status === "ready" ? keys.flatMap((key) => catalog.resources.get(key) ?? []) : [];

  return (
    <section
      aria-labelledby="crisis-card-title"
      className="mt-6 rounded-3xl border border-line px-5 pt-4 pb-1"
    >
      <h2 id="crisis-card-title" className="font-semibold">
        Reach someone now
      </h2>
      {resources.length === 0 ? (
        // The catalog hasn't loaded or can't be reached; the emergency number
        // must still show.
        <p className="pt-2 pb-3">
          If you&apos;re in danger, call <strong className="font-semibold">995</strong> now.
        </p>
      ) : (
        <ul className="divide-y divide-line">
          {resources.map((resource) => (
            <Service key={resource.key} resource={resource} />
          ))}
        </ul>
      )}
    </section>
  );
}
