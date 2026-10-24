import { memo } from "react";
import Markdown, { type Components } from "react-markdown";

// The coach writes short replies: paragraphs, emphasis and simple lists.
// Anything else (headings, images, tables, raw HTML) is shown as plain text.
const ALLOWED = ["p", "strong", "em", "ul", "ol", "li", "a", "br", "blockquote", "code"];

const components: Components = {
  p: ({ children }) => <p className="[&:not(:first-child)]:mt-3">{children}</p>,
  ul: ({ children }) => <ul className="mt-3 list-disc space-y-1 pl-6">{children}</ul>,
  ol: ({ children }) => <ol className="mt-3 list-decimal space-y-1 pl-6">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="mt-3 border-l-2 border-accent-wash pl-4 text-ink-soft">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline decoration-accent-wash decoration-2 underline-offset-2"
    >
      {children}
    </a>
  ),
};

// Memoised: while a reply streams, only that reply's markdown is re-parsed.
export const CoachReply = memo(function CoachReply({
  text,
  interrupted,
}: {
  text: string;
  interrupted: boolean;
}) {
  return (
    <div className="max-w-xl">
      <Markdown allowedElements={ALLOWED} unwrapDisallowed components={components}>
        {text}
      </Markdown>
      {interrupted && <p className="mt-2 text-sm text-ink-soft">This reply was cut off.</p>}
    </div>
  );
});
