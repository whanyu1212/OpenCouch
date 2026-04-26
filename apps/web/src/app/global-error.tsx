"use client";

export default function GlobalError({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          minHeight: "100vh",
          margin: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#f8f5ef",
          color: "#183b3a",
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }}
      >
        <main
          style={{
            width: "min(100% - 48px, 420px)",
            border: "1px solid #d8d0c3",
            borderRadius: 12,
            background: "#fffdf8",
            padding: 24,
          }}
        >
          <title>OpenCouch error</title>
          <p style={{ margin: 0, fontSize: 22, color: "#215f5a" }}>
            Something went wrong
          </p>
          <p style={{ margin: "10px 0 0", fontSize: 14, lineHeight: 1.6 }}>
            The web app hit an unexpected error while loading the main layout.
          </p>
          {error.digest ? (
            <p
              style={{
                margin: "14px 0 0",
                fontFamily:
                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
                fontSize: 11,
                color: "#7b7164",
              }}
            >
              digest: {error.digest}
            </p>
          ) : null}
          <button
            type="button"
            onClick={() => unstable_retry()}
            style={{
              marginTop: 20,
              border: "1px solid #9ecbc5",
              borderRadius: 8,
              background: "#e5f5f2",
              color: "#215f5a",
              cursor: "pointer",
              padding: "9px 14px",
              fontSize: 13,
              fontWeight: 600,
            }}
          >
            Try again
          </button>
        </main>
      </body>
    </html>
  );
}
