import type { Metadata, Viewport } from "next";
import { Figtree } from "next/font/google";
import "./globals.css";

// The brand's wordmark face (assets/brand).
const figtree = Figtree({
  variable: "--font-figtree",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OpenCouch",
  description: "A guided self-help program for anxiety, with a coach.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#1c1c1c" },
  ],
  // Draw under the phone's notch and home bar; the layout pads for them.
  viewportFit: "cover",
  // Shrink the page when the on-screen keyboard opens, so the composer stays
  // above it.
  interactiveWidget: "resizes-content",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${figtree.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">{children}</body>
    </html>
  );
}
