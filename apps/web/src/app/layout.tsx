import type { Metadata } from "next";
import { Geist, JetBrains_Mono } from "next/font/google";
import { DM_Serif_Display } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/app-shell";

const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
});

const dmSerif = DM_Serif_Display({
  variable: "--font-dm-serif",
  weight: "400",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OpenCouch",
  description: "Mental health support with persistent memory",
  robots: {
    index: false,
    follow: false,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geist.variable} ${dmSerif.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="min-h-full flex bg-oc-bg text-oc-text font-body">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
