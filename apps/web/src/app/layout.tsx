import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";

const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OpenCouch",
  description: "Mental health support with persistent memory",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geist.variable} h-full antialiased`}>
      <body className="min-h-full flex bg-oc-bg text-oc-text font-[family-name:var(--font-geist)]">
        <Sidebar />
        <main className="flex-1 flex flex-col">{children}</main>
      </body>
    </html>
  );
}
