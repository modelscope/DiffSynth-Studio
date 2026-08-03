import "./globals.css";
import type { Metadata, Viewport } from "next";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "DiffSynth-Studio 训练 UI",
  // description: "面向小白用户的 DiffSynth-Studio 训练 UI",
  icons: { icon: [] },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased">
        <div className="flex h-[100dvh] w-full overflow-hidden">
          <Sidebar />
          <main className="h-full min-w-0 flex-1 overflow-y-auto overflow-x-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
