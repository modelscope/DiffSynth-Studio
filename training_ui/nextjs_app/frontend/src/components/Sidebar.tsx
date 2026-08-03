"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { label: "总览", href: "/dashboard", icon: "grid" },
  { label: "数据集", href: "/datasets", icon: "folder" },
  { label: "新建任务", href: "/jobs/new", icon: "plus" },
  { label: "任务管理", href: "/jobs", icon: "list" },
  { label: "设置", href: "/settings", icon: "cog" },
];

const LOGO_SRC = `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/ModelScopeIcon.png`;

function isNavActive(pathname: string, href: string) {
  if (pathname === href) return true;
  if (href === "/jobs/new") return pathname.startsWith("/jobs/new/");
  if (href === "/jobs") return pathname.startsWith("/jobs/") && !pathname.startsWith("/jobs/new");
  return pathname.startsWith(href + "/");
}

function Icon({ name, className }: { name: string; className?: string }) {
  const common = "w-4 h-4 " + (className || "");
  switch (name) {
    case "grid":
      return (
        <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h6v6H4V6zm10 0h6v6h-6V6zM4 16h6v4H4v-4zm10-4h6v8h-6v-8z" />
        </svg>
      );
    case "folder":
      return (
        <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
        </svg>
      );
    case "plus":
      return (
        <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14M5 12h14" />
        </svg>
      );
    case "list":
      return (
        <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
      );
    case "cog":
      return (
        <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <circle cx="12" cy="12" r="3" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1A1.7 1.7 0 004.6 9a1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z" />
        </svg>
      );
    default:
      return null;
  }
}

export function Sidebar() {
  const pathname = usePathname() || "";

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-950 p-4">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-14 shrink-0 items-center justify-center overflow-hidden">
            <Image
              src={LOGO_SRC}
              alt="ModelScope Logo"
              width={200}
              height={108}
              className="h-full w-full object-contain"
              priority
            />
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight text-white">DiffSynth-Studio</div>
            <div className="text-[10px] text-slate-500">Training UI</div>
          </div>
        </div>
      </div>

      <nav className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
        {NAV.map((item) => {
          const active = isNavActive(pathname, item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={
                "flex items-center gap-2.5 rounded-md border px-3 py-2 text-sm transition-colors " +
                (active
                  ? "border-slate-700 bg-slate-800 text-white shadow-sm"
                  : "border-transparent text-slate-400 hover:border-slate-800 hover:bg-slate-900 hover:text-slate-100")
              }
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
