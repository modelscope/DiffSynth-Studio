"use client";

import React from "react";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
  padded = true,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <div
      className={
        "min-w-0 overflow-hidden rounded-lg border border-slate-800 bg-slate-900/95 " +
        "shadow-[0_8px_24px_rgba(0,0,0,0.16)] " +
        className
      }
    >
      {(title || actions) && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 bg-slate-950/30 px-4 py-3">
          <div className="min-w-0">
            {title && <div className="text-sm font-semibold text-slate-100">{title}</div>}
            {subtitle && <div className="text-xs text-slate-300 mt-0.5">{subtitle}</div>}
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </div>
      )}
      <div className={padded ? "p-3 sm:p-4" : "overflow-x-auto"}>{children}</div>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  disabled = false,
  className = "",
  type = "button",
  title,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger" | "ghost" | "outline";
  size?: "sm" | "md";
  disabled?: boolean;
  className?: string;
  type?: "button" | "submit";
  title?: string;
}) {
  const base =
    "inline-flex items-center gap-1.5 rounded-md font-medium transition-colors " +
    "disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none " +
    "focus:ring-2 focus:ring-blue-500/40";
  const sizes = {
    sm: "px-2.5 py-1 text-xs",
    md: "px-3 py-1.5 text-sm",
  }[size];
  const styles = {
    primary: "border border-blue-500 bg-blue-600 text-white shadow-sm shadow-blue-950/30 hover:bg-blue-500",
    secondary: "bg-slate-800 hover:bg-slate-700 text-slate-100",
    danger: "bg-red-600 hover:bg-red-500 text-white",
    ghost: "text-slate-300 hover:bg-slate-800 hover:text-slate-50",
    outline: "border border-slate-700 hover:bg-slate-800 text-slate-100",
  }[variant];
  return (
    <button
      type={type}
      title={title}
      className={`${base} ${sizes} ${styles} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  children,
  hint,
  required = false,
}: {
  label: string;
  children: React.ReactNode;
  hint?: React.ReactNode;
  required?: boolean;
}) {
  return (
    <div className="mb-4">
      <div className="text-xs font-semibold text-slate-200 mb-1.5 flex items-center gap-1">
        <span>{label}</span>
        {required && <span className="text-red-400">*</span>}
      </div>
      {children}
      {hint && <div className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">{hint}</div>}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    created: "bg-slate-700 text-slate-200",
    preparing: "bg-blue-600/90 text-white",
    running: "bg-blue-600/90 text-white",
    sampling: "bg-cyan-600/90 text-white",
    finished: "bg-emerald-600/90 text-white",
    failed: "bg-red-600/90 text-white",
    stopped: "bg-amber-600/90 text-white",
    unknown: "bg-violet-600/90 text-white",
  };
  const label: Record<string, string> = {
    created: "待启动",
    preparing: "准备中",
    running: "运行中",
    sampling: "采样中",
    finished: "已完成",
    failed: "失败",
    stopped: "已停止",
    unknown: "状态未知",
  };
  const cls = map[status] || "bg-slate-700 text-slate-200";
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium " +
        cls
      }
    >
      {(status === "running" || status === "preparing" || status === "sampling") && (
        <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
      )}
      {label[status] || status}
    </span>
  );
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-12 px-4">
      <div className="w-14 h-14 rounded-full bg-slate-800/70 border border-slate-700 flex items-center justify-center mb-3">
        <svg
          className="w-6 h-6 text-slate-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0l-2 8H6l-2-8m16 0H4"
          />
        </svg>
      </div>
      <div className="text-slate-200 font-medium">{title}</div>
      {hint && <div className="text-xs text-slate-400 mt-1 max-w-md">{hint}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ProgressBar({ value, max = 100, className = "" }: { value: number; max?: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className={"w-full h-1.5 rounded-full bg-slate-800 overflow-hidden " + className}>
      <div
        className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-col items-start justify-between gap-3 sm:mb-6 sm:flex-row sm:gap-4">
      <div className="min-w-0 border-l-2 border-blue-500 pl-3">
        <h1 className="text-xl font-semibold text-slate-50">{title}</h1>
        {subtitle && <div className="text-xs text-slate-300 mt-1">{subtitle}</div>}
      </div>
      {actions && <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:shrink-0 sm:justify-end">{actions}</div>}
    </div>
  );
}

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: string; label: string; count?: number }[];
  active: string;
  onChange: (key: string) => void;
}) {
  return (
    <div className="mb-4 flex items-center gap-1 overflow-x-auto border-b border-slate-800">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className={
            "relative -mb-px shrink-0 border-b-2 px-3 py-2 text-sm font-medium transition-colors " +
            (active === t.key
              ? "border-blue-500 text-blue-300"
              : "border-transparent text-slate-400 hover:text-slate-200")
          }
        >
          {t.label}
          {typeof t.count === "number" && (
            <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
              {t.count}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
