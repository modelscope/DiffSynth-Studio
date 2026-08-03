"use client";

import { useEffect } from "react";
import Link from "next/link";
import { BASE_PATH, withBasePath } from "@/lib/basePath";

/**
 * 全站 404 兜底页（纯根路径部署下，只有真正的错误路径才会命中）。
 */
export default function NotFound() {
  useEffect(() => {
    // DSW 里误点内部 Next 端口（如 8101）时，页面会落到 404；
    // 这里直接带回对外适配器端口对应的 dashboard（如 8100/dashboard）。
    const pathname = window.location.pathname.replace(/\/+$/, "") || "/";
    const enteredThroughWrongDswPort =
      !!BASE_PATH && pathname.includes("/ide/proxy/") && !pathname.startsWith(BASE_PATH);
    if (pathname === "/" || enteredThroughWrongDswPort) {
      window.location.replace(withBasePath("/dashboard"));
    }
  }, []);

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="rounded-xl border border-slate-800 bg-gradient-to-b from-slate-900/60 to-slate-950/60 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
            <span className="text-amber-300 text-lg font-semibold">!</span>
          </div>
          <div>
            <div className="text-lg font-semibold text-slate-100">404 · 页面不存在</div>
            <div className="text-xs text-slate-400 mt-0.5">
              可能来自过时链接或误输入
            </div>
          </div>
        </div>

        <div className="mt-4">
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-1 rounded-md bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 text-sm"
          >
            ← 回到总览
          </Link>
        </div>
      </div>
    </div>
  );
}
