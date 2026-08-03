"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { withBasePath } from "@/lib/basePath";

/**
 * 根路径客户端跳转到 /dashboard。
 * - router.replace 会自动带 basePath；
 * - fallback <a> 需要手动加 basePath，否则 DSW 场景下会跳到域名根路径 404。
 */
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/dashboard");
  }, [router]);
  return (
    <div className="p-6 text-slate-400 text-sm">
      正在跳转到{" "}
      <a className="text-blue-300 underline" href={withBasePath("/dashboard")}>
        总览
      </a>
      …
    </div>
  );
}
