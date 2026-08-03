import type { NextConfig } from "next";
import path from "path";

/**
 * 后端反代地址：默认打到本机 FastAPI (8000)。
 *   export DIFFSYNTH_UI_BACKEND=http://127.0.0.1:8000
 */
const backend = process.env.DIFFSYNTH_UI_BACKEND || "http://127.0.0.1:8000";

/**
 * ⚠ DSW 网关行为（经实测确认）：
 *   浏览器打开 https://dsw-gateway.../dsw-311560/ide/proxy/8100/foo
 *   DSW **不剥前缀**，直接把 `/dsw-311560/ide/proxy/8100/foo` 送到本机 8100。
 *
 * 因此 Next.js 必须以 basePath=/dsw-311560/ide/proxy/8100 build，
 * 产物里资源路径写成 `/dsw-311560/ide/proxy/8100/_next/...`，
 * 与浏览器和 DSW 转发的路径完全一致。
 *
 * 用法：
 *   export NEXT_BASE_PATH=/dsw-311560/ide/proxy/8100
 *   npm run build && npm run start
 * 无 DSW / 本地开发场景：不设 NEXT_BASE_PATH，走根路径部署。
 */
const rawBase = (process.env.NEXT_BASE_PATH || "").trim();
const basePath = rawBase.replace(/\/+$/, "");

// 兜底：static export，产纯静态站，用文件服务器分发
const staticExport = process.env.STATIC_EXPORT === "1";
const configuredUploadLimit = Number.parseInt(
  process.env.DIFFSYNTH_UI_UPLOAD_LIMIT_BYTES || "",
  10,
);
const uploadBodySize =
  Number.isFinite(configuredUploadLimit) && configuredUploadLimit > 0
    ? configuredUploadLimit
    : 256 * 1024 * 1024;

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // Next.js rewrites clone request bodies and otherwise truncate them at 10MB.
  // Dataset uploads commonly contain audio, video, or archives larger than that.
  experimental: {
    middlewareClientMaxBodySize: uploadBodySize,
  },

  // 钉死 tracing root 到当前 frontend/ 目录，避免上层有 package-lock.json 时
  // Next.js 15 把 _next/static/* 物理路径推断到别处。
  outputFileTracingRoot: path.resolve(__dirname),

  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },

  ...(staticExport ? { output: "export" as const, trailingSlash: true } : {}),

  async rewrites() {
    if (staticExport) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
