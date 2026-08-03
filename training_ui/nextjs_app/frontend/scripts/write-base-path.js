#!/usr/bin/env node
/**
 * 在 next build 结束后把 NEXT_BASE_PATH 写入 .next/DIFFSYNTH_BASE_PATH，
 * 用于 start 前校验一致性（check-build.js 读它）。
 */
const fs = require("fs");
const path = require("path");

const NEXT_DIR = path.resolve(__dirname, "..", ".next");
const SNAPSHOT = path.join(NEXT_DIR, "DIFFSYNTH_BASE_PATH");

const base = (process.env.NEXT_BASE_PATH || "").replace(/\/+$/, "");

if (!fs.existsSync(NEXT_DIR)) {
  console.warn("[write-base-path] .next 目录不存在，跳过。");
  process.exit(0);
}
fs.writeFileSync(SNAPSHOT, base, "utf8");
console.log(`[write-base-path] snapshot NEXT_BASE_PATH="${base || "(root)"}"`);
