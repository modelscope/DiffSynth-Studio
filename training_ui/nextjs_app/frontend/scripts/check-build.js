#!/usr/bin/env node
/**
 * 在 next start 之前校验：
 *   .next 目录必须存在；
 *   .next/DIFFSYNTH_BASE_PATH 快照与当前 NEXT_BASE_PATH 一致。
 */
const fs = require("fs");
const path = require("path");

const NEXT_DIR = path.resolve(__dirname, "..", ".next");
const SNAPSHOT = path.join(NEXT_DIR, "DIFFSYNTH_BASE_PATH");

const current = (process.env.NEXT_BASE_PATH || "").replace(/\/+$/, "");
const currentDisplay = current || "(root)";

if (!fs.existsSync(NEXT_DIR)) {
  console.error("[check-build] .next 目录不存在，请先执行 `npm run build`。");
  process.exit(1);
}

let built = "";
let hasSnapshot = false;
if (fs.existsSync(SNAPSHOT)) {
  built = fs.readFileSync(SNAPSHOT, "utf8").trim();
  hasSnapshot = true;
}
const builtDisplay = built || "(root)";

if (built !== current) {
  const runByLaunchScript = process.env.DIFFSYNTH_LAUNCHED_BY_SCRIPT === "1";
  console.error("");
  console.error("┌───────────────────────────────────────────────────────────────────┐");
  console.error("│ [check-build] build 与 start 的 NEXT_BASE_PATH 不一致，会导致 404 │");
  console.error("└───────────────────────────────────────────────────────────────────┘");
  console.error(`  build 快照: "${builtDisplay}"`);
  console.error(`  当前 env : "${currentDisplay}"`);
  if (!hasSnapshot) {
    console.error("  ⚠ 未找到 .next/DIFFSYNTH_BASE_PATH 快照（可能是旧 build 产物）");
  }
  console.error("");
  if (runByLaunchScript) {
    console.error("  你在通过 launch.sh 的 DSW 模式运行。修复：");
    console.error("    cd nextjs_app/frontend && rm -rf .next");
    console.error("    cd .. && export NEXT_BASE_PATH=<你的前缀>");
    console.error("    cd ../.. && bash launch.sh --dsw");
  } else {
    console.error("  解决办法：");
    console.error("    A) 根路径部署： unset NEXT_BASE_PATH && rm -rf .next && npm run build && npm run start");
    console.error("    B) DSW 部署：   export NEXT_BASE_PATH=<前缀> && rm -rf .next && npm run build && npm run start");
    console.error("    C) 一键脚本：   cd ../.. && bash launch.sh");
  }
  console.error("");
  process.exit(2);
}

console.log(`[check-build] OK, basePath="${currentDisplay}"`);
