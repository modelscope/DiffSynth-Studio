"use strict";
const http = require("http");

const BASE_PATH = (process.env.BASE_PATH || "").replace(/\/+$/, "");
const LISTEN_PORT = Number(process.env.LISTEN_PORT || 8100);
const UPSTREAM_PORT = Number(process.env.UPSTREAM_PORT || 8101);
const UPSTREAM_HOST = process.env.UPSTREAM_HOST || "127.0.0.1";
const QUIET = process.env.QUIET === "1";

if (!BASE_PATH) {
  console.error("[dsw-adapter] 未设置 BASE_PATH。");
  process.exit(1);
}

function dashboardLocation() {
  return BASE_PATH + "/dashboard";
}

function routeRequest(originalUrl) {
  if (originalUrl === BASE_PATH || originalUrl.startsWith(BASE_PATH + "/") ||
      originalUrl.startsWith(BASE_PATH + "?")) {
    if (originalUrl === BASE_PATH || originalUrl === BASE_PATH + "/" || originalUrl.startsWith(BASE_PATH + "?")) {
      return { redirect: dashboardLocation() };
    }
    return { targetPath: originalUrl };
  }
  if (originalUrl === "/" || originalUrl === "") {
    return { redirect: dashboardLocation() };
  }
  return { targetPath: BASE_PATH + (originalUrl.startsWith("/") ? originalUrl : "/" + originalUrl) };
}

const server = http.createServer((req, res) => {
  const routed = routeRequest(req.url || "/");
  if (routed.redirect) {
    res.writeHead(302, { Location: routed.redirect });
    res.end();
    return;
  }
  const targetPath = routed.targetPath;
  const options = {
    hostname: UPSTREAM_HOST,
    port: UPSTREAM_PORT,
    path: targetPath,
    method: req.method,
    headers: { ...req.headers, host: `${UPSTREAM_HOST}:${UPSTREAM_PORT}` },
  };
  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
    proxyRes.pipe(res);
  });
  proxyReq.on("error", (err) => {
    console.error("[dsw-adapter] upstream error:", err.message, "path:", targetPath);
    res.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(`upstream error: ${err.message}`);
  });
  req.pipe(proxyReq);
});

server.listen(LISTEN_PORT, "0.0.0.0", () => {
  if (!QUIET) {
    console.log(`[dsw-adapter] listening 0.0.0.0:${LISTEN_PORT}`);
    console.log(`[dsw-adapter] BASE_PATH="${BASE_PATH}"`);
    console.log(`[dsw-adapter] upstream = http://${UPSTREAM_HOST}:${UPSTREAM_PORT}`);
  }
});
