/**
 * basePath 助手。build 时通过 next.config.ts 的 env 注入 NEXT_PUBLIC_BASE_PATH。
 *
 * 无 DSW / 根路径部署时，BASE_PATH === ""，所有 helper 等价于原样返回。
 */
export const BASE_PATH: string = (process.env.NEXT_PUBLIC_BASE_PATH || "").replace(/\/+$/, "");

export function withBasePath(path: string): string {
  if (!path) return BASE_PATH || "/";
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PATH}${normalized}`;
}

export function apiUrl(path: string): string {
  return withBasePath(path);
}
