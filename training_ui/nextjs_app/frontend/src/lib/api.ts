import { apiUrl } from "@/lib/basePath";

// 静态导出场景：由于没有 Next.js rewrites，需要让 /api/* 直接打到 FastAPI。
// 在浏览器里通过 NEXT_PUBLIC_API_BASE（build 时注入）覆盖。
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/+$/, "");

function resolve(url: string): string {
  if (API_BASE && url.startsWith("/api/")) {
    return API_BASE + url;
  }
  return apiUrl(url);
}

async function req<T = any>(url: string, init?: RequestInit): Promise<T> {
  let resolvedUrl = resolve(url);
  if (!init?.method || init.method.toUpperCase() === "GET") {
    resolvedUrl += `${resolvedUrl.includes("?") ? "&" : "?"}_ts=${Date.now()}`;
  }
  const resp = await fetch(resolvedUrl, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!resp.ok) {
    const text = await resp.text();
    let message = text;
    try {
      const payload = JSON.parse(text);
      if (typeof payload?.detail === "string") message = payload.detail;
      else if (typeof payload?.message === "string") message = payload.message;
    } catch {
      // Keep a non-JSON response body as the error message.
    }
    throw new Error(message || `${resp.status} ${resp.statusText}`);
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return (await resp.json()) as T;
  }
  return (await resp.text()) as unknown as T;
}

export const api = {
  meta: () => req<any>("/api/meta"),
  gpu: () => req<any>("/api/gpu"),
  recipes: () => req<{ recipes: any[] }>("/api/recipes"),

  listDatasets: () => req<{ datasets: any[] }>("/api/datasets"),
  createDataset: (name: string, kind: string) =>
    req<any>("/api/datasets", { method: "POST", body: JSON.stringify({ name, kind }) }),
  deleteDataset: (name: string) =>
    req<any>(`/api/datasets/${encodeURIComponent(name)}`, { method: "DELETE" }),
  datasetDetail: (name: string) => req<any>(`/api/datasets/${encodeURIComponent(name)}`),
  saveMetadata: (name: string, items: any[]) =>
    req<any>(`/api/datasets/${encodeURIComponent(name)}/metadata`, {
      method: "PUT",
      body: JSON.stringify({ items }),
    }),
  generateDatasetPrompt: (
    name: string,
    mediaPath: string,
    model: string,
    currentPrompt: string,
    instruction: string,
  ) =>
    req<{ prompt: string }>(
      `/api/datasets/${encodeURIComponent(name)}/generate_prompt`,
      {
        method: "POST",
        body: JSON.stringify({
          media_path: mediaPath,
          model,
          current_prompt: currentPrompt,
          instruction,
        }),
      },
    ),
  uploadFiles: async (name: string, files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const resp = await fetch(resolve(`/api/datasets/${encodeURIComponent(name)}/upload`), {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      const detail = (await resp.text()).trim();
      if (resp.status === 413) {
        throw new Error("文件总大小超过服务器允许的上传上限");
      }
      if (resp.status >= 500) {
        throw new Error(
          detail && detail !== "Internal Server Error"
            ? detail
            : "服务器处理上传失败，请检查上传大小限制或服务日志",
        );
      }
      throw new Error(detail || `${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  },
  deleteDatasetMedia: (name: string, files: string[]) =>
    req<{ deleted: string[] }>(`/api/datasets/${encodeURIComponent(name)}/media`, {
      method: "DELETE",
      body: JSON.stringify({ files }),
    }),

  listJobs: () => req<{ jobs: any[] }>("/api/jobs"),
  createJob: (name: string, config: any, startNow: boolean) =>
    req<any>("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ name, config, start_now: startNow }),
    }),
  updateJob: (id: string, name: string, config: any) =>
    req<any>(`/api/jobs/${id}`, {
      method: "PUT",
      body: JSON.stringify({ name, config }),
    }),
  getJob: (id: string) => req<any>(`/api/jobs/${id}`),
  startJob: (id: string) => req<any>(`/api/jobs/${id}/start`, { method: "POST" }),
  stopJob: (id: string) => req<any>(`/api/jobs/${id}/stop`, { method: "POST" }),
  deleteJob: (id: string) => req<any>(`/api/jobs/${id}`, { method: "DELETE" }),
  jobLog: (id: string) => req<string>(`/api/jobs/${id}/log`),
  jobSamples: (id: string) => req<{ samples: string[] }>(`/api/jobs/${id}/samples`),
  jobSamplingStatus: (id: string) => req<any>(`/api/jobs/${id}/sampling_status`),
  jobCheckpoints: (id: string) => req<{ checkpoints: any[] }>(`/api/jobs/${id}/checkpoints`),
  jobLoss: (id: string) => req<{ series: any[] }>(`/api/jobs/${id}/loss`),
  previewCommand: (config: any) =>
    req<any>("/api/preview_command", { method: "POST", body: JSON.stringify({ config }) }),

  getSettings: () => req<any>("/api/settings"),
  setSettings: (settings: Record<string, string>) =>
    req<any>("/api/settings", { method: "PUT", body: JSON.stringify({ settings }) }),
  clearDashscopeApiKey: () =>
    req<any>("/api/settings/dashscope_api_key", { method: "DELETE" }),
};
