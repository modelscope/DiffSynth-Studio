"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { apiUrl } from "@/lib/basePath";
import { formatDateTime, formatShellCommand } from "@/lib/format";
import {
  Button,
  Card,
  EmptyState,
  PageHeader,
  StatusBadge,
  Tabs,
} from "@/components/ui";
import { LossChart } from "@/components/LossChart";

const WEIGHT_EXTENSIONS = new Set([
  ".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".onnx", ".gguf", ".pkl", ".pickle",
]);
const TEXT_EXTENSIONS = new Set([
  ".txt", ".log", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".md",
  ".py", ".sh", ".toml", ".ini", ".cfg", ".xml", ".html", ".htm",
]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".webm", ".mov", ".mkv", ".avi"]);
const AUDIO_EXTENSIONS = new Set([".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]);

function fileExtension(path: string) {
  const name = path.toLowerCase();
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index) : "";
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  const kilobytes = bytes / 1024;
  if (kilobytes < 1024) return `${kilobytes.toFixed(1)} KB`;
  const megabytes = kilobytes / 1024;
  if (megabytes < 1024) return `${megabytes.toFixed(1)} MB`;
  return `${(megabytes / 1024).toFixed(2)} GB`;
}

export default function JobDetailPage() {
  const params = useParams<{ id: string }>();
  const jobId = params.id;

  const [tab, setTab] = useState<"overview" | "log" | "files" | "config">("overview");
  const [job, setJob] = useState<any>(null);
  const [log, setLog] = useState<string>("");
  const [samples, setSamples] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [files, setFiles] = useState<any[]>([]);
  const [loss, setLoss] = useState<any[]>([]);
  const [samplingStatus, setSamplingStatus] = useState<any>({ status: "not_started" });
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [msg, setMsg] = useState("");
  const [previewFile, setPreviewFile] = useState<any>(null);
  const [previewText, setPreviewText] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  async function openFilePreview(file: any) {
    setPreviewFile(file);
    setPreviewText("");
    setPreviewError("");
    const extension = fileExtension(file.rel_path);
    if (!TEXT_EXTENSIONS.has(extension)) return;
    setPreviewLoading(true);
    try {
      const response = await fetch(
        apiUrl(`/api/jobs/${jobId}/artifact?path=${encodeURIComponent(file.rel_path)}`),
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      setPreviewText(await response.text());
    } catch (error: any) {
      setPreviewError(error.message || "预览加载失败");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function reloadCore() {
    try {
      const j = await api.getJob(jobId);
      setJob(j);
      const l = await api.jobLog(jobId);
      setLog(l);
      requestAnimationFrame(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
      });
    } catch (e: any) {
      setMsg(e.message);
    }
  }

  async function reloadArtifacts() {
    const [s, sampling, c, f, ls] = await Promise.all([
      api.jobSamples(jobId).catch(() => null),
      api.jobSamplingStatus(jobId).catch(() => null),
      api.jobCheckpoints(jobId).catch(() => null),
      (async () => {
        try {
          const separator = apiUrl(`/api/jobs/${jobId}/files`).includes("?") ? "&" : "?";
          const r = await fetch(
            `${apiUrl(`/api/jobs/${jobId}/files`)}${separator}_ts=${Date.now()}`,
            { cache: "no-store" },
          );
          if (!r.ok) return null;
          return r.json();
        } catch {
          return null;
        }
      })(),
      api.jobLoss(jobId).catch(() => null),
    ]);
    if (s) setSamples(s.samples || []);
    if (sampling) setSamplingStatus(sampling);
    if (c) setCheckpoints(c.checkpoints || []);
    if (f) setFiles((f as any).files || []);
    if (ls) setLoss(ls.series || []);
  }

  useEffect(() => {
    reloadCore();
    reloadArtifacts();
  }, [jobId]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = setInterval(() => {
      reloadCore();
      reloadArtifacts();
    }, 2500);
    return () => clearInterval(timer);
  }, [autoRefresh, jobId]);

  if (!job) {
    return (
      <div className="p-3 text-slate-400 sm:p-4 lg:p-6">
        加载中... {msg && <div className="text-red-400 mt-2">{msg}</div>}
      </div>
    );
  }

  const cmd = (job.command || []) as string[];
  const runConfig = job.latest_run?.config || job.config;
  const shellCmd = formatShellCommand(cmd);
  const latestLoss = loss.length > 0 ? loss[loss.length - 1] : null;

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-3 sm:p-4 lg:p-6">
      <PageHeader
        title={job.name}
        subtitle={
          <>
            <span className="mr-2">
              <StatusBadge status={job.status} />
            </span>
          </>
        }
        actions={
          <>
            <label className="flex items-center gap-1 text-xs text-slate-400 mr-2">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              自动刷新
            </label>
            {!autoRefresh && (
              <Button
                variant="outline"
                onClick={() => {
                  reloadCore();
                  reloadArtifacts();
                }}
              >
                刷新
              </Button>
            )}
            {!["running", "preparing", "sampling"].includes(job.status) && (
              <>
                <Link href={`/jobs/new?edit=${encodeURIComponent(job.id)}`}>
                  <Button variant="outline">编辑</Button>
                </Link>
                <Button
                  onClick={async () => {
                    await api.startJob(job.id);
                    reloadCore();
                  }}
                >
                  启动
                </Button>
              </>
            )}
            {["running", "preparing", "sampling"].includes(job.status) && (
              <Button
                variant="danger"
                onClick={async () => {
                  await api.stopJob(job.id);
                  reloadCore();
                }}
              >
                停止
              </Button>
            )}
            <Link href="/jobs">
              <Button variant="ghost" size="sm">
                ← 返回列表
              </Button>
            </Link>
          </>
        }
      />

      <Tabs
        tabs={[
          { key: "overview", label: "总览" },
          { key: "log", label: "日志" },
          { key: "files", label: "文件", count: files.length },
          { key: "config", label: "配置 & 命令" },
        ]}
        active={tab}
        onChange={(k) => setTab(k as any)}
      />

      {job.status === "unknown" && (
        <div className="mb-4 rounded border border-violet-500/40 bg-violet-500/10 px-4 py-3 text-sm text-violet-200">
          训练进程已经退出，但后端无法恢复退出码，因此不能确认成功或失败。请检查完整日志和产物后再决定是否重新启动。
        </div>
      )}

      {tab === "overview" && (
        <div className="space-y-4">
          <div className="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
            <Card
              title="Loss 曲线"
              subtitle={
                latestLoss
                  ? `最新 step ${latestLoss.step} · loss ${Number(latestLoss.loss).toFixed(6)}`
                  : undefined
              }
            >
              <LossChart data={loss.map((p: any) => ({ step: p.step, loss: p.loss }))} />
            </Card>
            <Card title="产物概览">
              <MetricRow label="Checkpoints" value={checkpoints.length} />
              <MetricRow label="最终样本" value={samples.length} />
              <MetricRow label="其他文件" value={files.length} />
              <div className="mt-3 text-[11px] text-slate-400 space-y-1">
                <div>
                  <span className="text-slate-400">output_path:</span>{" "}
                  <code className="text-slate-300">{job.output_path || "-"}</code>
                </div>
                <div>
                  <span className="text-slate-400">log_path:</span>{" "}
                  <code className="text-slate-300">{job.log_path || "-"}</code>
                </div>
                <div>
                  <span className="text-slate-400">运行 GPU:</span>{" "}
                  <code className="text-slate-300">GPU {runConfig?.gpu_index ?? 0}</code>
                </div>
                <div className="text-slate-400">
                  创建 {formatDateTime(job.created_at)} · 启动 {formatDateTime(job.started_at)} · 结束{" "}
                  {formatDateTime(job.finished_at)}
                </div>
              </div>
            </Card>
          </div>

          <Card
            title="Checkpoints"
            actions={
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  api.jobCheckpoints(jobId).then((r) => setCheckpoints(r.checkpoints || []))
                }
              >
                刷新
              </Button>
            }
            padded={false}
          >
            {checkpoints.length === 0 ? (
              <div className="py-6">
                <EmptyState title="尚无 checkpoint"/>
              </div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>文件</th>
                    <th className="w-32">大小</th>
                    <th className="w-40">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.map((c) => (
                    <tr key={c.rel_path}>
                      <td className="text-slate-200 mono text-xs">{c.rel_path}</td>
                      <td className="text-slate-400 mono text-xs">
                        {formatFileSize(c.size)}
                      </td>
                      <td>
                        <a
                          href={apiUrl(
                            `/api/jobs/${jobId}/artifact?path=${encodeURIComponent(c.rel_path)}&download=true`,
                          )}
                          download={c.rel_path.split("/").pop()}
                        >
                          <Button variant="outline" size="sm">
                            下载
                          </Button>
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          <Card
            title={`最终采样`}
            subtitle={
              samplingStatus.checkpoint
                ? `使用 ${String(samplingStatus.checkpoint).split("/").pop()}`
                : "训练完成后使用最后一个 .safetensors 和测试 prompts 采样"
            }
            actions={<SamplingState status={samplingStatus.status} />}
          >
            {samplingStatus.status === "failed" && (
              <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                {samplingStatus.message || "最终采样失败，请查看 final_samples 下的日志"}
              </div>
            )}
            {samples.length === 0 ? (
              <EmptyState
                title={
                  samplingStatus.status === "running" || samplingStatus.status === "queued"
                    ? "正在生成最终样本"
                    : "尚无最终样本"
                }
                hint={
                  samplingStatus.status === "running" || samplingStatus.status === "queued"
                    ? samplingStatus.current
                      ? `正在生成第 ${samplingStatus.current} / ${samplingStatus.total || 0} 个 prompt`
                      : `采样任务已排队，共 ${samplingStatus.total || 0} 个 prompt`
                    : undefined
                }
              />
            ) : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
                {samples.map((sample) => (
                  <SamplePreview key={sample.rel_path} jobId={jobId} sample={sample} />
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {tab === "log" && (
        <Card title="训练日志">
          <pre
            ref={logRef}
            className="text-xs whitespace-pre-wrap mono bg-black/60 rounded-lg p-3 max-h-[560px] overflow-y-auto text-slate-200"
          >
            {log || "尚无日志"}
          </pre>
        </Card>
      )}

      {tab === "files" && (
        <Card title={`产物文件 (${files.length})`} padded={false}>
          {files.length === 0 ? (
            <div className="py-6">
              <EmptyState title="尚无产物"/>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>路径</th>
                  <th className="w-32">大小</th>
                  <th className="w-40">操作</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => {
                  const weightFile = WEIGHT_EXTENSIONS.has(fileExtension(f.rel_path));
                  return (
                    <tr key={f.rel_path}>
                      <td className="text-slate-200 mono text-xs">{f.rel_path}</td>
                      <td className="text-slate-400 mono text-xs">
                        {formatFileSize(f.size)}
                      </td>
                      <td>
                        <div className="flex items-center gap-2">
                          {!weightFile && (
                            <Button variant="outline" size="sm" onClick={() => openFilePreview(f)}>
                              预览
                            </Button>
                          )}
                          <a
                            href={apiUrl(
                              `/api/jobs/${jobId}/artifact?path=${encodeURIComponent(f.rel_path)}&download=true`,
                            )}
                          >
                            <Button variant="outline" size="sm">
                              下载
                            </Button>
                          </a>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Card>
      )}

      {tab === "config" && (
        <div className="space-y-4">
          <Card title="启动命令">
            <pre className="text-xs whitespace-pre-wrap break-words mono text-slate-300 overflow-x-auto">
              {shellCmd || "(尚未启动)"}
            </pre>
          </Card>
          <Card title="任务配置 (JSON)">
            <pre className="text-xs whitespace-pre-wrap mono text-slate-300 max-h-[400px] overflow-y-auto">
              {JSON.stringify(runConfig, null, 2)}
            </pre>
          </Card>
        </div>
      )}

      {previewFile && (
        <ArtifactPreviewModal
          jobId={jobId}
          file={previewFile}
          text={previewText}
          loading={previewLoading}
          error={previewError}
          onClose={() => setPreviewFile(null)}
        />
      )}
    </div>
  );
}

function ArtifactPreviewModal({
  jobId,
  file,
  text,
  loading,
  error,
  onClose,
}: {
  jobId: string;
  file: any;
  text: string;
  loading: boolean;
  error: string;
  onClose: () => void;
}) {
  const extension = fileExtension(file.rel_path);
  const src = apiUrl(`/api/jobs/${jobId}/artifact?path=${encodeURIComponent(file.rel_path)}`);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-6" onClick={onClose}>
      <div
        className="flex h-[82vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-950 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-4 border-b border-slate-800 px-4 py-3">
          <div className="min-w-0 truncate text-sm font-medium text-slate-100">{file.rel_path}</div>
          <Button variant="ghost" size="sm" onClick={onClose}>关闭</Button>
        </div>
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-black/35 p-4">
          {loading ? (
            <div className="text-sm text-slate-400">正在加载预览...</div>
          ) : error ? (
            <div className="text-sm text-red-400">{error}</div>
          ) : TEXT_EXTENSIONS.has(extension) ? (
            <pre className="h-full w-full overflow-auto whitespace-pre-wrap break-words rounded bg-black/50 p-4 text-xs text-slate-200 mono">
              {text}
            </pre>
          ) : IMAGE_EXTENSIONS.has(extension) ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={src} alt={file.rel_path} className="max-h-full max-w-full object-contain" />
            </>
          ) : VIDEO_EXTENSIONS.has(extension) ? (
            <video src={src} className="max-h-full max-w-full" controls preload="metadata" />
          ) : AUDIO_EXTENSIONS.has(extension) ? (
            <audio src={src} className="w-full max-w-2xl" controls preload="metadata" />
          ) : extension === ".pdf" ? (
            <iframe src={src} title={file.rel_path} className="h-full w-full bg-white" />
          ) : (
            <iframe src={src} title={file.rel_path} className="h-full w-full bg-white" sandbox="" />
          )}
        </div>
      </div>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-b-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-sm text-slate-100 mono">{value}</span>
    </div>
  );
}

function SamplingState({ status }: { status: string }) {
  const labels: Record<string, string> = {
    not_started: "等待训练完成",
    queued: "等待采样",
    running: "采样中",
    finished: "采样完成",
    failed: "采样失败",
    skipped: "已跳过",
    stopped: "已停止",
  };
  const classes: Record<string, string> = {
    queued: "border-blue-500/30 bg-blue-500/10 text-blue-300",
    running: "border-blue-500/30 bg-blue-500/10 text-blue-300",
    finished: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    failed: "border-red-500/30 bg-red-500/10 text-red-300",
    skipped: "border-slate-700 bg-slate-800 text-slate-300",
    stopped: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    not_started: "border-slate-700 bg-slate-800 text-slate-300",
  };
  return (
    <span className={`rounded-md border px-2 py-1 text-[11px] ${classes[status] || classes.not_started}`}>
      {labels[status] || status}
    </span>
  );
}

function SamplePreview({ jobId, sample }: { jobId: string; sample: any }) {
  const src = apiUrl(
    `/api/jobs/${jobId}/artifact?path=${encodeURIComponent(sample.rel_path)}`,
  );
  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/40">
      {sample.kind === "video" ? (
        <video className="aspect-video w-full bg-black object-contain" src={src} controls preload="metadata" />
      ) : sample.kind === "audio" ? (
        <div className="flex min-h-28 items-center bg-slate-950/70 px-3">
          <audio className="w-full" src={src} controls preload="metadata" />
        </div>
      ) : (
        <a href={src} target="_blank" rel="noreferrer">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={sample.prompt || sample.name} className="aspect-[1/1] w-full object-cover" />
        </a>
      )}
      <div className="space-y-1 border-t border-slate-800 px-2.5 py-2">
        {sample.prompt && <div className="line-clamp-2 text-xs text-slate-400">{sample.prompt}</div>}
      </div>
    </div>
  );
}
