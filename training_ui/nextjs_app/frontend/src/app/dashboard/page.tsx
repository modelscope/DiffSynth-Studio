"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { GpuMonitor } from "@/components/GpuMonitor";
import { Button, Card, EmptyState, PageHeader, StatusBadge } from "@/components/ui";
import { api } from "@/lib/api";
import { formatDateTime } from "@/lib/format";

const ACTIVE_STATUSES = new Set(["preparing", "running", "sampling"]);

export default function DashboardPage() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [error, setError] = useState("");

  async function reload() {
    try {
      const [jobResult, datasetResult] = await Promise.all([
        api.listJobs(),
        api.listDatasets(),
      ]);
      setJobs(jobResult.jobs || []);
      setDatasets(datasetResult.datasets || []);
      setError("");
    } catch (err: any) {
      setError(`总览加载失败：${err?.message || "无法连接后端"}`);
    }
  }

  useEffect(() => {
    reload();
    const timer = setInterval(reload, 3000);
    return () => clearInterval(timer);
  }, []);

  const summary = useMemo(() => {
    const active = jobs.filter((job) => ACTIVE_STATUSES.has(job.status)).length;
    const waiting = jobs.filter((job) => job.status === "created").length;
    const finished = jobs.filter((job) => job.status === "finished").length;
    return { active, waiting, finished };
  }, [jobs]);

  const recentJobs = jobs.slice(0, 10);

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-6">
      <PageHeader
        title="总览"
        actions={
          <Link href="/jobs/new">
            <Button>+ 新建任务</Button>
          </Link>
        }
      />

      {error && (
        <div className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      <div className="mb-4 grid grid-cols-4 gap-4">
        <Metric label="进行中" value={summary.active} tone="blue" href="/jobs" />
        <Metric label="待启动" value={summary.waiting} tone="amber" href="/jobs" />
        <Metric label="已完成" value={summary.finished} tone="emerald" href="/jobs" />
        <Metric label="数据集" value={datasets.length} tone="cyan" href="/datasets" />
      </div>

      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_400px] items-stretch gap-4">
        <Card
          title="最近任务"
          className="h-full"
          actions={
            <Link href="/jobs">
              <Button variant="ghost" size="sm">查看全部 →</Button>
            </Link>
          }
          padded={false}
        >
          {recentJobs.length === 0 ? (
            <EmptyState
              title="尚未创建训练任务"
              action={
                <Link href="/jobs/new">
                  <Button>+ 新建任务</Button>
                </Link>
              }
            />
          ) : (
            <table className="min-w-[750px] leading-5">
              <thead>
                <tr>
                  <th>任务名称</th>
                  <th>训练模型</th>
                  <th>数据集</th>
                  <th>状态</th>
                  <th>创建时间</th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.map((job) => (
                  <tr key={job.id} className="h-12">
                    <td>
                      <Link href={`/jobs/${job.id}`} className="font-medium text-blue-300 hover:underline">
                        {job.name}
                      </Link>
                    </td>
                    <td className="text-slate-200">{job.model_type}</td>
                    <td className="text-slate-300">{job.dataset}</td>
                    <td><StatusBadge status={job.status} /></td>
                    <td className="mono text-xs text-slate-300">{formatDateTime(job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <GpuMonitor />
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
  href,
}: {
  label: string;
  value: number;
  tone: "blue" | "amber" | "emerald" | "cyan";
  href: string;
}) {
  const tones = {
    blue: "border-t-blue-400 text-blue-300",
    amber: "border-t-amber-400 text-amber-300",
    emerald: "border-t-emerald-400 text-emerald-300",
    cyan: "border-t-cyan-400 text-cyan-300",
  }[tone];
  return (
    <Link
      href={href}
      className={`block rounded-lg border border-slate-800 border-t-2 bg-slate-900/95 px-4 py-3 shadow-[0_8px_24px_rgba(0,0,0,0.14)] transition-colors hover:border-slate-600 ${tones}`}
    >
      <div className="text-xs font-semibold">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-50">{value}</div>
    </Link>
  );
}
