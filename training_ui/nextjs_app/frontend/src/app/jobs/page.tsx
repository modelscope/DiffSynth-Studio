"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { Button, Card, EmptyState, PageHeader, StatusBadge, Tabs } from "@/components/ui";

export default function JobsPage() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [filter, setFilter] = useState<"all" | "running" | "history">("all");
  const [msg, setMsg] = useState("");

  async function reload() {
    try {
      const r = await api.listJobs();
      setJobs(r.jobs || []);
    } catch (e: any) {
      setMsg("加载失败: " + e.message);
    }
  }

  useEffect(() => {
    reload();
    const timer = setInterval(reload, 3000);
    return () => clearInterval(timer);
  }, []);

  const counts = {
    all: jobs.length,
    running: jobs.filter((j) => ["running", "preparing", "sampling"].includes(j.status)).length,
    history: jobs.filter((j) => !["running", "preparing", "sampling"].includes(j.status)).length,
  };
  const filtered = jobs.filter((j) => {
    if (filter === "running") return ["running", "preparing", "sampling"].includes(j.status);
    if (filter === "history") return !["running", "preparing", "sampling"].includes(j.status);
    return true;
  });

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-3 sm:p-4 lg:p-6">
      <PageHeader
        title="任务管理"
        actions={
          <>
            <Link href="/jobs/new">
              <Button>+ 新建任务</Button>
            </Link>
          </>
        }
      />

      <Tabs
        tabs={[
          { key: "all", label: "全部", count: counts.all },
          { key: "running", label: "进行中", count: counts.running },
          { key: "history", label: "历史", count: counts.history },
        ]}
        active={filter}
        onChange={(k) => setFilter(k as any)}
      />

      {msg && <div className="text-xs text-red-400 mb-2">{msg}</div>}

      <Card padded={false}>
        {filtered.length === 0 ? (
          <div className="py-8">
            <EmptyState
              title="暂无任务"
              hint="点右上角『新建任务』开始"
              action={
                <Link href="/jobs/new">
                  <Button>+ 立即新建</Button>
                </Link>
              }
            />
          </div>
        ) : (
          <table className="min-w-[860px]">
            <thead>
              <tr>
                <th>名称</th>
                <th>模型</th>
                <th>数据集</th>
                <th>GPU</th>
                <th>状态</th>
                <th>创建时间</th>
                <th className="w-52">操作</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((j) => (
                <tr key={j.id}>
                  <td>
                    <Link href={`/jobs/${j.id}`} className="text-blue-300 hover:underline font-medium">
                      {j.name}
                    </Link>
                  </td>
                  <td className="text-slate-300">{j.model_type}</td>
                  <td className="text-slate-400">{j.dataset}</td>
                  <td className="text-slate-300 mono">{j.config?.gpu_index ?? 0}</td>
                  <td>
                    <StatusBadge status={j.status} />
                  </td>
                  <td className="text-slate-400 mono text-xs">
                    {formatDateTime(j.created_at)}
                  </td>
                  <td>
                    <div className="flex gap-2">
                      {!["running", "preparing", "sampling"].includes(j.status) && (
                        <>
                          <Link href={`/jobs/new?edit=${encodeURIComponent(j.id)}`}>
                            <Button variant="outline" size="sm">编辑</Button>
                          </Link>
                          <Button
                            size="sm"
                            onClick={async () => {
                              await api.startJob(j.id);
                              reload();
                            }}
                          >
                            启动
                          </Button>
                        </>
                      )}
                      {["running", "preparing", "sampling"].includes(j.status) && (
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={async () => {
                            await api.stopJob(j.id);
                            reload();
                          }}
                        >
                          停止
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={async () => {
                          if (!confirm(`删除任务“${j.name}”及其所有历史进程和输出？`)) return;
                          await api.deleteJob(j.id);
                          reload();
                        }}
                      >
                        删除
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
