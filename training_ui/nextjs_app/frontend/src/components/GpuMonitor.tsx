"use client";

import { useEffect, useState } from "react";
import { Card, ProgressBar } from "@/components/ui";
import { api } from "@/lib/api";

export function GpuMonitor() {
  const [gpus, setGpus] = useState<any[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    async function reload() {
      try {
        const result = await api.gpu();
        if (alive) {
          setGpus(result.gpus || []);
          setError("");
        }
      } catch (err: any) {
        if (alive) setError(err?.message || "无法读取 GPU 状态");
      }
    }
    reload();
    const timer = setInterval(reload, 2500);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <Card title="GPU 资源" className="h-full">
      {gpus.length === 0 ? (
        <div className="py-6 text-center text-sm text-slate-400">未检测到 GPU</div>
      ) : (
        <div className="space-y-3">
          {gpus.map((gpu: any) => {
            const memoryPercent = gpu.memory_total_mb
              ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100)
              : 0;
            return (
              <div key={gpu.index} className="rounded-md border border-slate-800 bg-slate-950/40 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <div>
                    <span className="text-xs text-slate-400">GPU #{gpu.index}</span>
                    <span className="ml-2 text-sm font-medium text-slate-100">{gpu.name}</span>
                  </div>
                  <div className="text-xs text-slate-400">
                    <span className="mr-3">利用率 {gpu.utilization}%</span>
                    <span>{gpu.temperature}°C</span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1"><ProgressBar value={memoryPercent} /></div>
                  <div className="mono text-[11px] text-slate-400">
                    {Math.round(gpu.memory_used_mb / 1024)}G / {Math.round(gpu.memory_total_mb / 1024)}G
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
      {error && (
        <div className="mt-3 rounded-md border border-red-500/25 bg-red-500/5 px-3 py-2 text-xs text-red-300">
          GPU 状态暂不可用：{error}
        </div>
      )}
    </Card>
  );
}
