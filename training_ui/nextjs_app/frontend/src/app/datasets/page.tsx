"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Button, Card, EmptyState, Field, PageHeader } from "@/components/ui";

const KINDS = ["image", "edit", "video", "audio"];

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [newName, setNewName] = useState("");
  const [newKind, setNewKind] = useState("image");
  const [msg, setMsg] = useState("");

  async function reload() {
    try {
      const r = await api.listDatasets();
      setDatasets(r.datasets || []);
    } catch (e: any) {
      setMsg("加载失败: " + e.message);
    }
  }
  useEffect(() => {
    reload();
  }, []);

  async function onCreate() {
    if (!newName.trim()) {
      setMsg("请输入数据集名");
      return;
    }
    try {
      await api.createDataset(newName.trim(), newKind);
      setMsg("");
      setNewName("");
      reload();
    } catch (e: any) {
      const msg = String(e?.message || "");
      if (msg.includes("数据集已存在")) {
        setMsg("数据集已存在，请更换一个名称");
      } else {
        setMsg("创建失败，请稍后重试");
      }
    }
  }

  async function onDelete(name: string) {
    if (!confirm(`确认删除数据集 [${name}] ?`)) return;
    try {
      await api.deleteDataset(name);
      setMsg("");
      reload();
    } catch (e: any) {
      setMsg("删除失败: " + e.message);
    }
  }

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-3 sm:p-4 lg:p-6">
      <PageHeader title="数据集" />

      <div className="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
        <Card title="新建数据集">
          <Field label="数据集名" required>
            <input
              className="w-full"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
          </Field>
          <Field label="数据集类型">
            <select className="w-full" value={newKind} onChange={(e) => setNewKind(e.target.value)}>
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </Field>
          <Button onClick={onCreate}>+ 创建数据集</Button>
          {msg && <div className="text-xs text-slate-400 mt-3">{msg}</div>}
        </Card>

        <Card title="数据集列表" padded={false}>
          {datasets.length === 0 ? (
            <div className="py-6">
              <EmptyState
                title="还没有数据集"
                hint="在左侧输入名称，选择类型，创建你的第一个数据集"
              />
            </div>
          ) : (
            <table className="min-w-[620px]">
              <thead>
                <tr>
                  <th>名称</th>
                  <th>类型</th>
                  <th>样本数</th>
                  <th className="w-40">操作</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.name}>
                    <td>
                      <Link
                        href={`/datasets/${encodeURIComponent(d.name)}`}
                        className="text-blue-300 hover:underline"
                      >
                        {d.name}
                      </Link>
                    </td>
                    <td className="text-slate-300">{d.kind}</td>
                    <td className="text-slate-300 mono">{d.num_items}</td>
                    <td>
                      <div className="flex gap-2">
                        <Link href={`/datasets/${encodeURIComponent(d.name)}`}>
                          <Button variant="outline" size="sm">
                            打开
                          </Button>
                        </Link>
                        <Button variant="danger" size="sm" onClick={() => onDelete(d.name)}>
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
    </div>
  );
}
