"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Button, Card, Field, PageHeader } from "@/components/ui";

const ROWS: Array<{ key: string; label: string; hint?: string }> = [
  {
    key: "DATASETS_ROOT",
    label: "数据集保存路径",
    hint: "默认使用 training_ui/data/datasets",
  },
  {
    key: "MODEL_SAVE_ROOT",
    label: "训练输出路径",
    hint: "默认使用 training_ui/data/outputs",
  },
  {
    key: "DIFFSYNTH_MODEL_BASE_PATH",
    label: "模型下载路径 (DIFFSYNTH_MODEL_BASE_PATH)",
    hint: "本地保存下载模型的根目录",
  },
  {
    key: "DIFFSYNTH_DOWNLOAD_SOURCE",
    label: "模型下载源 (DIFFSYNTH_DOWNLOAD_SOURCE)",
    hint: "modelscope 或 huggingface",
  },
  {
    key: "DIFFSYNTH_ATTENTION_IMPLEMENTATION",
    label: "Attention 实现 (DIFFSYNTH_ATTENTION_IMPLEMENTATION)",
  },
];

export default function SettingsPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<any>({});
  const [msg, setMsg] = useState("");
  const [secretConfigured, setSecretConfigured] = useState<Record<string, boolean>>({});

  async function reload() {
    const [settingsResult, metaResult] = await Promise.allSettled([
      api.getSettings(),
      api.meta(),
    ]);
    const errors: string[] = [];
    if (settingsResult.status === "fulfilled") {
      setValues(settingsResult.value.settings || {});
      setSecretConfigured(settingsResult.value.secret_configured || {});
    } else {
      errors.push(`设置加载失败: ${settingsResult.reason?.message || settingsResult.reason}`);
    }
    if (metaResult.status === "fulfilled") {
      setMeta(metaResult.value || {});
    } else {
      errors.push(`路径信息加载失败: ${metaResult.reason?.message || metaResult.reason}`);
    }
    setMsg(errors.join("；"));
  }
  useEffect(() => {
    reload();
  }, []);

  async function onSave() {
    try {
      await api.setSettings(values);
      await reload();
      setMsg("已保存");
    } catch (e: any) {
      setMsg("保存失败: " + e.message);
    }
  }

  async function onClearDashscopeApiKey() {
    if (!confirm("确认清除已保存的API Key？")) return;
    try {
      await api.clearDashscopeApiKey();
      setValues((current) => ({ ...current, DASHSCOPE_API_KEY: "" }));
      await reload();
      setMsg("API Key 已清除");
    } catch (e: any) {
      setMsg("清除失败: " + e.message);
    }
  }

  return (
    <div className="mx-auto w-full max-w-5xl p-3 sm:p-4 lg:p-6">
      <PageHeader title="设置"/>
      <div className="space-y-4">
        <Card title="阿里云百炼">
          <Field
            label="API Key"
            hint={secretConfigured.DASHSCOPE_API_KEY ? "已配置" : "尚未配置"}
          >
            <input
              type="password"
              autoComplete="new-password"
              className="w-full"
              value={values.DASHSCOPE_API_KEY || ""}
              onChange={(e) => setValues({ ...values, DASHSCOPE_API_KEY: e.target.value })}
              placeholder={secretConfigured.DASHSCOPE_API_KEY ? "已配置" : "sk-..."}
            />
          </Field>
          <Field label="Base URL">
            <input
              className="w-full"
              value={values.DASHSCOPE_BASE_URL || ""}
              onChange={(e) => setValues({ ...values, DASHSCOPE_BASE_URL: e.target.value })}
            />
          </Field>
          <div className="mt-2 flex items-center gap-3">
            <Button onClick={onSave}>保存设置</Button>
            <Button
              variant="danger"
              disabled={!secretConfigured.DASHSCOPE_API_KEY}
              onClick={onClearDashscopeApiKey}
            >
              清除 API Key
            </Button>
          </div>
        </Card>

        <Card title="环境变量">
          {ROWS.map((r) => (
            <Field key={r.key} label={r.label} hint={r.hint}>
              <input
                className="w-full"
                value={values[r.key] || ""}
                onChange={(e) => setValues({ ...values, [r.key]: e.target.value })}
              />
            </Field>
          ))}
          <div className="mt-2 flex items-center gap-3">
            <Button onClick={onSave}>保存</Button>
            <span className="text-xs text-slate-400">{msg}</span>
          </div>
        </Card>

        <Card title="当前路径">
          <MetaRow label="DiffSynth-Studio 目录" value={meta.diffsynth_studio_root} />
          <MetaRow label="数据集目录" value={meta.datasets_root} />
          <MetaRow label="训练输出目录" value={meta.outputs_root} />
          <MetaRow label="模型存储目录" value={meta.model_base_path} />
        </Card>
      </div>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value?: string }) {
  return (
    <div className="flex flex-col gap-1 border-b border-slate-800 py-2 last:border-b-0 sm:flex-row sm:items-center sm:justify-between">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="max-w-full break-all text-xs text-slate-200 mono sm:max-w-[65%]">{value || "-"}</span>
    </div>
  );
}
