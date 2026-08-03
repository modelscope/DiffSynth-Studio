"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { apiUrl } from "@/lib/basePath";
import { Button, Card, EmptyState, Field, PageHeader } from "@/components/ui";

const CAPTION_MODELS = [
  "qwen3.7-plus",
  "qwen3.6-plus",
  "qwen3.6-flash",
  "qwen3.5-plus",
  "qwen3.5-flash",
  "qwen3-vl-plus",
];
const DEFAULT_INSTRUCTION =
  "请生成准确、自然、适合图像生成模型训练的英文 prompt。只返回 prompt 正文。";

type Filter = "all" | "labeled" | "missing";
type BatchScope = "current" | "selected" | "missing";
type BatchResult = {
  running: boolean;
  current: number;
  total: number;
  success: number;
  failed: number;
  errors: string[];
};

function mediaField(mediaPath: string) {
  const extension = mediaPath.split(".").pop()?.toLowerCase();
  if (["mp4", "webm", "mov", "mkv", "avi"].includes(extension || "")) return "video";
  if (["wav", "mp3", "flac", "ogg", "m4a", "aac"].includes(extension || "")) return "audio";
  return "image";
}

export default function DatasetDetailPage() {
  const params = useParams<{ name: string }>();
  const name = decodeURIComponent(params.name);
  const [detail, setDetail] = useState<any>(null);
  const [msg, setMsg] = useState("");
  const [selected, setSelected] = useState("");
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<Filter>("all");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [batchOpen, setBatchOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [captionModel, setCaptionModel] = useState("qwen3.7-plus");
  const [instruction, setInstruction] = useState(DEFAULT_INSTRUCTION);
  const [batchScope, setBatchScope] = useState<BatchScope>("selected");
  const [rewriteExisting, setRewriteExisting] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [generatedDrafts, setGeneratedDrafts] = useState(false);
  const stopBatchRef = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function reload() {
    try {
      const data = await api.datasetDetail(name);
      setDetail(data);
      setGeneratedDrafts(false);
      setSelected((current) =>
        current && data.media?.includes(current) ? current : (data.media?.[0] || ""),
      );
      setSelection((current) => new Set([...current].filter((item) => data.media.includes(item))));
    } catch (error: any) {
      setMsg("加载失败: " + error.message);
    }
  }

  useEffect(() => {
    reload();
  }, [name]);

  const metadataByPath = useMemo(() => {
    const index = new Map<string, any>();
    for (const item of detail?.metadata || []) index.set(getMediaPath(item), item);
    return index;
  }, [detail]);

  const filteredMedia = useMemo(() => {
    if (!detail) return [];
    return detail.media.filter((mediaPath: string) => {
      const prompt = String(metadataByPath.get(mediaPath)?.prompt || "").trim();
      if (filter === "labeled") return Boolean(prompt);
      if (filter === "missing") return !prompt;
      return true;
    });
  }, [detail, filter, metadataByPath]);

  const selectedItem = useMemo(() => {
    if (!detail || !selected) return null;
    return metadataByPath.get(selected) || { [mediaField(selected)]: selected, prompt: "" };
  }, [detail, metadataByPath, selected]);

  function chooseFiles(files: FileList | null) {
    setPendingFiles(files ? Array.from(files) : []);
  }

  async function onUpload() {
    if (pendingFiles.length === 0) return;
    setUploading(true);
    try {
      const result = await api.uploadFiles(name, pendingFiles);
      setMsg(`已上传 ${result.saved?.length ?? 0} 个文件`);
      setPendingFiles([]);
      setUploadOpen(false);
      if (fileRef.current) fileRef.current.value = "";
      await reload();
    } catch (error: any) {
      setMsg("上传失败: " + error.message);
    } finally {
      setUploading(false);
    }
  }

  function toggleSelection(mediaPath: string) {
    setSelection((current) => {
      const next = new Set(current);
      if (next.has(mediaPath)) next.delete(mediaPath);
      else next.add(mediaPath);
      return next;
    });
  }

  async function onDeleteSelected() {
    if (selection.size === 0) {
      setMsg("请先选择要删除的素材");
      return;
    }
    if (generatedDrafts) {
      setMsg("请先保存或放弃 AI 生成结果，再删除素材");
      return;
    }
    const files = [...selection];
    if (!confirm(`确认永久删除选中的 ${files.length} 个文件？`)) return;
    setDeleting(true);
    try {
      const result = await api.deleteDatasetMedia(name, files);
      setMsg(`已删除 ${result.deleted?.length ?? 0} 个文件`);
      setSelection(new Set());
      await reload();
    } catch (error: any) {
      setMsg("删除失败: " + error.message);
    } finally {
      setDeleting(false);
    }
  }

  function updatePromptLocally(mediaPath: string, prompt: string) {
    setDetail((current: any) => {
      if (!current) return current;
      const items = [...current.metadata];
      const index = items.findIndex((item) => getMediaPath(item) === mediaPath);
      if (index >= 0) items[index] = { ...items[index], prompt };
      else items.push({ [mediaField(mediaPath)]: mediaPath, prompt });
      return { ...current, metadata: items };
    });
  }

  function batchTargets(): string[] {
    if (!detail) return [];
    let candidates: string[];
    if (batchScope === "current") candidates = selected ? [selected] : [];
    else if (batchScope === "selected") candidates = [...selection];
    else candidates = detail.media.filter((path: string) => {
      return !String(metadataByPath.get(path)?.prompt || "").trim();
    });
    return candidates.filter((path) => {
      if (!isImagePath(path)) return false;
      if (rewriteExisting) return true;
      return !String(metadataByPath.get(path)?.prompt || "").trim();
    });
  }

  async function startBatchGeneration() {
    const targets = batchTargets();
    if (targets.length === 0) {
      setBatchResult({
        running: false,
        current: 0,
        total: 0,
        success: 0,
        failed: 0,
        errors: ["没有符合条件的图像"],
      });
      return;
    }
    stopBatchRef.current = false;
    let success = 0;
    let failed = 0;
    const errors: string[] = [];
    setBatchResult({ running: true, current: 0, total: targets.length, success, failed, errors });

    for (let index = 0; index < targets.length; index += 1) {
      if (stopBatchRef.current) break;
      const mediaPath = targets[index];
      setBatchResult({
        running: true,
        current: index + 1,
        total: targets.length,
        success,
        failed,
        errors: [...errors],
      });
      try {
        const currentPrompt = String(metadataByPath.get(mediaPath)?.prompt || "");
        const result = await api.generateDatasetPrompt(
          name,
          mediaPath,
          captionModel,
          currentPrompt,
          instruction,
        );
        updatePromptLocally(mediaPath, result.prompt);
        success += 1;
      } catch (error: any) {
        failed += 1;
        errors.push(`${mediaPath}: ${error.message || "生成失败"}`);
      }
    }
    if (success > 0) setGeneratedDrafts(true);
    setBatchResult({
      running: false,
      current: Math.min(success + failed, targets.length),
      total: targets.length,
      success,
      failed,
      errors,
    });
  }

  async function saveGeneratedDrafts() {
    if (!detail) return;
    try {
      await api.saveMetadata(name, detail.metadata);
      setMsg("AI 生成结果已保存");
      setGeneratedDrafts(false);
      setBatchOpen(false);
      setBatchResult(null);
      await reload();
    } catch (error: any) {
      setMsg("保存失败: " + error.message);
    }
  }

  async function discardGeneratedDrafts() {
    stopBatchRef.current = true;
    setBatchOpen(false);
    setBatchResult(null);
    await reload();
    setMsg("已放弃未保存的 AI 生成结果");
  }

  async function onSaveOne(newItem: any) {
    if (!detail) return;
    if (generatedDrafts) {
      setMsg("请先保存或放弃批量 AI 生成结果");
      return;
    }
    const items = [...detail.metadata];
    const mediaPath = getMediaPath(newItem);
    const index = items.findIndex((item) => getMediaPath(item) === mediaPath);
    if (index >= 0) {
      // Keep fields saved in earlier edits; the editor submits only the fields
      // currently present in its JSON textarea.
      items[index] = { ...items[index], ...newItem };
    }
    else items.push(newItem);
    try {
      await api.saveMetadata(name, items);
      setMsg("已保存");
      await reload();
    } catch (error: any) {
      setMsg("保存失败: " + error.message);
    }
  }

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-4 lg:p-6">
      <PageHeader
        title={`数据集 · ${name}`}
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => setUploadOpen(true)}>
              上传文件
            </Button>
            <Button variant="outline" size="sm" onClick={() => {
              setBatchResult(null);
              setBatchOpen(true);
            }}>
              AI 生成 Prompt
            </Button>
            <Button
              variant="danger"
              size="sm"
              disabled={deleting || selection.size === 0}
              onClick={onDeleteSelected}
            >
              {deleting ? "删除中..." : `删除选中${selection.size ? ` (${selection.size})` : ""}`}
            </Button>
            <Link href="/datasets">
              <Button variant="ghost" size="sm">← 返回列表</Button>
            </Link>
          </>
        }
      />

      {msg && <div className="mb-3 text-xs text-slate-400">{msg}</div>}
      {generatedDrafts && (
        <div className="mb-3 flex items-center justify-between border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          <span>AI 生成结果尚未写入 metadata.jsonl</span>
          <div className="flex gap-2">
            <Button size="sm" onClick={saveGeneratedDrafts}>保存结果</Button>
            <Button variant="ghost" size="sm" onClick={discardGeneratedDrafts}>放弃</Button>
          </div>
        </div>
      )}

      {!detail ? (
        <div className="text-slate-400">加载中...</div>
      ) : (
        <>
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_440px] gap-4">
            <Card
              title={`素材 (${detail.media.length})`}
              actions={
                <div className="flex items-center gap-2">
                  <select
                    className="h-8 text-xs"
                    value={filter}
                    onChange={(event) => setFilter(event.target.value as Filter)}
                  >
                    <option value="all">全部</option>
                    <option value="labeled">已有 Prompt</option>
                    <option value="missing">无 Prompt</option>
                  </select>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelection(new Set(filteredMedia))}
                  >
                    全选当前
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setSelection(new Set())}>
                    清除选择
                  </Button>
                  <span className="text-xs text-slate-400">已选 {selection.size}</span>
                </div>
              }
            >
              {filteredMedia.length === 0 ? (
                <EmptyState title={detail.media.length ? "没有符合筛选条件的素材" : "空数据集"} />
              ) : (
                <div className="grid max-h-[calc(100vh-250px)] grid-cols-4 gap-3 overflow-y-auto pr-1 2xl:grid-cols-5">
                  {filteredMedia.map((mediaPath: string) => {
                    const image = isImagePath(mediaPath);
                    const prompt = String(metadataByPath.get(mediaPath)?.prompt || "").trim();
                    const checked = selection.has(mediaPath);
                    const active = selected === mediaPath;
                    const url = apiUrl(
                      `/api/datasets/${encodeURIComponent(name)}/media/${encodeURIComponent(mediaPath)}`,
                    );
                    return (
                      <div
                        key={mediaPath}
                        role="button"
                        tabIndex={0}
                        className={
                          "relative cursor-pointer overflow-hidden rounded border transition-all " +
                          (active
                            ? "border-blue-500 ring-2 ring-blue-500/30"
                            : checked
                              ? "border-emerald-500 ring-2 ring-emerald-500/20"
                              : "border-slate-800 hover:border-slate-600")
                        }
                        onClick={() => setSelected(mediaPath)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") setSelected(mediaPath);
                        }}
                      >
                        <button
                          type="button"
                          className={
                            "absolute right-2 top-2 z-10 grid h-5 w-5 place-items-center rounded-full border-2 shadow " +
                            (checked
                              ? "border-emerald-300 bg-emerald-500"
                              : "border-white/80 bg-slate-950/70")
                          }
                          aria-label={checked ? `取消选择 ${mediaPath}` : `选择 ${mediaPath}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleSelection(mediaPath);
                          }}
                        >
                          {checked && <span className="h-2 w-2 rounded-full bg-white" />}
                        </button>
                        {image ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={url} alt={mediaPath} className="h-28 w-full object-cover" />
                        ) : (
                          <div className="flex h-28 w-full items-center justify-center bg-slate-800 text-xs text-slate-400">
                            {mediaPath.split(".").pop()?.toUpperCase()}
                          </div>
                        )}
                        <div className="bg-slate-950/70 px-2 py-1.5">
                          <div className="truncate text-[11px] text-slate-300">{mediaPath}</div>
                          <div className={prompt ? "text-[10px] text-emerald-400" : "text-[10px] text-slate-500"}>
                            {prompt ? "已有 Prompt" : "无 Prompt"}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </Card>

            <div className="min-w-0 space-y-4">
              <Card title={selected ? `编辑标注 · ${selected}` : "编辑标注"}>
                {!selectedItem ? (
                  <EmptyState title="选择一个素材以编辑标注" />
                ) : (
                  <ItemEditor
                    key={`${selected}-${String(selectedItem.prompt || "")}`}
                    item={selectedItem}
                    onSave={onSaveOne}
                    previewUrl={apiUrl(
                      `/api/datasets/${encodeURIComponent(name)}/media/${encodeURIComponent(selected)}`,
                    )}
                    captionModel={captionModel}
                    instruction={instruction}
                    onModelChange={setCaptionModel}
                    onInstructionChange={setInstruction}
                    onGeneratePrompt={async (model, currentPrompt, requestInstruction) => {
                      const result = await api.generateDatasetPrompt(
                        name,
                        selected,
                        model,
                        currentPrompt,
                        requestInstruction,
                      );
                      return result.prompt;
                    }}
                  />
                )}
              </Card>
            </div>
          </div>

          <div className="mt-4">
            <Card title="metadata.jsonl">
              <MetadataTable
                items={detail.metadata}
                disabled={generatedDrafts}
                onSave={async (items) => {
                  await api.saveMetadata(name, items);
                  setMsg("已保存全部 metadata.jsonl");
                  await reload();
                }}
              />
            </Card>
          </div>
        </>
      )}

      {uploadOpen && (
        <Modal title="上传文件" onClose={() => !uploading && setUploadOpen(false)}>
          <div
            className={
              "cursor-pointer rounded border-2 border-dashed px-4 py-8 transition-colors " +
              (dragging
                ? "border-blue-400 bg-blue-500/10"
                : pendingFiles.length
                  ? "border-emerald-500/50 bg-emerald-500/5"
                  : "border-slate-700 bg-slate-950/30")
            }
            role="button"
            tabIndex={0}
            onClick={() => fileRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") fileRef.current?.click();
            }}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={(event) => {
              event.preventDefault();
              if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFiles(event.dataTransfer.files);
            }}
          >
            {pendingFiles.length === 0 ? (
              <div className="text-center">
                <div className="text-sm font-medium text-slate-200">点击选择或拖入文件</div>
                <div className="mt-1 text-xs text-slate-400">图像 / 视频 / 音频 / ZIP / TAR / TGZ</div>
              </div>
            ) : (
              <div>
                <div className="mb-2 text-sm text-emerald-300">已选择 {pendingFiles.length} 个文件</div>
                <div className="max-h-48 space-y-1 overflow-y-auto">
                  {pendingFiles.map((file, index) => (
                    <div key={`${file.name}-${index}`} className="flex justify-between gap-3 text-xs">
                      <span className="truncate text-slate-300">{file.name}</span>
                      <span className="shrink-0 text-slate-400">{formatBytes(file.size)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept="image/*,video/*,audio/*,.zip,.tar,.tgz,.tar.gz,application/zip,application/x-tar,application/gzip"
            className="hidden"
            onChange={(event) => chooseFiles(event.target.files)}
          />
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="ghost" disabled={uploading} onClick={() => setUploadOpen(false)}>取消</Button>
            <Button disabled={uploading || pendingFiles.length === 0} onClick={onUpload}>
              {uploading ? "上传中..." : `上传${pendingFiles.length ? ` (${pendingFiles.length})` : ""}`}
            </Button>
          </div>
        </Modal>
      )}

      {batchOpen && (
        <Modal
          title="AI 生成 Prompt"
          onClose={() => {
            if (!batchResult?.running && !generatedDrafts) {
              setBatchOpen(false);
              setBatchResult(null);
            }
          }}
        >
          <Field label="处理范围">
            <select
              className="w-full"
              value={batchScope}
              disabled={batchResult?.running}
              onChange={(event) => setBatchScope(event.target.value as BatchScope)}
            >
              <option value="current">当前图片</option>
              <option value="selected">已选择图片 ({selection.size})</option>
              <option value="missing">所有无 Prompt 图片</option>
            </select>
          </Field>
          <Field label="选择模型">
            <select
              className="w-full"
              value={captionModel}
              disabled={batchResult?.running}
              onChange={(event) => setCaptionModel(event.target.value)}
            >
              {CAPTION_MODELS.map((model) => <option key={model}>{model}</option>)}
            </select>
          </Field>
          <Field label="已有 Prompt">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={rewriteExisting}
                disabled={batchResult?.running}
                onChange={(event) => setRewriteExisting(event.target.checked)}
              />
              修改已有 Prompt；不勾选时仅处理空 Prompt
            </label>
          </Field>
          <Field label="生成要求">
            <textarea
              className="min-h-24 w-full"
              value={instruction}
              disabled={batchResult?.running}
              onChange={(event) => setInstruction(event.target.value)}
            />
          </Field>

          {batchResult && (
            <div className="mb-4 border-t border-slate-800 pt-3 text-sm text-slate-300">
              <div>
                {batchResult.running ? "正在处理" : "处理结束"} {batchResult.current} / {batchResult.total}
              </div>
              <div className="mt-1 text-xs text-slate-400">
                成功 {batchResult.success}，失败 {batchResult.failed}
              </div>
              {batchResult.errors.length > 0 && (
                <div className="mt-2 max-h-28 overflow-y-auto text-xs text-red-400">
                  {batchResult.errors.map((error, index) => <div key={index}>{error}</div>)}
                </div>
              )}
            </div>
          )}

          <div className="flex justify-end gap-2">
            {batchResult?.running ? (
              <Button variant="danger" onClick={() => { stopBatchRef.current = true; }}>
                停止
              </Button>
            ) : generatedDrafts ? (
              <>
                <Button variant="ghost" onClick={discardGeneratedDrafts}>放弃</Button>
                <Button onClick={saveGeneratedDrafts}>保存全部结果</Button>
              </>
            ) : (
              <>
                <Button variant="ghost" onClick={() => setBatchOpen(false)}>取消</Button>
                <Button onClick={startBatchGeneration}>开始生成</Button>
              </>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}

function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-6" onMouseDown={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-xl overflow-y-auto rounded-md border border-slate-700 bg-slate-900 p-5 shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 className="text-base font-semibold text-slate-100">{title}</h2>
          <button type="button" className="text-xl text-slate-400 hover:text-white" onClick={onClose}>×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

function ItemEditor({
  item,
  onSave,
  previewUrl,
  captionModel,
  instruction,
  onModelChange,
  onInstructionChange,
  onGeneratePrompt,
}: {
  item: any;
  onSave: (item: any) => void;
  previewUrl: string;
  captionModel: string;
  instruction: string;
  onModelChange: (model: string) => void;
  onInstructionChange: (instruction: string) => void;
  onGeneratePrompt: (model: string, currentPrompt: string, instruction: string) => Promise<string>;
}) {
  const mediaPath = getMediaPath(item);
  const [text, setText] = useState(String(item.prompt || ""));
  const [extras, setExtras] = useState(
    JSON.stringify(
      Object.fromEntries(
        Object.entries(item).filter(([key]) => !["image", "video", "audio", "prompt"].includes(key)),
      ),
      null,
      2,
    ),
  );
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState("");
  const image = isImagePath(mediaPath);

  async function generatePrompt() {
    setGenerating(true);
    setGenerateError("");
    try {
      setText(await onGeneratePrompt(captionModel, text, instruction));
    } catch (error: any) {
      setGenerateError(error.message || "生成失败");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div>
      {image && (
        <div className="mb-3 flex justify-center bg-black/20">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt={mediaPath} className="max-h-[320px] max-w-full object-contain" />
        </div>
      )}
      <Field label="Caption (Prompt)">
        <textarea className="min-h-32 w-full" value={text} onChange={(event) => setText(event.target.value)} />
      </Field>
      {image && (
        <div className="mb-4 border-t border-slate-800 pt-4">
          <Field label="选择模型">
            <select className="w-full" value={captionModel} onChange={(event) => onModelChange(event.target.value)}>
              {CAPTION_MODELS.map((model) => <option key={model}>{model}</option>)}
            </select>
          </Field>
          <Field label="指令">
            <textarea
              className="min-h-20 w-full"
              value={instruction}
              onChange={(event) => onInstructionChange(event.target.value)}
            />
          </Field>
          <div className="flex items-center gap-3">
            <Button variant="outline" disabled={generating} onClick={generatePrompt}>
              {generating ? "生成中..." : text.trim() ? "AI 修改 Prompt" : "AI 生成 Prompt"}
            </Button>
            {generateError && (
              <span className="text-xs text-red-400">
                {generateError}
                {generateError.includes("API Key") && (
                  <Link href="/settings" className="ml-2 text-blue-400 hover:text-blue-300">前往设置</Link>
                )}
              </span>
            )}
          </div>
        </div>
      )}
      <Field label="其它字段 (JSON)">
        <textarea className="min-h-24 w-full mono text-xs" value={extras} onChange={(event) => setExtras(event.target.value)} />
      </Field>
      <Button
        onClick={() => {
          let extraObject: Record<string, unknown> = {};
          try {
            const parsed = extras.trim() ? JSON.parse(extras) : {};
            if (!isJsonObject(parsed)) throw new Error("必须是一个 JSON 对象");
            extraObject = parsed;
          } catch (error: any) {
            alert("其它字段不是合法 JSON: " + error.message);
            return;
          }
          for (const key of ["image", "video", "audio", "prompt"]) delete extraObject[key];
          const mediaFields = Object.fromEntries(
            Object.entries(item).filter(([key]) => ["image", "video", "audio"].includes(key)),
          );
          onSave({ ...mediaFields, prompt: text, ...extraObject });
        }}
      >
        保存这一条
      </Button>
    </div>
  );
}

function MetadataTable({
  items,
  disabled,
  onSave,
}: {
  items: any[];
  disabled: boolean;
  onSave: (items: any[]) => void;
}) {
  const [text, setText] = useState(items.map((item) => JSON.stringify(item)).join("\n"));
  useEffect(() => {
    setText(items.map((item) => JSON.stringify(item)).join("\n"));
  }, [items]);
  return (
    <div>
      <textarea
        className="min-h-64 w-full mono text-xs"
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
      />
      <div className="mt-2">
        <Button
          disabled={disabled}
          onClick={() => {
            let parsed: any[];
            try {
              parsed = parseMetadataText(text);
            } catch (error: any) {
              alert("metadata 不是合法 JSON: " + error.message);
              return;
            }
            if (confirm(`确认保存全部 ${parsed.length} 条 metadata 记录？`)) onSave(parsed);
          }}
        >
          保存全部 metadata.jsonl
        </Button>
      </div>
    </div>
  );
}

function getMediaPath(item: any): string {
  return item.image || item.video || item.audio || "";
}

function isImagePath(path: string): boolean {
  return /\.(png|jpe?g|webp|bmp)$/i.test(path);
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseMetadataText(text: string): any[] {
  const content = text.trim();
  if (!content) return [];
  try {
    const value = JSON.parse(content);
    const items = Array.isArray(value) ? value : [value];
    if (!items.every(isJsonObject)) throw new Error("每条 metadata 记录都必须是 JSON 对象");
    return items;
  } catch (error) {
    if (content.startsWith("[")) throw error;
  }
  return content.split(/\r?\n/).flatMap((line, index) => {
    if (!line.trim()) return [];
    try {
      const item = JSON.parse(line);
      if (!isJsonObject(item)) throw new Error("必须是 JSON 对象");
      return [item];
    } catch (error: any) {
      throw new Error(`第 ${index + 1} 行不是合法 JSON: ${error.message}`);
    }
  });
}
