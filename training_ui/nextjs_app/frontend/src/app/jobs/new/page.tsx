"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { formatShellCommand } from "@/lib/format";
import { Button, Card, Field, PageHeader } from "@/components/ui";

type ModelPath = { model_id: string; file_pattern: string; local_path: string; fp8: boolean };
type StageParameters = {
  max_timestep_boundary?: number;
  min_timestep_boundary?: number;
};

const OPTIMIZERS = ["torch.optim.AdamW", "bitsandbytes.optim.Adam8bit"];
const DATASET_KIND_LABELS: Record<string, string> = {
  image: "图像",
  edit: "图像编辑",
  video: "视频",
  audio: "音频",
};

function resolvedStageParameters(recipe: any, configured?: any[]): StageParameters[] {
  return (recipe?.editable_stage_parameters || []).map((stage: any, index: number) => {
    const saved = configured?.[index] || {};
    const result: StageParameters = {};
    if (stage.max_timestep_boundary !== undefined) {
      result.max_timestep_boundary = Number(
        saved.max_timestep_boundary ?? stage.max_timestep_boundary,
      );
    }
    if (stage.min_timestep_boundary !== undefined) {
      result.min_timestep_boundary = Number(
        saved.min_timestep_boundary ?? stage.min_timestep_boundary,
      );
    }
    return result;
  });
}

export default function NewJobPage() {
  const router = useRouter();
  const [recipes, setRecipes] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [gpus, setGpus] = useState<any[]>([]);
  const [editJobId, setEditJobId] = useState("");
  const [loadedEditJobId, setLoadedEditJobId] = useState("");

  const [name, setName] = useState("");
  const [gpuIndex, setGpuIndex] = useState<string>("");
  const [modelType, setModelType] = useState<string>("");
  const [modelPaths, setModelPaths] = useState<ModelPath[]>([]);
  const [enableCustomLoraTarget, setEnableCustomLoraTarget] = useState(false);
  const [loraTargetModules, setLoraTargetModules] = useState("");
  const [loraRank, setLoraRank] = useState(32);
  const [dataset, setDataset] = useState<string>("");
  const [datasetRepeat, setDatasetRepeat] = useState(1);
  const [resolutionMode, setResolutionMode] = useState<"max_pixels" | "hw">("max_pixels");
  const [maxPixels, setMaxPixels] = useState(1048576);
  const [height, setHeight] = useState<number | "">(1024);
  const [width, setWidth] = useState<number | "">(1024);
  const [numFrames, setNumFrames] = useState<number | "">("");
  const [numEpochs, setNumEpochs] = useState(5);
  const [learningRate, setLearningRate] = useState(1e-4);
  const [optimizer, setOptimizer] = useState(OPTIMIZERS[0]);
  const [samplePrompts, setSamplePrompts] = useState("a dog");
  const [startNow, setStartNow] = useState(true);

  const [gradAccum, setGradAccum] = useState<number | "">(1);
  const [datasetNumWorkers, setDatasetNumWorkers] = useState<number | "">(0);
  const [findUnusedParameters, setFindUnusedParameters] = useState(false);
  const [extraInputs, setExtraInputs] = useState("");
  const [stageParameters, setStageParameters] = useState<StageParameters[]>([]);

  const [previewCmd, setPreviewCmd] = useState<string[]>([]);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    api.recipes().then((r) => setRecipes(r.recipes || []));
    api.listDatasets().then((r) => setDatasets(r.datasets || []));
    api.gpu().then((r) => setGpus(r.gpus || []));
    setEditJobId(new URLSearchParams(window.location.search).get("edit") || "");
  }, []);

  useEffect(() => {
    if (!editJobId || loadedEditJobId === editJobId || recipes.length === 0) return;
    api.getJob(editJobId)
      .then((job) => {
        const cfg = job.config || {};
        const recipe = recipes.find((item) => item.name === cfg.model_type);
        setName(job.name || "");
        setGpuIndex(String(cfg.gpu_index ?? 0));
        setModelType(cfg.model_type || "");
        setModelPaths(cfg.model_paths || []);
        setEnableCustomLoraTarget(!!cfg.enable_custom_lora_target);
        setLoraTargetModules(cfg.lora_target_modules || "");
        setLoraRank(cfg.lora_rank ?? 32);
        setDataset(cfg.dataset || "");
        setDatasetRepeat(cfg.dataset_repeat ?? 50);
        setResolutionMode(cfg.resolution_mode || "max_pixels");
        setMaxPixels(cfg.max_pixels ?? 1048576);
        setHeight(cfg.height ?? "");
        setWidth(cfg.width ?? "");
        setNumFrames(cfg.num_frames ?? "");
        setNumEpochs(cfg.num_epochs ?? 5);
        setLearningRate(cfg.learning_rate ?? 1e-4);
        setOptimizer(cfg.optimizer || OPTIMIZERS[0]);
        setSamplePrompts((cfg.sample_prompts || []).join("\n"));
        setGradAccum(cfg.gradient_accumulation ?? 1);
        setDatasetNumWorkers(cfg.dataset_num_workers ?? 8);
        setFindUnusedParameters(!!cfg.find_unused_parameters);
        setExtraInputs(cfg.extra_inputs || "");
        setStageParameters(
          resolvedStageParameters(
            recipe,
            Array.isArray(cfg.stage_parameters) ? cfg.stage_parameters : undefined,
          ),
        );
        setStartNow(false);
        setLoadedEditJobId(editJobId);
      })
      .catch((error) => setMsg("加载任务失败: " + error.message));
  }, [editJobId, loadedEditJobId, recipes]);

  const currentRecipe = useMemo(
    () => recipes.find((r) => r.name === modelType),
    [recipes, modelType],
  );
  function applyRecipe(name: string) {
    setModelType(name);
    const r = recipes.find((x) => x.name === name);
    if (!r) return;
    setModelPaths(
      (r.default_model_paths || []).map((mp: any) => ({
        model_id: mp.model_id || "",
        file_pattern: mp.file_pattern || "",
        local_path: mp.local_path || "",
        fp8: !!mp.fp8,
      })),
    );
    setLoraTargetModules(r.default_lora_target || "");
    setEnableCustomLoraTarget(!!r.default_enable_custom_lora_target);
    setLoraRank(r.default_lora_rank || 32);
    setResolutionMode(r.default_resolution_mode || "max_pixels");
    setMaxPixels(r.default_max_pixels || 1048576);
    setHeight(r.default_height ?? "");
    setWidth(r.default_width ?? "");
    setNumFrames(r.default_num_frames ?? "");
    setNumEpochs(r.default_epochs || 5);
    setLearningRate(r.default_lr || 1e-4);
    setDatasetRepeat(r.default_dataset_repeat || 1);
    setExtraInputs(r.default_extra_inputs || "");
    setFindUnusedParameters(!!r.default_find_unused_parameters);
    setOptimizer(r.default_optimizer || OPTIMIZERS[0]);
    setGradAccum(r.default_gradient_accumulation ?? 1);
    setDatasetNumWorkers(r.default_dataset_num_workers ?? 0);
    setSamplePrompts((r.default_sample_prompts || []).join("\n"));
    setStageParameters(resolvedStageParameters(r));
  }

  function currentConfig() {
    const config: Record<string, any> = {
      model_type: modelType,
      gpu_index: Number(gpuIndex),
      model_paths: modelPaths.filter((mp) => mp.model_id || mp.local_path),
      enable_custom_lora_target: enableCustomLoraTarget,
      lora_target_modules: loraTargetModules,
      lora_rank: loraRank,
      dataset,
      dataset_repeat: datasetRepeat,
      num_epochs: numEpochs,
      learning_rate: Number(learningRate),
      optimizer,
      sample_prompts: samplePrompts
        .split(/\r?\n/)
        .map((p) => p.trim())
        .filter(Boolean),
      gradient_accumulation: gradAccum === "" ? null : Number(gradAccum),
      dataset_num_workers: datasetNumWorkers === "" ? null : Number(datasetNumWorkers),
      find_unused_parameters: findUnusedParameters,
      extra_inputs: extraInputs || null,
    };
    if (!currentRecipe?.disable_sections?.includes("resolution")) {
      config.resolution_mode = resolutionMode;
      config.max_pixels = maxPixels;
      config.height = height === "" ? null : Number(height);
      config.width = width === "" ? null : Number(width);
    }
    if (currentRecipe?.dataset_kind === "video") {
      config.num_frames = numFrames === "" ? null : Number(numFrames);
    }
    if (stageParameters.some((stage) => Object.keys(stage).length > 0)) {
      config.stage_parameters = stageParameters;
    }
    return config;
  }

  async function onPreview() {
    setMsg("");
    try {
      const r = await api.previewCommand(currentConfig());
      setPreviewCmd(r.argv || []);
    } catch (e: any) {
      setMsg("预览失败: " + e.message);
    }
  }

  async function onSubmit() {
    setMsg("");
    if (!name.trim()) {
      setMsg("请填任务名");
      return;
    }
    if (!modelType) {
      setMsg("请选择模型类型");
      return;
    }
    if (gpuIndex === "") {
      setMsg("请选择运行 GPU");
      return;
    }
    if (!dataset) {
      setMsg("请选择数据集");
      return;
    }
    try {
      const r = editJobId
        ? await api.updateJob(editJobId, name.trim(), currentConfig())
        : await api.createJob(name.trim(), currentConfig(), startNow);
      setMsg(editJobId ? `任务 ${r.name} 已更新` : `任务 ${r.name} 已创建`);
      router.push(`/jobs/${r.id}`);
    } catch (e: any) {
      setMsg((editJobId ? "保存失败: " : "创建失败: ") + e.message);
    }
  }

  return (
    <div className="mx-auto w-full max-w-screen-2xl p-3 sm:p-4 lg:p-6">
      <PageHeader
        title={editJobId ? "编辑训练任务" : "新建训练任务"}
        actions={
          <Link href="/jobs">
            <Button variant="ghost" size="sm">
              ← 返回列表
            </Button>
          </Link>
        }
      />

      <div className="grid min-w-0 grid-cols-1 gap-4 2xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-4">
          <Card title="任务基本信息">
            <Field
              label="任务名"
              required
            >
              <input
                className="w-full"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="例：z-image-dog-v1"
                disabled={!!editJobId}
              />
            </Field>
            <Field label="运行 GPU" required>
              <select
                className="w-full"
                value={gpuIndex}
                onChange={(e) => setGpuIndex(e.target.value)}
              >
                <option value="">-- 请选择 --</option>
                {gpus.map((gpu) => (
                  <option key={gpu.index} value={gpu.index}>
                    GPU {gpu.index} · {gpu.name} · 空闲 {gpu.memory_free_mb} MB / {gpu.memory_total_mb} MB
                  </option>
                ))}
              </select>
              {gpus.length === 0 && (
                <div className="mt-2 text-xs text-amber-300">未检测到可用 NVIDIA GPU</div>
              )}
            </Field>
          </Card>

          <Card title="模型">
            <Field label="训练模型" required>
              <select
                className="w-full"
                value={modelType}
                onChange={(e) => applyRecipe(e.target.value)}
              >
                <option value="">-- 请选择 --</option>
                {recipes.map((r) => (
                  <option key={r.name} value={r.name}>
                    {r.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="模型路径列表"
            >
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="min-w-[820px]">
                  <thead>
                    <tr>
                      <th>model_id</th>
                      <th>file_pattern</th>
                      <th>local_path</th>
                      <th className="w-12 text-center">FP8</th>
                      <th className="w-10"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelPaths.map((mp, idx) => (
                      <tr key={idx}>
                        <td>
                          <input
                            className="w-full"
                            value={mp.model_id}
                            onChange={(e) => {
                              const next = [...modelPaths];
                              next[idx].model_id = e.target.value;
                              setModelPaths(next);
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className="w-full"
                            value={mp.file_pattern}
                            onChange={(e) => {
                              const next = [...modelPaths];
                              next[idx].file_pattern = e.target.value;
                              setModelPaths(next);
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className="w-full"
                            value={mp.local_path}
                            onChange={(e) => {
                              const next = [...modelPaths];
                              next[idx].local_path = e.target.value;
                              setModelPaths(next);
                            }}
                          />
                        </td>
                        <td className="text-center">
                          <input
                            type="checkbox"
                            checked={mp.fp8}
                            onChange={(e) => {
                              const next = [...modelPaths];
                              next[idx].fp8 = e.target.checked;
                              setModelPaths(next);
                            }}
                          />
                        </td>
                        <td>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setModelPaths(modelPaths.filter((_, i) => i !== idx))}
                          >
                            ×
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setModelPaths([
                      ...modelPaths,
                      { model_id: "", file_pattern: "", local_path: "", fp8: false },
                    ])
                  }
                >
                  + 添加一行
                </Button>
              </div>
            </Field>

            <Field label="LoRA 设置">
              <div className="mb-2 flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:gap-6">
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={enableCustomLoraTarget}
                    onChange={(e) => setEnableCustomLoraTarget(e.target.checked)}
                  />
                  启用自定义 LoRA 层
                </label>
                <div className="flex w-full items-center justify-between gap-2 text-sm text-slate-300 sm:w-auto sm:justify-start">
                  lora_rank
                  <input
                    type="number"
                    className="w-24"
                    value={loraRank}
                    onChange={(e) => setLoraRank(Number(e.target.value))}
                  />
                </div>
              </div>
              <input
                className="w-full"
                placeholder="逗号分隔的层名"
                value={loraTargetModules}
                onChange={(e) => setLoraTargetModules(e.target.value)}
                disabled={!enableCustomLoraTarget}
              />
            </Field>
          </Card>

          <Card title="数据集">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field label="选择数据集" required>
                <select
                  className="w-full"
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="">-- 请选择 --</option>
                  {datasets.map((d) => (
                    <option key={d.name} value={d.name}>
                      {DATASET_KIND_LABELS[d.kind] || d.kind} · {d.name}（{d.num_items} 条）
                    </option>
                  ))}
                </select>
              </Field>
              <Field
                label={
                  currentRecipe?.dataset_repeat_stage_index !== null &&
                  currentRecipe?.dataset_repeat_stage_index !== undefined
                    ? "训练阶段数据集重复次数"
                    : "数据集重复次数"
                }
                hint={
                  currentRecipe?.dataset_repeat_stage_index !== null &&
                  currentRecipe?.dataset_repeat_stage_index !== undefined
                    ? "data_process 阶段使用模型固定值；这里控制后续训练阶段。"
                    : undefined
                }
              >
                <input
                  type="number"
                  className="w-full"
                  value={datasetRepeat}
                  onChange={(e) => setDatasetRepeat(Number(e.target.value))}
                />
              </Field>
            </div>

            {!currentRecipe?.disable_sections?.includes("resolution") && (
              <Field label="分辨率">
                <div className="grid grid-cols-1 gap-3 text-sm text-slate-300 xl:grid-cols-2">
                <label className="flex flex-wrap items-center gap-2">
                  <input
                    type="radio"
                    checked={resolutionMode === "max_pixels"}
                    onChange={() => setResolutionMode("max_pixels")}
                  />
                  max_pixels
                  <input
                    type="number"
                    className="ml-auto w-32"
                    value={maxPixels}
                    onChange={(e) => setMaxPixels(Number(e.target.value))}
                    disabled={resolutionMode !== "max_pixels"}
                  />
                </label>
                <label className="flex flex-wrap items-center gap-2">
                  <input
                    type="radio"
                    checked={resolutionMode === "hw"}
                    onChange={() => setResolutionMode("hw")}
                  />
                  height × width
                  <input
                    type="number"
                    className="ml-auto w-20"
                    value={height}
                    onChange={(e) => setHeight(e.target.value === "" ? "" : Number(e.target.value))}
                    disabled={resolutionMode !== "hw"}
                  />
                  <span>×</span>
                  <input
                    type="number"
                    className="w-20"
                    value={width}
                    onChange={(e) => setWidth(e.target.value === "" ? "" : Number(e.target.value))}
                    disabled={resolutionMode !== "hw"}
                  />
                </label>
                </div>
              </Field>
            )}

            {currentRecipe?.dataset_kind === "video" && (
              <Field label="num_frames">
                <input
                  type="number"
                  className="w-32"
                  value={numFrames}
                  onChange={(e) => setNumFrames(e.target.value === "" ? "" : Number(e.target.value))}
                />
              </Field>
            )}

          </Card>

          {stageParameters.some((stage) => Object.keys(stage).length > 0) && (
            <Card title="多阶段训练参数">
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {stageParameters.map((stage, index) => {
                  if (Object.keys(stage).length === 0) return null;
                  return (
                    <div key={index} className="border-l-2 border-slate-700 pl-4">
                      <div className="mb-3 text-sm font-medium text-slate-200">
                        第 {index + 1} 阶段
                      </div>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        {stage.min_timestep_boundary !== undefined && (
                          <Field label="min_timestep_boundary">
                            <input
                              type="number"
                              min={0}
                              max={1}
                              step="any"
                              className="w-full"
                              value={stage.min_timestep_boundary}
                              onChange={(event) => {
                                const next = stageParameters.map((item) => ({ ...item }));
                                next[index].min_timestep_boundary = Number(event.target.value);
                                setStageParameters(next);
                              }}
                            />
                          </Field>
                        )}
                        {stage.max_timestep_boundary !== undefined && (
                          <Field label="max_timestep_boundary">
                            <input
                              type="number"
                              min={0}
                              max={1}
                              step="any"
                              className="w-full"
                              value={stage.max_timestep_boundary}
                              onChange={(event) => {
                                const next = stageParameters.map((item) => ({ ...item }));
                                next[index].max_timestep_boundary = Number(event.target.value);
                                setStageParameters(next);
                              }}
                            />
                          </Field>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          <Card title="训练">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <Field label="epoch 数量">
                <input
                  type="number"
                  className="w-full"
                  value={numEpochs}
                  onChange={(e) => setNumEpochs(Number(e.target.value))}
                />
              </Field>
              <Field label="learning_rate">
                <input
                  type="number"
                  step="any"
                  className="w-full"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                />
              </Field>
              <Field label="Optimizer">
                <select
                  className="w-full"
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value)}
                >
                  {OPTIMIZERS.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Field
              label="测试 prompts"
              required
              hint="训练完成后，使用最后一个 .safetensors 进行采样，每行一个 prompt。"
            >
              <textarea
                className="min-h-28 w-full resize-y"
                value={samplePrompts}
                onChange={(e) => setSamplePrompts(e.target.value)}
              />
            </Field>
          </Card>

          <Card title="训练选项">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field label="gradient_accumulation_steps">
                  <input
                    type="number"
                    className="w-full"
                    value={gradAccum}
                    onChange={(e) =>
                      setGradAccum(e.target.value === "" ? "" : Number(e.target.value))
                    }
                  />
                </Field>
                <Field label="dataset_num_workers">
                  <input
                    type="number"
                    className="w-full"
                    value={datasetNumWorkers}
                    onChange={(e) =>
                      setDatasetNumWorkers(e.target.value === "" ? "" : Number(e.target.value))
                    }
                  />
                </Field>
                <Field label="find_unused_parameters">
                  <label className="flex items-center gap-2 text-sm text-slate-300 mt-1.5">
                    <input
                      type="checkbox"
                      checked={findUnusedParameters}
                      onChange={(e) => setFindUnusedParameters(e.target.checked)}
                    />
                    启用
                  </label>
                </Field>
              </div>
              <Field
                label="extra_inputs"
              >
                <input
                  className="w-full"
                  value={extraInputs}
                  onChange={(e) => setExtraInputs(e.target.value)}
                />
              </Field>
          </Card>
        </div>

        <div className="min-w-0 space-y-4">
          <Card title={editJobId ? "保存" : "创建"} className="sticky top-4">
            {!editJobId && <label className="flex items-center gap-2 text-sm text-slate-300 mb-3">
              <input
                type="checkbox"
                checked={startNow}
                onChange={(e) => setStartNow(e.target.checked)}
              />
              创建后立即启动
            </label>}
            <div className="flex flex-col gap-2">
              <Button onClick={onSubmit}>{editJobId ? "保存修改" : "创建任务"}</Button>
              <Button variant="outline" onClick={onPreview}>
                预览启动命令
              </Button>
            </div>
            {msg && <div className="text-xs text-slate-400 mt-3">{msg}</div>}
          </Card>

          {previewCmd.length > 0 && (
            <Card title="启动命令">
              <pre className="whitespace-pre mono text-[11px] text-slate-300 max-h-80 overflow-auto">
                {formatShellCommand(previewCmd)}
              </pre>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
