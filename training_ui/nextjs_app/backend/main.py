from __future__ import annotations

import sys
import tempfile
import shutil
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from diffsynth_train_ui import (
    artifacts,
    captioning,
    config,
    datasets as ds_core,
    gpu_info,
    jobs as job_core,
    recipes as recipes_core,
    runner,
    settings as settings_core,
)


settings_core.apply_path_settings()

app = FastAPI(title="DiffSynth-Studio 训练 UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Ok(BaseModel):
    ok: bool = True
    message: str = ""


@app.get("/api/meta")
def api_meta():
    effective_settings = settings_core.get_all()
    model_base_value = effective_settings.get("DIFFSYNTH_MODEL_BASE_PATH", "").strip()
    model_base_path = Path(model_base_value).expanduser() if model_base_value else Path("models")
    if not model_base_path.is_absolute():
        model_base_path = config.DIFFSYNTH_STUDIO_ROOT / model_base_path
    return {
        "diffsynth_studio_root": str(config.DIFFSYNTH_STUDIO_ROOT),
        "ui_data_root": str(config.UI_DATA_ROOT),
        "datasets_root": str(config.DATASETS_ROOT),
        "outputs_root": str(config.OUTPUTS_ROOT),
        "model_base_path": str(model_base_path.resolve()),
    }


@app.get("/api/gpu")
def api_gpu():
    return {"gpus": gpu_info.get_gpus()}


@app.get("/api/recipes")
def api_list_recipes():
    result = []
    for name in recipes_core.list_recipes():
        r = recipes_core.get_recipe(name)
        result.append({
            "name": r.name,
            "label": r.label,
            "train_script": r.train_script,
            "config_path": r.config_path,
            "source_script": r.source_script,
            "generation_type": r.generation_type,
            "family": r.family,
            "dataset_kind": r.dataset_kind,
            "lora_base_model": r.lora_base_model,
            "default_lora_target": r.default_lora_target,
            "default_model_paths": [mp.__dict__ for mp in r.default_model_paths],
            "extra_defaults": r.extra_defaults,
            "default_data_file_keys": r.default_data_file_keys,
            "default_resolution_mode": r.default_resolution_mode,
            "default_max_pixels": r.default_max_pixels,
            "default_height": r.default_height,
            "default_width": r.default_width,
            "default_num_frames": r.default_num_frames,
            "default_extra_inputs": r.default_extra_inputs,
            "default_find_unused_parameters": r.default_find_unused_parameters,
            "default_dataset_repeat": r.default_dataset_repeat,
            "default_lr": r.default_lr,
            "default_epochs": r.default_epochs,
            "default_lora_rank": r.default_lora_rank,
            "default_dataset_num_workers": r.default_dataset_num_workers,
            "default_optimizer": r.default_optimizer,
            "default_gradient_accumulation": r.default_gradient_accumulation,
            "default_enable_custom_lora_target": r.default_enable_custom_lora_target,
            "default_sample_prompts": r.default_sample_prompts,
            "sampling": r.sampling,
            "disable_sections": r.disable_sections,
            "additional_sections": r.additional_sections,
            "dataset_repeat_stage_index": r.dataset_repeat_stage_index,
            "editable_stage_parameters": r.editable_stage_parameters,
        })
    return {"recipes": result}


class CreateDatasetReq(BaseModel):
    name: str
    kind: str = "image"


@app.get("/api/datasets")
def api_list_datasets():
    return {"datasets": [d.__dict__ for d in ds_core.list_datasets()]}


@app.post("/api/datasets")
def api_create_dataset(req: CreateDatasetReq):
    try:
        d = ds_core.create_dataset(req.name, kind=req.kind)
    except FileExistsError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return d.__dict__


@app.delete("/api/datasets/{name}")
def api_delete_dataset(name: str):
    ds_core.delete_dataset(name)
    return Ok(message=f"deleted {name}")


@app.get("/api/datasets/{name}")
def api_dataset_detail(name: str):
    try:
        p = ds_core.dataset_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {
        "name": name,
        "path": str(p),
        "media": ds_core.list_media(name),
        "metadata": ds_core.read_metadata(name),
        "extra_input_keys": ds_core.get_extra_input_keys(name),
    }


class MetadataReq(BaseModel):
    items: List[Dict[str, Any]]


class DeleteMediaReq(BaseModel):
    files: List[str]


class GeneratePromptReq(BaseModel):
    media_path: str
    model: str
    current_prompt: str = ""
    instruction: str = ""


@app.put("/api/datasets/{name}/metadata")
def api_save_metadata(name: str, req: MetadataReq):
    try:
        ds_core.dataset_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    ds_core.write_metadata(name, req.items)
    return Ok(message=f"saved {len(req.items)} items")


@app.post("/api/datasets/{name}/generate_prompt")
def api_generate_dataset_prompt(name: str, req: GeneratePromptReq):
    try:
        path = ds_core.image_path(name, req.media_path)
        prompt = captioning.generate_prompt(
            path,
            model=req.model,
            current_prompt=req.current_prompt,
            instruction=req.instruction,
        )
    except captioning.CaptioningConfigurationError as e:
        raise HTTPException(409, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(502, f"API调用异常: {type(e).__name__}: {e}")
    return {"prompt": prompt}


@app.post("/api/datasets/{name}/upload")
async def api_upload_files(name: str, files: List[UploadFile] = File(...)):
    try:
        ds_core.dataset_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    tmp_paths: List[Path] = []
    tempdir = Path(tempfile.mkdtemp(prefix="diffsynth_train_ui_upload_"))
    try:
        for f in files:
            filename = Path(f.filename or "unnamed").name
            tp = tempdir / filename
            with tp.open("wb") as output:
                while chunk := await f.read(1024 * 1024):
                    output.write(chunk)
            tmp_paths.append(tp)
        saved = ds_core.add_files(name, tmp_paths)
    finally:
        for f in files:
            await f.close()
        shutil.rmtree(tempdir, ignore_errors=True)
    return {"saved": saved}


@app.delete("/api/datasets/{name}/media")
def api_delete_media(name: str, req: DeleteMediaReq):
    try:
        deleted = ds_core.delete_media(name, req.files)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"deleted": deleted}


@app.get("/api/datasets/{name}/media/{filename:path}")
def api_get_media(name: str, filename: str):
    try:
        d = ds_core.dataset_path(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    p = (d / filename).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p)


class CreateJobReq(BaseModel):
    name: str
    config: Dict[str, Any]
    start_now: bool = True


class UpdateJobReq(BaseModel):
    name: str
    config: Dict[str, Any]


def _validate_job_config(cfg: Dict[str, Any]) -> tuple[str, str]:
    model_type = str(cfg.get("model_type") or "")
    dataset = str(cfg.get("dataset") or "")
    if not model_type:
        raise HTTPException(400, "config.model_type is required")
    if not dataset:
        raise HTTPException(400, "config.dataset is required")
    try:
        recipes_core.get_recipe(model_type)
    except KeyError:
        raise HTTPException(400, f"unknown model type: {model_type}")
    try:
        ds_core.dataset_path(dataset)
    except FileNotFoundError:
        raise HTTPException(400, f"dataset not found: {dataset}")
    gpu_index = cfg.get("gpu_index")
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise HTTPException(400, "config.gpu_index must be a non-negative integer")
    return model_type, dataset


@app.get("/api/jobs")
def api_list_jobs():
    result = []
    for j in job_core.list_jobs():
        if j.status in job_core.ACTIVE_RUN_STATUSES:
            j = runner.refresh_status(j.id)
        result.append(j.to_dict())
    return {"jobs": result}


@app.post("/api/jobs")
def api_create_job(req: CreateJobReq):
    cfg = req.config or {}
    if not req.name.strip():
        raise HTTPException(400, "name is required")
    model_type, dataset = _validate_job_config(cfg)
    try:
        job = job_core.create_job(name=req.name.strip(), model_type=model_type, dataset=dataset, config_data=cfg)
    except Exception as e:
        raise HTTPException(400, f"create job failed: {e}")
    if req.start_now:
        try:
            runner.start_job(job.id)
        except Exception as e:
            raise HTTPException(500, f"start job failed: {e}")
    result = job_core.get_job(job.id).to_dict()
    result["preview_command"] = result.get("command", [])
    return result


@app.put("/api/jobs/{job_id}")
def api_update_job(job_id: str, req: UpdateJobReq):
    cfg = req.config or {}
    if not req.name.strip():
        raise HTTPException(400, "name is required")
    model_type, dataset = _validate_job_config(cfg)
    try:
        job = job_core.edit_job(
            job_id,
            name=req.name.strip(),
            model_type=model_type,
            dataset=dataset,
            config_data=cfg,
        )
        return job.to_dict()
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(400, f"update job failed: {e}")


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    try:
        job = job_core.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    job = runner.refresh_status(job.id)
    return job.to_dict()


@app.post("/api/jobs/{job_id}/start")
def api_start_job(job_id: str):
    try:
        job_core.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    try:
        runner.start_job(job_id)
    except Exception as e:
        raise HTTPException(500, f"start failed: {e}")
    return job_core.get_job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/stop")
def api_stop_job(job_id: str):
    try:
        job_core.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    runner.stop_job(job_id)
    return job_core.get_job(job_id).to_dict()


@app.delete("/api/jobs/{job_id}")
def api_delete_job(job_id: str):
    try:
        job = job_core.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    if job.latest_run and job.latest_run.status in job_core.ACTIVE_RUN_STATUSES:
        raise HTTPException(409, "运行中的任务不能删除，请先停止任务")
    settings_core.apply_path_settings()
    output_root = config.OUTPUTS_ROOT.resolve()
    targets = {Path(job.task_dir).resolve()}
    targets.update(Path(run.output_path).resolve() for run in job_core.list_runs(job_id))
    checked_targets: List[Path] = []
    for target in targets:
        try:
            target.relative_to(output_root)
        except ValueError:
            raise HTTPException(500, f"invalid task output directory: {target}")
        if target == output_root:
            raise HTTPException(500, "refusing to delete output root")
        checked_targets.append(target)
    for target in sorted(checked_targets, key=lambda item: len(item.parts), reverse=True):
        if target.exists():
            shutil.rmtree(target)
    job_core.delete_job_records(job_id)
    return Ok(message=f"deleted {job_id}")


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def api_get_log(job_id: str):
    try:
        return runner.read_log(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")


@app.get("/api/jobs/{job_id}/samples")
def api_job_samples(job_id: str):
    return {"samples": artifacts.list_samples(job_id)}


@app.get("/api/jobs/{job_id}/sampling_status")
def api_job_sampling_status(job_id: str):
    return artifacts.read_sampling_status(job_id)


@app.get("/api/jobs/{job_id}/checkpoints")
def api_job_checkpoints(job_id: str):
    return {"checkpoints": artifacts.list_checkpoints(job_id)}


@app.get("/api/jobs/{job_id}/files")
def api_job_files(job_id: str):
    return {"files": artifacts.list_files(job_id)}


@app.get("/api/jobs/{job_id}/loss")
def api_job_loss(job_id: str):
    return {"series": artifacts.read_loss(job_id)}


@app.get("/api/jobs/{job_id}/artifact")
def api_job_artifact(job_id: str, path: str, download: bool = False):
    try:
        p = artifacts.read_artifact(job_id, path)
    except FileNotFoundError:
        raise HTTPException(404, "not found")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    weight_extensions = {".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".onnx", ".gguf", ".pkl", ".pickle"}
    if download:
        return FileResponse(
            p,
            filename=p.name,
            media_type="application/octet-stream",
            content_disposition_type="attachment",
        )
    if p.suffix.lower() in weight_extensions:
        raise HTTPException(400, "checkpoint and weight files can only be downloaded")
    text_extensions = {
        ".txt", ".log", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
        ".md", ".py", ".sh", ".toml", ".ini", ".cfg", ".xml", ".html", ".htm",
    }
    media_type = "text/plain; charset=utf-8" if p.suffix.lower() in text_extensions else None
    if media_type is None:
        media_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return FileResponse(
        p,
        filename=p.name,
        media_type=media_type,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


class PreviewReq(BaseModel):
    config: Dict[str, Any]


@app.post("/api/preview_command")
def api_preview_command(req: PreviewReq):
    try:
        preview_path = Path(tempfile.gettempdir()) / "diffsynth_train_ui_preview"
        argv, output_path, log_path = runner.build_command(req.config, preview_path)
    except Exception as e:
        raise HTTPException(400, f"build_command failed: {e}")
    return {"argv": argv, "output_path": output_path, "log_path": log_path}


@app.get("/api/settings")
def api_get_settings():
    public = settings_core.get_public()
    return {**public, "keys": settings_core.SETTING_KEYS}


class SettingsReq(BaseModel):
    settings: Dict[str, str]


@app.put("/api/settings")
def api_set_settings(req: SettingsReq):
    try:
        settings_core.set_many(req.settings or {})
        settings_core.apply_path_settings()
    except OSError as e:
        raise HTTPException(400, f"path is not writable: {e}")
    return settings_core.get_public()


@app.delete("/api/settings/dashscope_api_key")
def api_clear_dashscope_api_key():
    settings_core.clear_secret("DASHSCOPE_API_KEY")
    return settings_core.get_public()


@app.get("/api/health")
def health():
    return {"ok": True}
