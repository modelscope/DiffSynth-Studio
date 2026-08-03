from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import jobs as job_core

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".gif"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_SAMPLE_EXTS = _IMG_EXTS | _VIDEO_EXTS | _AUDIO_EXTS
_CKPT_EXTS = {".safetensors", ".pt", ".pth", ".bin"}
_INTERNAL_FILES = {".train_exit_code"}
_LOSS_PATTERN = re.compile(
    r"(?:\bstep\s*[:=]\s*(\d+)[^\r\n]*?)?"
    r"\bloss\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.I,
)


def _output_dir(job_id: str) -> Path:
    job = job_core.get_job(job_id)
    if not job.output_path:
        raise FileNotFoundError("job has no output_path yet")
    p = Path(job.output_path)
    if not p.exists():
        raise FileNotFoundError(f"output dir not exist: {p}")
    return p


def list_checkpoints(job_id: str) -> List[Dict[str, Any]]:
    try:
        d = _output_dir(job_id)
    except FileNotFoundError:
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in _CKPT_EXTS:
            out.append({
                "name": p.name,
                "rel_path": str(p.relative_to(d)),
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    return out


def list_samples(job_id: str) -> List[Dict[str, Any]]:
    try:
        d = _output_dir(job_id)
    except FileNotFoundError:
        return []
    sample_root = d / "final_samples"
    if not sample_root.is_dir():
        return []
    run = job_core.get_job(job_id).latest_run
    prompts = (run.config.get("sample_prompts") if run else None) or []
    if isinstance(prompts, str):
        prompts = prompts.splitlines()
    prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    out: List[Dict[str, Any]] = []
    for p in sorted(sample_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _SAMPLE_EXTS:
            rel = str(p.relative_to(d))
            out.append({
                "name": p.name,
                "rel_path": rel,
                "mtime": p.stat().st_mtime,
                "kind": (
                    "image" if p.suffix.lower() in _IMG_EXTS
                    else "video" if p.suffix.lower() in _VIDEO_EXTS
                    else "audio"
                ),
                "prompt": prompts[len(out)] if len(out) < len(prompts) else None,
            })
    return out


def read_sampling_status(job_id: str) -> Dict[str, Any]:
    run = job_core.get_job(job_id).latest_run
    if not run:
        return {"status": "not_started", "outputs": []}
    return {
        "status": run.sampling_status,
        "pid": run.sampling_pid,
        "returncode": run.sampling_returncode,
        "current": run.sampling_current,
        "total": run.sampling_total,
        "checkpoint": run.sampling_checkpoint,
        "validate_script": run.sampling_script,
        "message": run.sampling_message,
        "started_at": run.sampling_started_at,
        "finished_at": run.sampling_finished_at,
        "outputs": [],
    }


def list_files(job_id: str) -> List[Dict[str, Any]]:
    try:
        d = _output_dir(job_id)
    except FileNotFoundError:
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name not in _INTERNAL_FILES:
            out.append({
                "name": p.name,
                "rel_path": str(p.relative_to(d)),
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    return out


def read_artifact(job_id: str, rel_path: str) -> Path:
    d = _output_dir(job_id)
    p = (d / rel_path).resolve()
    if not str(p).startswith(str(d.resolve())):
        raise PermissionError("path escapes job output dir")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def read_loss(job_id: str) -> List[Dict[str, Any]]:
    try:
        d = _output_dir(job_id)
    except FileNotFoundError:
        return []
    series: List[Dict[str, Any]] = []

    csv_path = d / "loss.csv"
    if csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        item: Dict[str, Any] = {}
                        if "step" in row: item["step"] = int(float(row["step"]))
                        if "loss" in row: item["loss"] = float(row["loss"])
                        if "lr" in row and row["lr"]: item["lr"] = float(row["lr"])
                        if "step" in item and math.isfinite(item.get("loss", math.nan)):
                            series.append(item)
                    except Exception:
                        continue
            if series:
                return series
        except Exception:
            pass